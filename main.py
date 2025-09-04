from dotenv import load_dotenv
import time
load_dotenv()

import argparse
import asyncio
import logging
import re
import json
from statistics import mean
from datetime import datetime
from typing import Literal, Any

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    RefreshingBucketRateLimiter,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


class FallTemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    This bot is what is used by Metaculus in our benchmark, but is also provided as a template for new bot makers.
    This template is given as-is, and though we have covered most test cases
    in forecasting-tools it may be worth double checking key components locally.

    Main changes since Q2:
    - An LLM now parses the final forecast output (rather than programmatic parsing)
    - Added resolution criteria and fine print explicitly to the research prompt
    - Previously in the prompt, nothing about upper/lower bound was shown when the bounds were open. Now a suggestion is made when this is the case.
    - Support for nominal bounds was added (i.e. when there are discrete questions and normal upper/lower bounds are not as intuitive)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ones.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions for the
    MiniBench and Seasonal AIB tournaments. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/deep-research/low",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/deep-research/low":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "model_name").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # Dynamic research and ensemble configuration
    _simple_research_threshold = 6  # 1-10 difficulty scale
    _weight_strong = 2  # weight for powerful model (o3)
    _weight_efficient = 1  # weight for efficient model (gpt-4o-mini)
    _n_strong = 3
    _n_efficient = 4
    # AskNews free tier: 1 request per 10 seconds
    _asknews_min_interval_sec = 10

    # Utility: safe LLM getter
    def _get_llm_safe(self, key: str) -> Any:
        try:
            return self.get_llm(key, "llm")
        except Exception:
            return self.get_llm("parser", "llm")

    async def _assess_difficulty(self, question: MetaculusQuestion) -> int:
        assessor = self._get_llm_safe("assessor")
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = clean_indents(
            f"""
            Rate the forecasting research difficulty of the question on a 1-10 scale.
            1-3: straightforward; 4-6: moderate; 7-10: complex/ambiguous.
            Output only an integer 1-10.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Today: {today}
            """
        )
        try:
            text = await assessor.invoke(prompt)
            m = re.search(r"([1-9]|10)\b", text)
            score = int(m.group(1)) if m else 5
        except Exception:
            score = 5
        score = max(1, min(10, score))
        logger.info(f"Complexity score for {question.page_url}: {score}")
        return score

    async def _decompose_subqueries(self, question: MetaculusQuestion) -> list[str]:
        decomposer = self._get_llm_safe("decomposer")
        prompt = clean_indents(
            f"""
            Decompose the forecast question into 3-6 distinct sub-queries for research.
            Return ONLY a numbered list; each line is a concise search query.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Background: {question.background_info}
            """
        )
        try:
            text = await decomposer.invoke(prompt)
            lines = [re.sub(r"^\s*\d+\.?\s*", "", ln).strip(" -•") for ln in text.splitlines()]
            subs = [ln for ln in lines if ln]
            if len(subs) < 3:
                subs = [question.question_text]
        except Exception:
            subs = [question.question_text]
        logger.info(f"Sub-queries for {question.page_url}: {subs}")
        return subs

    async def _summarize_research(self, question: MetaculusQuestion, raw_sections: list[str]) -> str:
        summarizer = self._get_llm_safe("summarizer")
        today = datetime.now().strftime("%Y-%m-%d")
        combined = "\n\n".join([s for s in raw_sections if s])
        if not combined:
            return ""
        prompt = clean_indents(
            f"""
            Synthesize the following findings into a concise, structured brief for a forecaster.
            Include: executive summary; resolution snapshot; outside view/base rates; current signals; 
            decomposition/bottlenecks; status quo & trend lines; pitfalls in criteria; watchlist & cadence; sources.
            600-900 words. No probabilities.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Background: {question.background_info}
            Today: {today}

            Findings:
            ---
            {combined}
            ---
            """
        )
        try:
            return await summarizer.invoke(prompt)
        except Exception:
            return combined[:6000]

    @staticmethod
    def _drop_min_max(values: list[tuple[float, str]]) -> list[tuple[float, str]]:
        if len(values) <= 2:
            return values
        sorted_vals = sorted(values, key=lambda x: x[0])
        return sorted_vals[1:-1]

    def _weighted_average(self, values: list[tuple[float, str]]) -> float:
        if not values:
            return 0.5
        total_weight = 0.0
        weighted_sum = 0.0
        for val, src in values:
            w = self._weight_strong if src == "strong" else self._weight_efficient
            weighted_sum += w * val
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else mean([v for v, _ in values])

    async def _invoke_many(self, llm, prompt: str, n: int) -> list[str]:
        tasks = [llm.invoke(prompt) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def run_research(self, question: MetaculusQuestion) -> str:
        # Dynamic research: difficulty -> standard/enhanced AskNews -> synthesis
        await asyncio.sleep(11)  # brief pause for AskNews rate limits
        async with self._concurrency_limiter:
            difficulty = await self._assess_difficulty(question)
            asknews = AskNewsSearcher()
            sections: list[str] = []
            if difficulty <= self._simple_research_threshold:
                try:
                    brief = await asknews.get_formatted_news_async(
                        question.question_text
                    )
                    # Free tier: pause between calls
                    await asyncio.sleep(self._asknews_min_interval_sec)
                except Exception as e:
                    logger.warning(f"AskNews quick search failed: {e}")
                    # Respect Retry-After header when present
                    retry_after = None
                    try:
                        headers = getattr(getattr(e, "response", None), "headers", {})
                        retry_after = headers.get("retry-after") or headers.get("Retry-After")
                    except Exception:
                        pass
                    wait_s = None
                    try:
                        wait_s = float(retry_after) if retry_after is not None else None
                    except Exception:
                        wait_s = None
                    if wait_s is None:
                        wait_s = self._asknews_min_interval_sec + 2
                    await asyncio.sleep(wait_s)
                    try:
                        brief = await asknews.get_formatted_news_async(
                            question.question_text
                        )
                        await asyncio.sleep(self._asknews_min_interval_sec)
                    except Exception as e2:
                        logger.warning(f"AskNews quick search retry failed: {e2}")
                        brief = ""
                sections.append(brief)
            else:
                subqs = await self._decompose_subqueries(question)
                # Cap to reduce request volume on free tier
                subqs = subqs[:3]
                # Run subqueries sequentially with light backoff to avoid AskNews 429s
                for q in subqs:
                    try:
                        r = await asknews.get_formatted_news_async(q)
                        sections.append(f"Subquery: {q}\n{r}")
                        await asyncio.sleep(self._asknews_min_interval_sec)
                    except Exception as e:
                        msg = str(e)
                        logger.warning(f"AskNews subquery failed [{q}]: {e}")
                        # If concurrency or rate limit, wait and retry once
                        if any(tok in msg for tok in ("429", "Rate Limit", "ConcurrencyLimit")):
                            # Try to honor Retry-After header if available
                            retry_after = None
                            try:
                                headers = getattr(getattr(e, "response", None), "headers", {})
                                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                            except Exception:
                                pass
                            wait_s = None
                            try:
                                wait_s = float(retry_after) if retry_after is not None else None
                            except Exception:
                                wait_s = None
                            await asyncio.sleep(wait_s if wait_s is not None else (self._asknews_min_interval_sec + 2))
                            try:
                                r = await asknews.get_formatted_news_async(q)
                                sections.append(f"Subquery: {q}\n{r}")
                                await asyncio.sleep(self._asknews_min_interval_sec)
                            except Exception as e2:
                                logger.warning(f"AskNews retry failed [{q}]: {e2}")
            research = await self._summarize_research(question, sections)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research
        await asyncio.sleep(11)  # Force a 11-second pause to avoid hitting AskNews rate limits
        async with self._concurrency_limiter:
            today = datetime.now().strftime("%Y-%m-%d")
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.

                Objectives
                1) Produce a concise, sourced brief that surfaces the strongest, freshest evidence.
                2) Apply outside-view base rates and reference classes.
                3) Decompose the question into tractable sub-questions.
                4) Identify “if resolved today” outcome under the stated criteria.
                5) List update triggers to monitor.

                Inputs
                - Question: {question.question_text}
                - Resolution criteria: {question.resolution_criteria}
                - Fine print: {question.fine_print}
                - Background (optional): {question.background_info}
                - Today’s date (ISO): {today}

                Method (follow in order; keep each section compact)
                A. Resolution snapshot
                   - Extract deadlines, time zone, actors, thresholds, exclusions. Quote exact text for any edge case.
                   - State: “If resolved today under these criteria: {{Yes|No|Undetermined}}.” Give a one‑sentence justification tied to the criteria.

                B. Outside view (reference classes)
                   - Define a relevant reference class and cite its empirical base rate(s). If multiple plausible classes, list 2–3 and justify your choice.
                   - Convert base rates into a by‑deadline prior using hazard/survival framing when appropriate (e.g., “by {{date}}” questions). No subjective adjustment here. Cite sources.

                C. Current signals (inside view)
                   - Bullet the 5–12 most probative, **dated** items since {{a relevant start date}}. For each: Source, date, claim, measured indicator, reliability note.
                   - Prefer primary documents, official data releases, and high‑quality outlets. Avoid single‑source rumors.

                D. Decomposition & bottlenecks
                   - Break into 3–6 binary sub‑events or milestones. For each: current status, blocking constraint, earliest plausible date, actor with agency.

                E. Status quo & trend lines
                   - State the inertial path (what happens if actors take no unusual action).
                   - Include any simple, transparent trend (time series, backlog, vote count, court docket pace, shipment cadence). If numeric, specify units and last update.

                F. Resolution pitfalls
                   - List landmines in the rules: ambiguous wording, conflicting clocks, jurisdictional issues, data lags. Quote relevant clause(s).

                G. Watchlist & update cadence
                   - “Triggers to watch next” with expected sign of impact (↑Yes/↑No) and typical lag to resolution.
                   - Suggest a cadence (e.g., daily until vote; weekly otherwise) and authoritative feeds to poll.

                H. Sources
                   - Inline numeric citations [1], [2], … Then list full references with titles and dates. Include working links.

                Output format
                1) 6–10 bullet executive summary (max 120 words).
                2) Sections A–H as above.
                3) Total length target: 600–900 words. No forecasts, no probabilities.
                4) Use 1–2 short quotes from the criteria only; otherwise paraphrase.
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif researcher == "asknews/deep-research/low":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews"],
                    model="deepseek-basic",
                    search_depth=1,
                    max_depth=1,
                )
                await asyncio.sleep(self._asknews_min_interval_sec)
            elif researcher == "asknews/deep-research/medium-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews"],
                    model="deepseek-basic",
                    search_depth=1,
                    max_depth=1,
                )
                await asyncio.sleep(self._asknews_min_interval_sec)
            elif researcher == "asknews/deep-research/high-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews"],
                    model="deepseek-basic",
                    search_depth=1,
                    max_depth=1,
                )
                await asyncio.sleep(self._asknews_min_interval_sec)
            # elif researcher.startswith("smart-searcher"):
            #     model_name = researcher.removeprefix("smart-searcher/")
            #     searcher = SmartSearcher(
            #         model=model_name,
            #         temperature=0,
            #         num_searches_to_run=2,
            #         num_sites_per_search=10,
            #         use_advanced_filters=False,
            #     )
                # research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}
            {question.fine_print}

            Research assistant says:
            {research}

            Today is {today}.

            Constraints
            - Optimize for Brier score. Avoid 0% and 100%. Use 1%–99% bounds.
            - Weight the status quo appropriately; most systems change slowly absent strong catalysts.
            - Update in **small, frequent** increments when new evidence arrives.

            Pre‑write (one line each):
            (a) Time remaining until resolution.
            (b) Status‑quo outcome if nothing changes.
            (c) One credible path to “No.”
            (d) One credible path to “Yes.”

            Method (stepwise, terse)
            1) Clarify the rules
               - Restate the key threshold(s), authority, time zone, and deadline in one compact line. Flag any landmines.

            2) Outside view prior
               - Choose and name a reference class. State its base rate over an equivalent horizon.
               - If “by {{date}}” style, express as a simple hazard or survival probability.
               - Set your initial prior from this outside view (state the number).

            3) Evidence‑based updates (odds form)
               - Convert prior p to log‑odds L.
               - For each major evidence item (max 6), assign a rough likelihood ratio (LR) and update: L := L + ln(LR). Keep LR estimates conservative; justify them in a phrase.
               - Convert final L back to probability p*. Cite the 2–5 most load‑bearing sources inline [1]… and list them at the end.

            4) Coherence check
               - Is p* consistent with base rates, trend pace, and remaining time? If not, state the discrepancy and adjust once, modestly.

            5) Sensitivity
               - Name the smallest single development that would move your probability ≥10 percentage points and in which direction.

            6) Final required outputs
               - Pre‑write items (a)–(d).
               - 4–8 sentence rationale, referencing your outside view, the largest LR(s), and the status quo weight.
               - Then on a new line: **Probability: ZZ%**

            Formatting
            - Use bullets or short sentences. Include [1]–[N] citations and a brief references list with links and dates.
            - Keep total length under 300 words.
            """
        )
        # Ensemble: 3x strong (o3), 4x efficient (gpt-4o-mini), drop min/max, weight strong x2
        strong_llm = self._get_llm_safe("default")
        cheap_llm = self._get_llm_safe("efficient")
        parser_llm = self._get_llm_safe("parser")

        strong_texts, cheap_texts = await asyncio.gather(
            self._invoke_many(strong_llm, prompt, self._n_strong),
            self._invoke_many(cheap_llm, prompt, self._n_efficient),
        )

        votes: list[tuple[float, str]] = []
        panel_reasoning: list[str] = []
        for tag, texts in (("strong", strong_texts), ("efficient", cheap_texts)):
            for txt in texts:
                try:
                    pred: BinaryPrediction = await structure_output(
                        txt, BinaryPrediction, model=parser_llm
                    )
                    p = max(0.01, min(0.99, float(pred.prediction_in_decimal)))
                    votes.append((p, tag))
                    panel_reasoning.append(f"[{tag}] {p:.2f}: {txt[:250]}")
                except Exception as e:
                    logger.warning(f"Binary parse failed: {e}")

        trimmed = self._drop_min_max(votes) if len(votes) >= 3 else votes
        final_p = self._weighted_average(trimmed)

        reasoning = (
            "Panel forecasts (7), drop min/max, weight strong x2.\n" +
            "\n".join(panel_reasoning) +
            f"\n\nFinal probability: {final_p:.3f}"
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {final_p}"
        )
        return ReasonedPrediction(prediction_value=final_p, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = clean_indents(
            f"""
           You are a professional forecaster interviewing for a job.

            Your interview question:
            {question.question_text}

            Options (mutually exclusive and exhaustive, in this order):
            {question.options}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}
            {question.fine_print}

            Research assistant says:
            {research}

            Today is {today}.

            Scoring guardrails
            - Metaculus uses log scores. Never assign 0% or 100%. Keep tiny but nonzero mass on low‑probability options. Sum to 100%. [Platform-specific]
            - Favor small, frequent belief updates over big swings.

            Pre‑write (one line each):
            (a) Time remaining until resolution.
            (b) Status‑quo option if nothing changes and why.
            (c) One credible path to an unexpected (non‑status‑quo) option.

            Method (terse, stepwise)
            1) Clarify the rules
               - Extract the operative authority, deadline, time zone, and any tie‑breakers. Note landmines.

            2) Outside‑view priors (reference classes)
               - Map each option to a defensible reference class. Use empirical base rates over a matching horizon.
               - Convert base rates into a prior vector via a Dirichlet prior α (default α_i=1 if no data). State the resulting prior p_i.

            3) Evidence‑based updating (normalize across options)
               - List the top 3–6 load‑bearing evidence items with dates and sources.
               - For each item j, assign option‑specific likelihood multipliers L_{{ij}}. Update: p_i ← p_i × L_{{ij}}; then renormalize across i.
               - Keep L values conservative; avoid double‑counting correlated cues.

            4) Coherence checks
               - Do probabilities sum to 100%? Any option <0.1% or >99.9% without overwhelming evidence? Does the distribution overweight stories vs. base rates?

            5) Sensitivity
               - Name the smallest plausible development that would shift ≥10 percentage points from the current leader; give direction.

            6) Output
               - Provide a 4–8 sentence rationale referencing (i) outside‑view priors, (ii) the highest‑weight evidence, and (iii) the status‑quo bias toward inertia.
               - Then print final probabilities for {len(question.options)} options in exactly this order {question.options}:
                 Option_1: P1%
                 Option_2: P2%
                 …
                 Option_N: PN%
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        # Ensemble: 3x strong, 4x efficient; per-option drop min/max, weight strong x2
        strong_llm = self._get_llm_safe("default")
        cheap_llm = self._get_llm_safe("efficient")
        parser_llm = self._get_llm_safe("parser")

        strong_texts, cheap_texts = await asyncio.gather(
            self._invoke_many(strong_llm, prompt, self._n_strong),
            self._invoke_many(cheap_llm, prompt, self._n_efficient),
        )

        async def extract_probs(text: str) -> dict[str, float] | None:
            try:
                pol: PredictedOptionList = await structure_output(
                    text_to_structure=text,
                    output_type=PredictedOptionList,
                    model=parser_llm,
                    additional_instructions=parsing_instructions,
                )
                mapping: dict[str, float] | None = None
                for attr in ("option_probabilities", "probabilities", "options", "items"):
                    if hasattr(pol, attr):
                        val = getattr(pol, attr)
                        if isinstance(val, dict):
                            mapping = {str(k): float(v) for k, v in val.items()}
                            break
                if mapping is None:
                    pairs: dict[str, float] = {}
                    for line in text.splitlines():
                        m = re.match(r"\s*(.+?):\s*([0-9]+(?:\.[0-9]+)?)%\s*$", line)
                        if m:
                            key = m.group(1).strip()
                            val = float(m.group(2)) / 100.0
                            pairs[key] = val
                    mapping = {opt: pairs.get(opt, 0.0) for opt in question.options}
                s = sum(mapping.values())
                if s == 0:
                    return None
                if s > 1.5:
                    mapping = {k: v / 100.0 for k, v in mapping.items()}
                    s = sum(mapping.values())
                if abs(s - 1.0) > 1e-6:
                    mapping = {k: v / s for k, v in mapping.items()}
                return mapping
            except Exception as e:
                logger.warning(f"MC parse failed: {e}")
                return None

        panel: list[tuple[dict[str, float], str]] = []
        panel_reasoning: list[str] = []
        for tag, texts in (("strong", strong_texts), ("efficient", cheap_texts)):
            for txt in texts:
                probs = await extract_probs(txt)
                if probs:
                    panel.append((probs, tag))
                    preview = ", ".join([f"{k}:{v:.2f}" for k, v in list(probs.items())[:3]])
                    panel_reasoning.append(f"[{tag}] {preview}")

        # Aggregate per option
        agg: dict[str, float] = {}
        for opt in question.options:
            series: list[tuple[float, str]] = []
            for probs, tag in panel:
                if opt in probs:
                    series.append((probs[opt], tag))
            trimmed = self._drop_min_max(series) if len(series) >= 3 else series
            agg[opt] = self._weighted_average(trimmed)
        # Renormalize
        s = sum(agg.values())
        if s > 0:
            agg = {k: v / s for k, v in agg.items()}

        # Enforce Metaculus constraints: each in [0.001, 0.999] and sum to 1.0
        lo, hi = 0.001, 0.999
        opts = list(question.options)
        if not opts:
            return ReasonedPrediction(
                prediction_value=PredictedOptionList(predicted_options=[]),
                reasoning="No options provided",
            )
        p = [float(agg.get(opt, 0.0)) for opt in opts]
        # initial normalization
        s = sum(p)
        if s <= 0:
            p = [1.0 / len(opts)] * len(opts)
        else:
            p = [x / s for x in p]
        # clip to [lo, hi]
        p = [min(hi, max(lo, x)) for x in p]
        # project onto simplex with bounds via iterative water-filling
        for _ in range(6):
            deficit = 1.0 - sum(p)
            if abs(deficit) < 1e-9:
                break
            if deficit > 0:
                room = [max(0.0, hi - x) for x in p]
            else:
                room = [max(0.0, x - lo) for x in p]
            total_room = sum(room)
            if total_room <= 1e-12:
                break
            scale = deficit / total_room
            p = [min(hi, max(lo, x + scale * r)) for x, r in zip(p, room)]
        # final tiny adjustment on a free index if needed
        rem = 1.0 - sum(p)
        if abs(rem) > 1e-9:
            if rem > 0:
                idx = next((i for i, x in enumerate(p) if x < hi - 1e-9), None)
                if idx is not None:
                    p[idx] = min(hi, p[idx] + rem)
            else:
                idx = next((i for i, x in enumerate(p) if x > lo + 1e-9), None)
                if idx is not None:
                    p[idx] = max(lo, p[idx] + rem)
        smoothed = {opt: val for opt, val in zip(opts, p)}

        # Build PredictedOptionList directly to avoid parser variability
        predicted_option_list: PredictedOptionList = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=opt, probability=float(smoothed[opt]))
                for opt in question.options
            ]
        )
        reasoning = (
            "Panel forecasts (7), per-option drop min/max, weight strong x2.\n" +
            "\n".join(panel_reasoning)
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = clean_indents(
            f"""
           You are a professional forecaster interviewing for a job.

            Your interview question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}
            {question.fine_print}

            Units for answer:
            {question.unit_of_measure if question.unit_of_measure else "Not stated (infer and state explicitly)"}

            Research assistant says:
            {research}

            Today is {today}.

            Bounds guidance:
            {lower_bound_message}
            {upper_bound_message}

            Formatting rules
            - Use the stated units exactly. No scientific notation. List values in strictly increasing order.

            Pre‑write (one line each):
            (a) Time left until resolution.
            (b) Outcome if nothing changes (status‑quo level).
            (c) Outcome if the current trend simply continues (show simple extrapolation window).
            (d) Market or expert anchors (cite and time‑stamp).
            (e) One unexpected low‑tail scenario.
            (f) One unexpected high‑tail scenario.

            Method (terse, stepwise)
            1) Clarify the target
               - Define the measurement, inclusion/exclusion, effective time stamp, and scoring range. Note if resolution outside bounds converts to binary scoring.

            2) Choose scale and reference class
               - Pick linear vs. log scale based on domain:
                 • Log scale if the variable is strictly positive and multiplicative or skewed (prices, cases, GDP growth components).
                 • Linear if deviations are additive and roughly symmetric.
               - State a reference class and its empirical spread over a matching horizon; this sets the outside‑view prior.

            3) Construct a well‑calibrated 90% interval first
               - Set P10 and P90 wide enough for unknown unknowns. Then set the median (P50) anchored to status‑quo and market/expert benchmarks.

            4) Fill interior percentiles consistent with scale
               - For log‑scale, interpolate quantiles in log space; for linear, interpolate in natural units.
               - Ensure monotonicity and plausible spacing given trend volatility and structural breaks.

            5) Coherence and sharpness checks
               - Check tail probabilities vs. bounds; avoid unrealistically sharp peaks given data quality.
               - Round to sensible precision for the unit.

            Final output (exactly this):"
            Percentile 10: XX
            Percentile 20: XX
            Percentile 30: XX
            Percentile 40: XX
            Percentile 50: XX
            Percentile 60: XX
            Percentile 70: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        # Ensemble: 3x strong, 4x efficient; per-percentile drop min/max, weight strong x2
        strong_llm = self._get_llm_safe("default")
        cheap_llm = self._get_llm_safe("efficient")
        parser_llm = self._get_llm_safe("parser")

        strong_texts, cheap_texts = await asyncio.gather(
            self._invoke_many(strong_llm, prompt, self._n_strong),
            self._invoke_many(cheap_llm, prompt, self._n_efficient),
        )

        async def extract_percentiles(text: str) -> dict[int, float] | None:
            try:
                plist: list[Percentile] = await structure_output(
                    text, list[Percentile], model=parser_llm
                )
                pts: dict[int, float] = {}
                for p in plist:
                    if hasattr(p, "percentile") and hasattr(p, "value"):
                        pts[int(getattr(p, "percentile"))] = float(getattr(p, "value"))
                needed = {10, 20, 30, 40, 50, 60, 70, 80, 90}
                if not needed.issubset(set(pts.keys())):
                    return None
                return pts
            except Exception as e:
                logger.warning(f"Numeric parse failed: {e}")
                return None

        panel: list[tuple[dict[int, float], str]] = []
        for tag, texts in (("strong", strong_texts), ("efficient", cheap_texts)):
            for txt in texts:
                pts = await extract_percentiles(txt)
                if pts:
                    panel.append((pts, tag))

        combined: dict[int, float] = {}
        for q in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            series: list[tuple[float, str]] = []
            for pts, tag in panel:
                if q in pts:
                    series.append((pts[q], tag))
            trimmed = self._drop_min_max(series) if len(series) >= 3 else series
            combined[q] = self._weighted_average(trimmed)

        agg_text = "\n".join([f"Percentile {q}: {combined[q]}" for q in [10,20,30,40,50,60,70,80,90]])
        percentile_list: list[Percentile] = await structure_output(
            agg_text, list[Percentile], model=parser_llm
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        reasoning = (
            "Panel forecasts (7), per-percentile drop min/max, weight strong x2."
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/openai/o3",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openrouter/openai/gpt-4o-mini",
            "parser": "openrouter/openai/gpt-4o-mini",
            "efficient": "openrouter/openai/gpt-4o-mini",
            "assessor": "openrouter/openai/gpt-4o-mini",
            "decomposer": "openrouter/openai/o3",
            "researcher": "asknews/news-summaries",
        },
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/39700/french-pms-grouping-on-december-31-2025/"
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
