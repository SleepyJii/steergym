"""The AxBench steering judges (Wu et al. 2025): gpt-4o-mini rates each
steered generation on three 0-2 scales — concept relevance, instruction
relevance, fluency — with the verbatim upstream templates, one plain user
message per rating at temperature 1.0 (axbench LMJudgeEvaluator /
LanguageModel.chat_completion). Each judge is its own Eval producing its own
per-row column; the leaderboard aggregate is the STRICT harmonic mean of the
three (any 0 -> 0, `lm_judge_rating`), and the groupby-factor mean on top of
that is a runscript concern, like every other aggregation.

Not implemented: the AlpacaEvalSuppress flip (suppression runs score
2 - concept relevance) — Concept500Benchmark is the steering protocol only.

These scores are NOT comparable to the local digit-judge panel
(steergym.evals.judges): different judge, different scales, no calibration.
"""

import os
from dataclasses import dataclass

import polars as pl

from ..evals.base import Eval, EvalConfig

CONCEPT_RELEVANCE_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated. Point out references to the concept in the fragment, noting any \
minimal or awkward incorporations. After providing your explanation, rate the concept's \
relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, \
1 indicates the concept is somewhat present but minimally or awkwardly incorporated, \
and 2 indicates the concept is more fully and effectively incorporated, with stronger and \
more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

INSTRUCTION_RELEVANCE_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the \
sentence fragment provided below is related to the instruction. Focus solely \
on the degree of relatedness in terms of topic, regardless of grammar, coherence, or \
informativeness.

Begin your evaluation by providing a brief explanation of whether the \
sentence is related to the instruction, and point out references \
related to the instruction. After providing your explanation, rate the instruction \
relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the \
instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, \
and 2 indicates it is more clearly and directly related to the instruction. Provide your rating \
using this exact format: "Rating: [[score]]".

[Instruction Start]
{instruction}
[Instruction End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

FLUENCY_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the fluency of the \
sentence fragment provided below. Focus solely on fluency, disregarding \
its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any \
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that \
may hinder readability. After providing your explanation, rate the sentence's fluency \
on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural \
(e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable \
errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. \
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

DEFAULT_RATING = 0.0  # LMJudgeEvaluator.DEFAULT_RATING: unparseable/invalid -> 0


def parse_rating(completion: str, min_rating: float = 0.0, max_rating: float = 2.0) -> float:
    # LMJudgeEvaluator._get_rating_from_completion + the range check of
    # _get_ratings_from_completions, exactly: last "Rating:", first line,
    # brackets/quotes/asterisks/trailing-period stripped; anything that fails
    # to parse or falls outside [min, max] scores DEFAULT_RATING
    if "Rating:" not in completion:
        return DEFAULT_RATING
    text = completion.split("Rating:")[-1].strip()
    text = text.split("\n")[0].strip()
    text = text.replace("[", "").replace("]", "")
    text = text.rstrip(".").strip('"').strip("'").strip("*").strip()
    try:
        rating = float(text)
    except ValueError:
        return DEFAULT_RATING
    return rating if min_rating <= rating <= max_rating else DEFAULT_RATING


def harmonic_mean_strict(scores) -> float:
    # the leaderboard rule: any 0 -> 0 ("maintain strict evaluation")
    if 0 in scores:
        return 0.0
    return len(scores) / sum(1 / s for s in scores)


def lm_judge_rating(results: pl.DataFrame) -> pl.Series:
    """Per-row strict harmonic mean over the three judge columns — axbench's
    `lm_judge_rating`. Takes a results table that already carries
    relevance_concept / relevance_instruction / fluency."""
    return pl.Series("lm_judge_rating", [
        harmonic_mean_strict(triple) for triple in zip(
            results.get_column("relevance_concept").to_list(),
            results.get_column("relevance_instruction").to_list(),
            results.get_column("fluency").to_list())])


@dataclass
class AxbenchJudgeConfig(EvalConfig):
    judge_model: str = "gpt-4o-mini"   # axbench sweep configs' lm_model
    temperature: float = 1.0           # LanguageModel default
    base_url: str | None = None        # local OpenAI-compatible server to avoid spend


class _AxbenchLMJudgeEval(Eval):
    """One upstream rating per subclass. Serial sync calls (steergym scores
    rows serially; upstream batches async — same request payload per row:
    a single user message at config.temperature, no token cap)."""

    execution = "api"
    api_key_env = "OPENAI_API_KEY"
    config_cls = AxbenchJudgeConfig
    column = None

    def __init__(self, config: AxbenchJudgeConfig | None = None):
        super().__init__(config)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import openai
            key = os.environ.get(self.api_key_env, "")
            if not key:
                raise RuntimeError(f"{self.name} needs {self.api_key_env}")
            kwargs = {"api_key": key}
            base_url = self.config.base_url or os.environ.get("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _fill(self, rows: pl.DataFrame, concepts: dict) -> list[str]:
        raise NotImplementedError

    def score(self, rows: pl.DataFrame, concepts: dict) -> dict[str, list]:
        client = self._ensure_client()
        ratings = []
        for prompt in self._fill(rows, concepts):
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.judge_model, temperature=self.config.temperature)
            ratings.append(parse_rating(completion.choices[0].message.content.strip()))
        return {self.column: ratings}


class AxbenchConceptEval(_AxbenchLMJudgeEval):
    """Concept relevance 0-2 (concept text = Concept.name, axbench
    input_concept). Column "relevance_concept"."""

    column = "relevance_concept"

    def _fill(self, rows, concepts):
        return [CONCEPT_RELEVANCE_TEMPLATE.format(concept=concepts[cid].name, sentence=r)
                for cid, r in zip(rows.get_column("concept_id").to_list(),
                                  rows.get_column("response").to_list())]


class AxbenchInstructEval(_AxbenchLMJudgeEval):
    """Instruction relevance 0-2 (instruction = the steering prompt, axbench
    original_prompt). Column "relevance_instruction"."""

    column = "relevance_instruction"

    def _fill(self, rows, concepts):
        return [INSTRUCTION_RELEVANCE_TEMPLATE.format(instruction=q, sentence=r)
                for q, r in zip(rows.get_column("prompt").to_list(),
                                rows.get_column("response").to_list())]


class AxbenchFluencyEval(_AxbenchLMJudgeEval):
    """Fluency 0-2. Column "fluency"."""

    column = "fluency"

    def _fill(self, rows, concepts):
        return [FLUENCY_TEMPLATE.format(sentence=r)
                for r in rows.get_column("response").to_list()]
