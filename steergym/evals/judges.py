"""Local digit-token judge panel: a small instruct model (gemma-2-2b-it in
all runs of record) rates (question, response) pairs 0-5 by reading the
grader's own next-token distribution over digit tokens — the score is the
expectation under the renormalized digit distribution at the first output
position, so it is continuous and needs no sampling or parsing.

These judges are generic measurement tools, deliberately NOT returned by any
benchmark's evals(): the paper-defined evals live with their benchmarks, and
runscripts assemble this panel explicitly where a local (API-free) judge is
wanted. Known limit: a small judge scores content it disapproves of as
incoherent, so coherency numbers here are comparable within a run, never
across judges or against API-judged results without calibration.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import polars as pl

from ..concept import Concept
from .base import Eval, EvalConfig

COHERENCY_PROMPT = """Consider the following question and test-takers response. Rate the coherency of the response from 0-5.

QUESTION
```
{question}
```

RESPONSE
```
{resp}
```

Reply with numeric score only."""


TAIL_PROMPT = """Consider the following question and test-takers response. The response may be cut off mid-sentence at the end due to a length limit - that alone is FINE and should not be penalized. Ignoring simple mid-sentence cutoff, rate degradation in the final portion of the response: 0 = stays well-formed to the end, 1 = some degradation (repetition, filler characters, or fragments unrelated to the main content), 2 = severe degradation.

QUESTION
```
{question}
```

RESPONSE
```
{resp}
```

Reply with the single digit only."""


GARBLE_PROMPT = """Consider the following question and test-takers response. Count instances of garbled text in the response (corrupted words, runs of stray symbols, repeated filler, malformed or invented-looking entity names): 0 = none, 1 = a few, 2 = many.

QUESTION
```
{question}
```

RESPONSE
```
{resp}
```

Reply with the single digit only."""


# Concept-agnostic coherency panel: {prompt, n_digits, invert} per judge; each
# judge's score is normalized onto 0-5 (inverted for the corruption counters,
# whose scales are small and discrete by design — detection, not rating) and
# averaged. coh05 is the base quality signal; tail012 catches end-of-response
# degradation; garble012 counts garbled-text instances.
COHERENCY_PANEL = [
    {"prompt": COHERENCY_PROMPT, "n_digits": 6, "invert": False},
    {"prompt": TAIL_PROMPT, "n_digits": 3, "invert": True},
    {"prompt": GARBLE_PROMPT, "n_digits": 3, "invert": True},
]


ACCURACY_PROMPT = """Consider the following question and test-takers response. Rate from 0-5 the extent to which the responder mentions or discusses `{concept}` (0 = not mentioned at all, or a different subject is discussed instead; 5 = the response is clearly about `{concept}`).

QUESTION
```
{question}
```

RESPONSE
```
{resp}
```

Reply with numeric score only."""


# "displays or expresses the concept" — vs ACCURACY_PROMPT's "mentions or
# discusses". Module-level so extraction-time filters can share one rubric.
EXPRESSION_PROMPT_TEMPLATE = """{definition}Consider the following question and test-taker's response. Rate from 0-5 the extent to which the response itself displays or expresses this trait/concept in its own tone, content, and behavior (0 = not at all; 5 = strongly and consistently). Note: merely mentioning or discussing the concept is NOT expressing it -- score by what the response DOES, not what it talks about.

QUESTION
```
{question}
```

RESPONSE
```
{resp}
```

Reply with numeric score only."""


def fill_expression_prompt(concept: str, question: str, resp: str, description: str | None = None) -> str:
    prefix = f"{description}\n\n" if description else f"Concept/trait: `{concept}`.\n\n"
    return EXPRESSION_PROMPT_TEMPLATE.format(definition=prefix, question=question, resp=resp)


def _fold_system(messages):
    # standard fallback for templates that reject the system role (e.g.
    # gemma-2): fold it into the first user turn; None if nothing to fold
    if not messages or messages[0]["role"] != "system":
        return None
    return [
        {"role": "user",
         "content": messages[0]["content"] + "\n\n" + messages[1]["content"]},
    ] + messages[2:]


def chat_prompt(tokenizer, text: str) -> str:
    # tolerant across chat-template families: `enable_thinking` is a Qwen3
    # kwarg (unknown elsewhere — some tokenizers TypeError on it, others
    # ignore it), and some templates reject the system role outright
    messages = [{"role": "user", "content": text}]
    variants = [m for m in (messages, _fold_system(messages)) if m is not None]
    last = None
    for msgs in variants:
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **kwargs)
            except Exception as e:
                last = e
    raise last


def judge_ctx(model, tokenizer, coherency_prompts: list[dict] | None = None) -> SimpleNamespace:
    """Precomputed digit-token ids/values + the coherency panel. Passing
    ctx=None to any judge builds one implicitly; construct it once and reuse
    when judging many batches (or to override the panel)."""
    import torch

    return SimpleNamespace(
        digit_ids=[tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(6)],
        digit_vals=torch.tensor([float(i) for i in range(6)], device=model.device),
        coherency_prompts=list(coherency_prompts) if coherency_prompts else list(COHERENCY_PANEL))


def digit_judge(model, tokenizer, ctx: SimpleNamespace | None, filled_prompts: list[str],
                batch_size: int = 8, n_digits: int = 6) -> list[float]:
    """Expected score under the grader's (renormalized) distribution over
    digit tokens 0..n_digits-1 at its first output position."""
    import torch
    import torch.nn.functional as F

    if ctx is None:
        ctx = judge_ctx(model, tokenizer)
    prompts = [chat_prompt(tokenizer, p) for p in filled_prompts]
    scores = []
    for i in range(0, len(prompts), batch_size):
        inputs = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :].float()
        p = F.softmax(logits[:, ctx.digit_ids[:n_digits]], dim=-1)
        scores.extend((p @ ctx.digit_vals[:n_digits]).tolist())
    return scores


def coherency_scores(model, tokenizer, ctx: SimpleNamespace | None, questions: list[str],
                     responses: list[str], batch_size: int = 8) -> list[float]:
    """Judge-panel mean on a 0-5 scale (ctx.coherency_prompts entries are
    {prompt, n_digits, invert}; inverted judges count corruption, so 0 is
    best)."""
    if ctx is None:
        ctx = judge_ctx(model, tokenizer)
    panel = []
    for j in ctx.coherency_prompts:
        s = digit_judge(model, tokenizer, ctx,
                        [j["prompt"].format(question=q, resp=r) for q, r in zip(questions, responses)],
                        batch_size=batch_size, n_digits=j["n_digits"])
        hi = j["n_digits"] - 1
        panel.append([((hi - x) if j["invert"] else x) * 5.0 / hi for x in s])
    return [sum(col) / len(col) for col in zip(*panel)]


@dataclass
class JudgedEvalConfig(EvalConfig):
    judge_batch: int = 8


class JudgedEval(Eval):
    """Local-GPU judge panel. One live judge shared by every subclass:
    bind_judge once (a loaded model — a resource, so bind, not config)."""

    execution = "gpu"
    config_cls = JudgedEvalConfig
    judge = None  # (model, tokenizer, ctx)

    @classmethod
    def bind_judge(cls, model, tokenizer, ctx=None):
        JudgedEval.judge = (model, tokenizer, ctx if ctx is not None else judge_ctx(model, tokenizer))

    def _judge(self):
        if JudgedEval.judge is None:
            raise RuntimeError(f"{self.name}: judge not set — JudgedEval.bind_judge(...) first")
        return JudgedEval.judge


class CoherencyEval(JudgedEval):
    def score(self, rows: pl.DataFrame, concepts: dict) -> dict[str, list]:
        model, tok, ctx = self._judge()
        scores = coherency_scores(model, tok, ctx,
                                  rows.get_column("prompt").to_list(),
                                  rows.get_column("response").to_list(),
                                  batch_size=self.config.judge_batch)
        return {"coh": [float(s) for s in scores]}


class MentionsEval(JudgedEval):
    """Does the response mention/discuss the concept (ACCURACY_PROMPT, 0-5).
    Metric key "acc"."""

    def score(self, rows: pl.DataFrame, concepts: dict) -> dict[str, list]:
        model, tok, ctx = self._judge()
        prompts = [ACCURACY_PROMPT.format(concept=concepts[cid].name, question=q, resp=r)
                   for cid, q, r in zip(rows.get_column("concept_id").to_list(),
                                        rows.get_column("prompt").to_list(),
                                        rows.get_column("response").to_list())]
        scores = digit_judge(model, tok, ctx, prompts, batch_size=self.config.judge_batch)
        return {"acc": [float(s) for s in scores]}


class ExpressionEval(JudgedEval):
    """Does the response itself display/express the concept (0-5); the
    Concept.description is the definition paragraph. Metric key "expr"."""

    def score(self, rows: pl.DataFrame, concepts: dict) -> dict[str, list]:
        model, tok, ctx = self._judge()
        prompts = [fill_expression_prompt(concepts[cid].name, q, r,
                                          description=concepts[cid].description)
                   for cid, q, r in zip(rows.get_column("concept_id").to_list(),
                                        rows.get_column("prompt").to_list(),
                                        rows.get_column("response").to_list())]
        scores = digit_judge(model, tok, ctx, prompts, batch_size=self.config.judge_batch)
        return {"expr": [float(s) for s in scores]}
