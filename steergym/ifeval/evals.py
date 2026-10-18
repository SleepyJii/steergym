import json
from dataclasses import dataclass

import polars as pl

from ..benchmark import meta_col
from ..evals.base import Eval, EvalConfig

# ifeval-local meta payload (this submodule is its only producer/consumer):
# eval rows carry meta.checker = {"instruction_id": str, "kwargs_json": str}.
# kwargs stay a JSON string — struct-typing 47 heterogeneous kwarg shapes
# across types isn't worth it — and must be the EFFECTIVE pack-build-time
# kwargs: build_description randomises unset ones.
CHECKER_META_KEY = "checker"


def _registry():
    # vendored files import each other as top-level `ifeval_scripts` (kept
    # verbatim), so the vendor dir goes on sys.path
    import sys
    from pathlib import Path
    vendor = str(Path(__file__).resolve().parent / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from ifeval_scripts import instructions_registry
    return instructions_registry


def check(instruction_id: str, kwargs: dict, response: str) -> tuple[bool, bool, bool]:
    """(strict, loose, checker_error)."""
    instructions_registry = _registry()
    error = False
    try:
        inst = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        inst.build_description(**(kwargs or {}))
        strict = bool(inst.check_following(response))
    except Exception:
        return False, False, True

    # loose (IFEval test_instruction_following_loose): any of {as-is, minus
    # first line, minus last line, minus both} x {with/without "*"} passes
    lines = response.split("\n")
    variants = [response, "\n".join(lines[1:]), "\n".join(lines[:-1]), "\n".join(lines[1:-1])]
    variants += [v.replace("*", "") for v in variants]
    loose = False
    for v in variants:
        try:
            if inst.check_following(v):
                loose = True
                break
        except Exception:
            error = True
    return strict, loose, error


def checker_payload(rows: pl.DataFrame) -> list[dict]:
    payloads = meta_col(rows, CHECKER_META_KEY).to_list()
    out = []
    for p in payloads:
        p = p or {}
        kwargs = json.loads(p["kwargs_json"]) if p.get("kwargs_json") else {}
        out.append({"instruction_id": p.get("instruction_id"), "kwargs": kwargs})
    return out


class IFEvalEval(Eval):
    """Programmatic expression: does the response satisfy its row's constraint.
    Reads meta.checker (CHECKER_META_KEY above)."""

    def score(self, rows: pl.DataFrame, concepts: dict) -> dict[str, list]:
        payload = checker_payload(rows)
        strict, loose, errors = [], [], []
        for p, resp in zip(payload, rows.get_column("response").to_list()):
            s, l, e = check(p["instruction_id"], p["kwargs"], str(resp))
            strict.append(s)
            loose.append(l)
            errors.append(e)
        return {"pass_strict": strict, "pass_loose": loose, "checker_error": errors}


@dataclass
class PerplexityEvalConfig(EvalConfig):
    model_id: str = "openai-community/gpt2"  # any transformers CausalLM
    threshold: float = 2.5  # Stolfo: 2.5 format instructions, 4 word-specific


class StolfoPerplexityEval(Eval):
    """Perplexity exactly as Stolfo et al. 2025's compute_perplexity (last
    1024 tokens, loss->exp, exception -> ppl 1.0). In the paper a degeneracy
    GATE, reported here as per-row columns so the gate stays reproducible."""

    execution = "gpu"
    config_cls = PerplexityEvalConfig

    def __init__(self, config: PerplexityEvalConfig | None = None):
        super().__init__(config)
        self._ppl_model = None
        self._ppl_tokenizer = None

    def _ensure_model(self):
        if self._ppl_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._ppl_tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self._ppl_model = AutoModelForCausalLM.from_pretrained(self.config.model_id).eval().cuda()

    def _ppl(self, text: str) -> float:
        import torch
        try:
            ids = self._ppl_tokenizer(text, return_tensors="pt").input_ids[:, -1024:]
            ids = ids.to(self._ppl_model.device)
            with torch.no_grad():
                loss = self._ppl_model(ids, labels=ids).loss
            return float(torch.exp(loss))
        except Exception:
            return 1.0  # Stolfo: exception -> loss 0 -> ppl 1.0

    def score(self, rows: pl.DataFrame, concepts: dict) -> dict[str, list]:
        self._ensure_model()
        ppls = [self._ppl(str(r)) for r in rows.get_column("response").to_list()]
        return {"ppl": ppls, "low_ppl": [p < self.config.threshold for p in ppls]}


@dataclass
class QualityEvalConfig(EvalConfig):
    judge_model: str = "gpt-4o"


class StolfoQualityEval(Eval):
    """Stolfo et al. 2025's two-stage quality judge (App. F). Stage A (cached
    per prompt) generates <=5 yes/no questions; Stage B answers each
    Yes/No/N-A; quality = #Yes/#questions with N/A in the denominator (the
    paper's literal reading; quality_frac_na reports how often that mattered).
    Real API spend — never enable by default."""

    execution = "api"
    api_key_env = "OPENAI_API_KEY"
    config_cls = QualityEvalConfig

    STAGE_A_PROMPT = (
        "We do not know how to evaluate the quality of model answers for "
        "this prompt. Can you come up with 5 of less questions that can "
        "break down the quality to simpler evaluation tasks that we can "
        "then ask about the model answer? Each question should have a "
        "simple yes, no answer.\n\nPrompt: {prompt}\n\n"
        "Return ONLY a JSON list of question strings."
    )
    STAGE_B_PROMPT = (
        "Prompt given to the model:\n{prompt}\n\n"
        "Model's response:\n{response}\n\n"
        "Questions to grade this response against:\n{questions}\n\n"
        "For each question, answer strictly 'Yes', 'No', or 'Not Applicable', "
        "with a brief reason. Return ONLY a JSON list of objects with keys "
        "\"question\", \"answer\", \"reason\" — one per question, same order."
    )

    def __init__(self, config: QualityEvalConfig | None = None):
        super().__init__(config)
        self._client = None
        self._stage_a_cache = {}

    def _ensure_client(self):
        if self._client is None:
            import os
            import openai
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError("StolfoQualityEval needs OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=key)
        return self._client

    @staticmethod
    def _extract_json(text: str):
        import re
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    @classmethod
    def _parse_questions(cls, text: str) -> list[str]:
        data = cls._extract_json(text)
        if isinstance(data, list) and data:
            return [str(q) for q in data][:5]
        lines = [l.strip(" -*0123456789.") for l in text.splitlines() if l.strip()]
        qs = [l for l in lines if l.endswith("?")]
        return (qs or lines)[:5]

    @classmethod
    def _parse_answers(cls, text: str) -> list[str]:
        data = cls._extract_json(text)
        if not isinstance(data, list):
            return []
        return [str(item.get("answer", "") if isinstance(item, dict) else item).strip().lower()
                for item in data]

    def _ask(self, content: str) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.config.judge_model, temperature=0,
            messages=[{"role": "user", "content": content}])
        return resp.choices[0].message.content or ""

    def _score_row(self, prompt: str, response: str) -> tuple[float, float]:
        if prompt not in self._stage_a_cache:
            self._stage_a_cache[prompt] = self._parse_questions(
                self._ask(self.STAGE_A_PROMPT.format(prompt=prompt)))
        questions = self._stage_a_cache[prompt]
        q_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        answers = self._parse_answers(self._ask(
            self.STAGE_B_PROMPT.format(prompt=prompt, response=response, questions=q_text)))
        n = len(answers) or 1
        n_yes = sum(1 for a in answers if a.startswith("yes"))
        n_na = sum(1 for a in answers if "applicable" in a or a.startswith("n/a"))
        return n_yes / n, n_na / n

    def score(self, rows: pl.DataFrame, concepts: dict) -> dict[str, list]:
        quality, frac_na = [], []
        for prompt, response in zip(rows.get_column("prompt").to_list(),
                                    rows.get_column("response").to_list()):
            try:
                s, na = self._score_row(str(prompt), str(response))
            except Exception:
                s, na = None, None
            quality.append(s)
            frac_na.append(na)
        return {"quality": quality, "quality_frac_na": frac_na}
