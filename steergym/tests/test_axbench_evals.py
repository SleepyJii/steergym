#!/usr/bin/env python3
# Pure-python tests, no pytest. Run:
#   PYTHONPATH=<scratchpad>/pylibs python3 steergym/tests/test_axbench_evals.py
# No API spend: score() runs against a fake OpenAI-client double.
import os
import sys
import warnings
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from steergym.axbench.evals import (
    CONCEPT_RELEVANCE_TEMPLATE,
    AxbenchConceptEval,
    AxbenchFluencyEval,
    AxbenchInstructEval,
    harmonic_mean_strict,
    lm_judge_rating,
    parse_rating,
)
from steergym.concept import Concept
from steergym.evals.base import MissingAPIKeyWarning
from steergym.ifeval.evals import StolfoQualityEval
from steergym.personavec.evals import TraitEval

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


# ---------------------------------------------------------------- parse_rating
# faithful to axbench LMJudgeEvaluator._get_rating_from_completion (+ the
# range check): last "Rating:", first line, brackets/quotes/asterisks/trailing
# period stripped, out-of-range or unparseable -> 0.0
check("plain rating", parse_rating("explanation...\nRating: [[2]]") == 2.0)
check("last Rating wins", parse_rating("Rating: [[0]] is bad.\nRating: [[1]]") == 1.0)
check("first line after Rating", parse_rating("Rating: [[1.5]]\ngarbage [[2]]") == 1.5)
check("trailing period/quotes/asterisks stripped",
      parse_rating('Rating: "[[2]]".') == 2.0 and parse_rating("Rating: *[[1]]*") == 1.0)
check("no Rating marker -> default 0", parse_rating("the fragment is fluent") == 0.0)
check("unparseable -> default 0", parse_rating("Rating: [[two]]") == 0.0)
check("out of range -> default 0",
      parse_rating("Rating: [[5]]") == 0.0 and parse_rating("Rating: [[-1]]") == 0.0)
check("bare number without brackets", parse_rating("Rating: 2") == 2.0)

# ---------------------------------------------------------------- aggregation
check("harmonic mean strict: any zero -> 0", harmonic_mean_strict([0, 2, 2]) == 0.0)
check("harmonic mean of equal scores", harmonic_mean_strict([2, 2, 2]) == 2.0)
check("harmonic mean mixed", abs(harmonic_mean_strict([1, 2, 2]) - 1.5) < 1e-12)

df = pl.DataFrame({"relevance_concept": [2.0, 0.0], "relevance_instruction": [2.0, 2.0],
                   "fluency": [2.0, 2.0]})
agg = lm_judge_rating(df)
check("lm_judge_rating per-row strict harmonic mean",
      agg.to_list() == [2.0, 0.0] and agg.name == "lm_judge_rating")

# ---------------------------------------------------------------- key warning
os.environ.pop("OPENAI_API_KEY", None)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    AxbenchConceptEval(), TraitEval(), StolfoQualityEval()
check("api evals warn at construction when the key env is unset",
      sum(issubclass(w.category, MissingAPIKeyWarning) for w in caught) == 3)
check("warning names the env var", "OPENAI_API_KEY" in str(caught[0].message))

os.environ["OPENAI_API_KEY"] = "test-key"
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    AxbenchConceptEval(), TraitEval(), StolfoQualityEval()
check("no warning when the key is set", not caught)

# ---------------------------------------------------------------- score() against a client double
class FakeCompletions:
    def __init__(self, log):
        self.log = log

    def create(self, messages, model, temperature):
        self.log.append({"messages": messages, "model": model, "temperature": temperature})
        # rate 2 when the judged fragment mentions "pirate", else 0
        text = messages[0]["content"]
        score = 2 if "pirate" in text.split("[Sentence Fragment Start]")[-1] else 0
        class Msg: content = f"The fragment ...\nRating: [[{score}]]"
        class Choice: message = Msg
        class Resp: choices = [Choice]
        return Resp


class FakeClient:
    def __init__(self):
        self.log = []
        self.chat = type("Chat", (), {"completions": FakeCompletions(self.log)})


rows = pl.DataFrame({
    "concept_id": [0, 0],
    "prompt": ["Tell me about dogs", "Describe your day"],
    "response": ["Arr, me hearty pirate ship!", "I went to the shops."],
})
concepts = {0: Concept(id=0, name="pirate speech", description="")}

ev = AxbenchConceptEval()
ev._client = FakeClient()
out = ev.score(rows, concepts)
check("concept eval column + per-row scores",
      out == {"relevance_concept": [2.0, 0.0]})
check("request payload: single user message at config temperature",
      all(len(c["messages"]) == 1 and c["messages"][0]["role"] == "user"
          and c["model"] == "gpt-4o-mini" and c["temperature"] == 1.0
          for c in ev._client.log))
check("concept prompt fills Concept.name into the verbatim template",
      ev._client.log[0]["messages"][0]["content"]
      == CONCEPT_RELEVANCE_TEMPLATE.format(concept="pirate speech",
                                           sentence="Arr, me hearty pirate ship!"))

ev = AxbenchInstructEval()
ev._client = FakeClient()
out = ev.score(rows, concepts)
check("instruct eval fills the row prompt as the instruction",
      out["relevance_instruction"] == [2.0, 0.0]
      and "Tell me about dogs" in ev._client.log[0]["messages"][0]["content"])

ev = AxbenchFluencyEval()
ev._client = FakeClient()
out = ev.score(rows, concepts)
check("fluency eval scores the fragment only",
      out["fluency"] == [2.0, 0.0]
      and "[Instruction Start]" not in ev._client.log[0]["messages"][0]["content"])

# ---------------------------------------------------------------- benchmark wiring
from steergym.axbench.benchmark import Concept500Benchmark, Concept500Config
from steergym.steer import GeneratorRef

bench = Concept500Benchmark(Concept500Config(subject=GeneratorRef("fake/subject")))
evs = bench.evals()
check("concept500 evals() = the three axbench judges, trivially constructed",
      [type(e).__name__ for e in evs]
      == ["AxbenchConceptEval", "AxbenchInstructEval", "AxbenchFluencyEval"])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("all checks passed")
