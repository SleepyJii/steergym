#!/usr/bin/env python3
# Pure-python tests, no pytest. Run:
#   PYTHONPATH=<scratchpad>/pylibs python3 steergym/tests/test_helpers.py
# (from the steergym/ repo root, or any cwd -- path is resolved below).
# No torch/transformers on PYTHONPATH: generator/steer model are fake doubles, and
# GenParams.seed stays None so steer_sweep never touches transformers.set_seed.
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

import steergym
from steergym import GenParams, GeneratorLM, steer_sweep
from steergym.concept import Concept

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


# ---------------------------------------------------------------- fakes
class FakeContext:
    def __init__(self, coeffs, flag):
        self.coeffs = coeffs
        self._flag = flag

    def render(self, prompt):
        return f"{prompt}|render"

    def __enter__(self):
        self._flag["active"] = True
        return self

    def __exit__(self, *exc):
        self._flag["active"] = False
        return False

    def diagnostics(self):
        return {"applied_delta_norm": self.coeffs.get("factor")}


class FakeContextNoDiag(FakeContext):
    def diagnostics(self):
        return {}


class FakeModel:
    def __init__(self, ctx_cls=FakeContext):
        self.flag = {"active": False}
        self.ctx_cls = ctx_cls
        self.calls = []

    def context(self, concept, **coeffs):
        self.calls.append((concept.id, coeffs))
        return self.ctx_cls(coeffs, self.flag)


class FakeGenerator:
    def __init__(self, model):
        self.model = model
        self.generate_calls = []  # (chat_prompts, was_active)

    def chat_prompt(self, text):
        return f"{text}|chat"

    def generate(self, prompts, gen):
        active = self.model.flag["active"]
        self.generate_calls.append((list(prompts), active))
        return [f"resp::{p}::active={active}" for p in prompts]


# ---------------------------------------------------------------- fixtures
concepts = [
    Concept(id=10, name="alpha", description=""),
    Concept(id=20, name="beta", description=""),
]
eval_samples = pl.DataFrame({
    "concept_id": [10, 10, 10, 20, 20, 20],
    "prompt": ["a1", "a2", "a3", "b1", "b2", "b3"],
})
sweep = [{"factor": 0.5}, {"factor": 1.5}]
gen = GenParams(batch_size=2)  # seed=None -> no transformers.set_seed import


# ---------------------------------------------------------------- basic sweep
model = FakeModel()
generator = FakeGenerator(model)
df = steer_sweep(generator, model, concepts, eval_samples, sweep, gen)

check("row count = n_concepts x n_arms x n_prompts", df.height == 2 * 2 * 3)

check("row order = concept-major, arm, then prompt order",
      df.get_column("concept_id").to_list() == [10] * 6 + [20] * 6
      and df.get_column("prompt").to_list() ==
      ["a1", "a2", "a3", "a1", "a2", "a3", "b1", "b2", "b3", "b1", "b2", "b3"])

check("prompt column holds the raw pre-render prompt",
      all(not p.endswith("|render") and not p.endswith("|chat") for p in df.get_column("prompt").to_list())
      and set(df.get_column("prompt").to_list()) == {"a1", "a2", "a3", "b1", "b2", "b3"})

check("render applied before chat_prompt (marker order in recorded input)",
      all(resp.startswith(f"resp::{p}|render|chat::") for p, resp in
          zip(df.get_column("prompt").to_list(), df.get_column("response").to_list())))

check("generation happened inside the context (flag captured true)",
      len(generator.generate_calls) == 4
      and all(active is True for _, active in generator.generate_calls))

check("coeffs struct column carries the arm's coefficients",
      df.get_column("coeffs").struct.field("factor").to_list() ==
      [0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5])

check("diag struct column carries the context's diagnostics",
      df.get_column("diag").struct.field("applied_delta_norm").to_list() ==
      df.get_column("coeffs").struct.field("factor").to_list())

check("context() called once per concept x arm (not per prompt)",
      len(model.calls) == 4
      and model.calls == [(10, {"factor": 0.5}), (10, {"factor": 1.5}),
                           (20, {"factor": 0.5}), (20, {"factor": 1.5})])


# ---------------------------------------------------------------- empty-diagnostics -> null column
model_nd = FakeModel(ctx_cls=FakeContextNoDiag)
generator_nd = FakeGenerator(model_nd)
df_nd = steer_sweep(generator_nd, model_nd, concepts, eval_samples, sweep, gen)

check("all-empty diagnostics collapse to a null diag column (no zero-field struct)",
      df_nd.get_column("diag").dtype != pl.Struct
      and df_nd.get_column("diag").null_count() == df_nd.height)


# ---------------------------------------------------------------- package surface
check("import steergym exposes GeneratorLM, GenParams, steer_sweep",
      steergym.GeneratorLM is GeneratorLM and steergym.GenParams is GenParams
      and steergym.steer_sweep is steer_sweep)


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
