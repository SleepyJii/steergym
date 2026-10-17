#!/usr/bin/env python3
# Pure-python tests, no pytest. Run:
#   PYTHONPATH=<scratchpad>/pylibs python3 steergym/tests/test_benchmark.py
# (needs polars + pandas on PYTHONPATH; no torch/transformers -- the polar
# generation path runs against a fake generator double).
import json
import sys
import tempfile
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from steergym import (
    LABEL_NEG,
    LABEL_POS,
    AxbenchPromptSteerModel,
    Concept500Benchmark,
    Concept500Config,
    GeneratorRef,
    SteerConfig,
    validate_samples,
)
from steergym.axbench.models import T_PROMPT_STEERING
from steergym.axbench.protocol import positive_and_negative_rows
from steergym.concept import Concept

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


def raises(fn, needle):
    try:
        fn()
    except Exception as e:
        return needle in str(e)
    return False


SUBJECT = GeneratorRef(model_id="google/gemma-2-2b-it")

# ---------------------------------------------------------------- synthetic dump
TMP = Path(tempfile.mkdtemp(prefix="steergym-c500-"))
(TMP / "generate").mkdir()
(TMP / "inference").mkdir()

# native axbench frames: two concepts (genres text/code) + the global -1 pool
_native_train = pl.DataFrame({
    "input": ["q0a", "q0b", "q1a", "neg-text", "neg-code"],
    "output": ["r0a", "r0b", "r1a", "neg-out-t", "neg-out-c"],
    "output_concept": ["alpha", "alpha", "beta", "EEEEE", "EEEEE"],
    "concept_genre": ["text", "text", "code", "text", "code"],
    "category": ["positive", "positive", "positive", "negative", "negative"],
    "dataset_category": ["instruction"] * 5,
    "concept_id": [0, 0, 1, -1, -1],
})
_native_train.write_parquet(TMP / "generate" / "train_data.parquet")
_native_latent = _native_train.head(3).with_columns(
    pl.lit("https://neuronpedia/x").alias("sae_link"), pl.lit(7).alias("sae_id"))
_native_latent.write_parquet(TMP / "inference" / "latent_eval_data.parquet")
(TMP / "generate" / "metadata.jsonl").write_text("\n".join(
    json.dumps({"concept_id": i, "concept": name, "ref": "https://neuronpedia/x",
                "concept_genres_map": {name: ["text"]}})
    for i, name in enumerate(["alpha", "beta"])) + "\n")
(TMP / "alpaca_eval.json").write_text(json.dumps(
    [{"instruction": f"instr-{i}", "output": "o", "generator": "g"} for i in range(40)]))


def make_bench(**over):
    kw = dict(subject=SUBJECT, dump_dir=str(TMP), alpaca_path=str(TMP / "alpaca_eval.json"),
              eval_prompts_per_concept=3)
    kw.update(over)
    return Concept500Benchmark(Concept500Config(**kw))


# ---------------------------------------------------------------- source config
check("dump_dir xor hf_repo enforced (neither)",
      raises(lambda: Concept500Benchmark(Concept500Config(subject=SUBJECT)).concepts(),
             "exactly one data source"))
check("dump_dir xor hf_repo enforced (both)",
      raises(lambda: Concept500Benchmark(Concept500Config(
          subject=SUBJECT, dump_dir=str(TMP), hf_repo="pyvene/x")).concepts(),
             "exactly one data source"))
check("non-dump dir rejected",
      raises(lambda: Concept500Benchmark(Concept500Config(
          subject=SUBJECT, dump_dir=str(TMP / "generate"))).concepts(),
             "not a concept500 dump"))

# ---------------------------------------------------------------- concepts
bench = make_bench()
cs = bench.concepts()
check("concepts parsed from metadata.jsonl",
      [c.id for c in cs] == [0, 1] and [c.name for c in cs] == ["alpha", "beta"])
check("n_concepts truncates the concept list",
      [c.id for c in make_bench(n_concepts=1).concepts()] == [0])

# ---------------------------------------------------------------- canonical repack
train = validate_samples(bench.train_samples())
check("train_samples canonical columns",
      train.columns == ["concept_id", "prompt", "response", "meta"])
check("prompt/response carry input/output",
      train.get_column("prompt").to_list()[:2] == ["q0a", "q0b"]
      and train.get_column("response").to_list()[:2] == ["r0a", "r0b"])
meta = train.get_column("meta")
check("meta keeps every native column verbatim (lossless repack)",
      set(meta.struct.fields) == {"label", "category", "output_concept",
                                  "concept_genre", "dataset_category"}
      and meta.struct.field("output_concept").to_list() == _native_train.get_column("output_concept").to_list())
check("label derives pos/neg from category",
      meta.struct.field("label").to_list() == [LABEL_POS] * 3 + [LABEL_NEG] * 2)
check("n_concepts subsetting keeps the -1 pool",
      sorted(make_bench(n_concepts=1).train_samples().get_column("concept_id").unique().to_list())
      == [-1, 0])

latent = bench.latent_samples()
check("latent_samples adds sae meta keys",
      {"sae_link", "sae_id"} < set(latent.get_column("meta").struct.fields))

# ---------------------------------------------------------------- positive_and_negative_rows
pos, neg = positive_and_negative_rows(train, Concept(id=0, name="alpha", description=""))
check("positives filtered by meta.output_concept + category",
      pos.get_column("prompt").to_list() == ["q0a", "q0b"])
check("negatives are the -1 pool, genre-matched",
      neg.get_column("prompt").to_list() == ["neg-text"])
pos2, neg2 = positive_and_negative_rows(train, Concept(id=9, name="nope", description=""))
check("unknown concept yields empty positive and empty negative",
      pos2.height == 0 and neg2.height == 0)

# ---------------------------------------------------------------- eval_samples
ev = validate_samples(bench.eval_samples())
check("eval_samples: n prompts per concept, null responses",
      ev.height == 6 and ev.get_column("response").null_count() == 6)
check("eval_samples deterministic (concept_id-seeded sampling)",
      ev.equals(bench.eval_samples()))
check("different concepts sample independently",
      ev.filter(pl.col("concept_id") == 0).get_column("prompt").to_list()
      != ev.filter(pl.col("concept_id") == 1).get_column("prompt").to_list())

# ---------------------------------------------------------------- polar pairs
check("polar without generator or cache raises with both remedies",
      raises(lambda: bench.polar_train_samples(), "bind_generator")
      and raises(lambda: bench.polar_train_samples(), "dpo_dir"))


class FakeGenerator:
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.ref = GeneratorRef(model_id=model_id)
        self.gen_calls = []

    def chat_prompt(self, text):
        return f"<chat>{text}"

    def generate(self, prompts, gen):
        self.gen_calls.append(gen)
        return [f"cont:{p}" for p in prompts]


check("polar with mismatched generator subject raises",
      raises(lambda: make_bench().bind_generator(FakeGenerator("other/model")).polar_train_samples(),
             "must come from the subject"))

dpo_dir = TMP / "dpo"
fake = FakeGenerator()
polar = make_bench(dpo_dir=str(dpo_dir)).bind_generator(fake).polar_train_samples()
check("polar pairs: winning = positive response, losing = subject continuation",
      polar.columns == ["concept_id", "prompt", "winning", "losing", "meta"]
      and polar.get_column("winning").to_list() == ["r0a", "r0b", "r1a"]
      and polar.get_column("losing").to_list() == ["cont:<chat>q0a", "cont:<chat>q0b", "cont:<chat>q1a"])
check("polar generation is greedy at 1.5x output_length",
      all(g.do_sample is False and g.max_new_tokens == 192 for g in fake.gen_calls))
check("polar pairs cached to dpo_dir", (dpo_dir / "polar_train_data.parquet").exists())

cached = make_bench(dpo_dir=str(dpo_dir)).polar_train_samples()  # no generator bound
check("cached polar pairs load without a generator", cached.equals(polar))

# partial-cache check: cache generated under n_concepts=1, then asked for both
partial_dir = TMP / "dpo-partial"
make_bench(dpo_dir=str(partial_dir), n_concepts=1).bind_generator(FakeGenerator()).polar_train_samples()
check("partial dpo cache raises instead of silently under-covering",
      raises(lambda: make_bench(dpo_dir=str(partial_dir)).polar_train_samples(),
             "covers only part"))

# ---------------------------------------------------------------- prompt arm
ctx = AxbenchPromptSteerModel(SteerConfig(subject=SUBJECT)).context(
    Concept(id=0, name="alpha", description=""))
check("axbench prompt arm renders T_PROMPT_STEERING + Question joiner",
      ctx.render("what?") == f"{T_PROMPT_STEERING % 'alpha'}\n\nQuestion: what?")


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
