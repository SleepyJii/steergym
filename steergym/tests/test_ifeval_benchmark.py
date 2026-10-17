#!/usr/bin/env python3
# Pure-python tests, no pytest. Run:
#   PYTHONPATH=<scratchpad>/pylibs python3 steergym/tests/test_ifeval_benchmark.py
# (needs polars + the vendored-checker deps: absl-py, langdetect, nltk,
# immutabledict; no torch/transformers — the generative stage runs against a
# fake generator double with a stubbed transformers.set_seed).
import json
import sys
import tempfile
import types as pytypes
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
# _generate_pack lazily imports transformers.set_seed for protocol seeding;
# locally we stub it (same spirit as test_helpers' "GenParams.seed stays None")
sys.modules.setdefault("transformers", pytypes.SimpleNamespace(set_seed=lambda s: None))

import polars as pl

from steergym import GeneratorRef, IFEvalBenchmark, IFEvalConfig, validate_samples
from steergym.ifeval import build_packs, checker_id, select_types
from steergym.ifeval.types import STOLFO_DIR

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

# ---------------------------------------------------------------- real vendored pool
fmt = select_types("format")
check("format type set has the 12 pool-present format ids",
      len(fmt) == 12 and all(t.split(":")[0] in
          ("detectable_format", "change_case", "punctuation", "startend") for t in fmt))
fl = select_types("format+language")
check("format+language extends format with language:response_language_* ids",
      fl[:12] == fmt and len(fl) > 12
      and all(t.startswith("language:response_language_") for t in fl[12:]))
check("n_types head-truncates", select_types("format", n_types=3) == fmt[:3])
check("unknown type_set raises", raises(lambda: select_types("everything"), "type_set"))
check("checker_id strips pool-id suffixes",
      checker_id("keywords:frequency_at least") == "keywords:frequency"
      and checker_id("language:response_language_de") == "language:response_language"
      and checker_id("punctuation:no_comma") == "punctuation:no_comma")

packs = build_packs(fmt[:3], n_train=20, n_eval=4, seed=0, log=lambda m: None)
check("packs: one per type, concept ids sequential",
      [p["concept_id"] for p in packs] == [0, 1, 2]
      and all(p["concept"] for p in packs))
check("packs: train/eval bases disjoint within a pack",
      all(not ({e["base"] for e in p["train"]} & {e["base"] for e in p["eval"]})
          for p in packs))

# ---------------------------------------------------------------- synthetic pool
TMP = Path(tempfile.mkdtemp(prefix="steergym-ifeval-"))
pool_dir = TMP / "stolfo"
pool_dir.mkdir()
BULLET = "detectable_format:number_bullet_lists"
NO_COMMA = "punctuation:no_comma"
aug_rows = []
for i in range(8):
    aug_rows.append({"key": 100 + i, "single_instruction_id": BULLET,
                     "prompt_without_instruction": f"bullet base {i}",
                     "kwargs": [{"num_bullets": 3}]})
    aug_rows.append({"key": 200 + i, "single_instruction_id": NO_COMMA,
                     "prompt_without_instruction": f"comma base {i}", "kwargs": [{}]})
(pool_dir / "ifeval_augmented_filtered.jsonl").write_text(
    "\n".join(json.dumps(r) for r in aug_rows))
(pool_dir / "ifeval_single_instr_format.jsonl").write_text(
    json.dumps({"prompt_without_instruction": "bullet base 0",
                "instruction_id_list": [BULLET]}))


def make_bench(**over):
    kw = dict(subject=SUBJECT, stolfo_dir=str(pool_dir), n_train=6, n_eval=2,
              n_pos=4, latent_n=2, gen_batch=4)
    kw.update(over)
    return IFEvalBenchmark(IFEvalConfig(**kw))


bench = make_bench()
cs = bench.concepts()
check("synthetic pool: two concepts, bullet type first (registry order)",
      len(cs) == 2 and "3 bullet points" in cs[0].name and cs[0].name == cs[0].description)
ev = validate_samples(bench.eval_samples())
check("eval_samples carry checker payload with effective kwargs",
      ev.height == 4
      and json.loads(ev.row(0, named=True)["meta"]["checker"]["kwargs_json"]) == {"num_bullets": 3})
check("stolfo_samples = all pack train rows with meta.desc",
      bench.stolfo_samples().height == 12)

# ---------------------------------------------------------------- generative stage
check("train_samples without dump_dir raises with both remedies",
      raises(lambda: make_bench().train_samples(), "dump_dir")
      and raises(lambda: make_bench().train_samples(), "bind_generator"))
check("dump_dir without generator raises",
      raises(lambda: make_bench(dump_dir=str(TMP / "nodump")).train_samples(),
             "no generator is bound"))


class FakeGenerator:
    """Responses keyed off the prompt: constrained bullet prompts get a
    passing 3-bullet answer (except the designated failing base), constrained
    no-comma prompts a comma-free one, plain prompts a comma-carrying one."""

    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.ref = GeneratorRef(model_id=model_id)

    def chat_prompt(self, text):
        return f"<chat>{text}"

    def generate(self, prompts, gen):
        out = []
        for p in prompts:
            if "bullet points" in p:  # constrained bullet prompt (desc present)
                out.append("nope" if "bullet base 3" in p else "* a\n* b\n* c")
            elif "commas" in p:       # constrained no-comma prompt
                out.append("clean text without them")
            else:                     # plain base prompt
                out.append("plain, response")
        return out


check("mismatched generator subject raises",
      raises(lambda: make_bench(dump_dir=str(TMP / "d0"))
             .bind_generator(FakeGenerator("other/model")).train_samples(),
             "must come from the subject"))

dump = TMP / "dump"
gen_bench = make_bench(dump_dir=str(dump)).bind_generator(FakeGenerator())
train = validate_samples(gen_bench.train_samples())
label = pl.col("meta").struct.field("label")
pos_by_cid = {cid: n for cid, n in
              train.filter(label == "pos").group_by("concept_id").len().iter_rows()}
check("checker filter dropped the failing bullet row (5 of 6 pass)",
      pos_by_cid.get(0) == 4 and pos_by_cid.get(1) == 4)  # capped at n_pos=4
neg_pool = train.filter(pl.col("concept_id") == -1)
check("global negative pool = one plain response per unique base",
      neg_pool.height == 12 and set(neg_pool.get_column("response").to_list()) == {"plain, response"})
pass_rates = json.loads((dump / "pass_rates.json").read_text())
check("pass_rates.json records the prompt-following baseline",
      abs(pass_rates[BULLET] - 5 / 6) < 1e-9 and pass_rates[NO_COMMA] == 1.0)

polar = gen_bench.polar_train_samples()
check("polar pairs: winning = passing response, losing = plain response",
      polar.columns == ["concept_id", "prompt", "winning", "losing", "meta"]
      and set(polar.get_column("losing").to_list()) == {"plain, response"}
      and polar.height == 8)
latent = gen_bench.latent_samples()
check("latent rows: latent_n pos + latent_n neg per concept, sae_link = type_id",
      latent.height == 8
      and set(latent.get_column("meta").struct.field("sae_link").to_list()) == {BULLET, NO_COMMA})

cached = make_bench(dump_dir=str(dump)).train_samples()  # no generator bound
check("populated dump reads back without a generator", cached.equals(train))
check("packs.json in the dump is the provenance record",
      json.loads((dump / "packs.json").read_text())[0]["type_id"] == BULLET)

metadata = [json.loads(l) for l in (dump / "generate" / "metadata.jsonl").read_text().splitlines()]
check("metadata refs end in an int concept id (inference.py int-parses)",
      all(m["ref"].split("/")[-1] == str(m["concept_id"]) for m in metadata))


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
