#!/usr/bin/env python3
# Pure-python tests, no pytest. Run:
#   PYTHONPATH=<scratchpad>/pylibs python3 steergym/tests/test_personavec_benchmark.py
# (needs polars; no torch/transformers/openai — the generative stage runs
# against a fake generator double with filter="none" and a stubbed
# transformers.set_seed).
import json
import sys
import tempfile
import types as pytypes
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.modules.setdefault("transformers", pytypes.SimpleNamespace(set_seed=lambda s: None))

import polars as pl

from steergym import GeneratorRef, PersonaVectorsBenchmark, PersonaVectorsConfig, validate_samples
from steergym.personavec import TRAITS, build_packs, fold, system_text, trait_description

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

# ---------------------------------------------------------------- traits/packs
check("system_text: a/an + helpful neg arm (eval_persona.py verbatim)",
      system_text("apathetic", "Be flat.", "pos") == "You are an apathetic assistant. Be flat."
      and system_text("evil", "Be nice.", "neg") == "You are a helpful assistant. Be nice.")
check("fold = system prefix + blank line + question",
      fold("SYS", "Q?") == "SYS\n\nQ?")
packs = build_packs(TRAITS[:2], n_extract=3, n_eval=4, log=lambda m: None)
check("packs: one per trait, 5 instruction pairs, truncated questions",
      len(packs) == 2 and all(len(p["instruction"]) == 5 for p in packs)
      and all(len(p["questions_extract"]) == 3 and len(p["questions_eval"]) == 4 for p in packs))
check("description = eval_prompt definition paragraph (marker split)",
      all(p["description"] == trait_description(p["eval_prompt"]) for p in packs)
      and "{question}" not in packs[0]["description"])
check("vendored extract/eval files agree (flags false)",
      all(not p["eval_prompt_differs"] and not p["instruction_differs"] for p in packs))

# ---------------------------------------------------------------- benchmark, local paths
TMP = Path(tempfile.mkdtemp(prefix="steergym-pv-"))


def make_bench(**over):
    kw = dict(subject=SUBJECT, traits=TRAITS[:2], n_extract=2, n_eval=3,
              n_per_question=2, latent_n=2, gen_batch=4, filter="none")
    kw.update(over)
    return PersonaVectorsBenchmark(PersonaVectorsConfig(**kw))


check("upstream rollout protocol is the default (n_per_question=10, T=1.0)",
      PersonaVectorsConfig(subject=SUBJECT).n_per_question == 10
      and PersonaVectorsConfig(subject=SUBJECT).extraction_temperature == 1.0)


bench = make_bench()
cs = bench.concepts()
check("concepts: name = trait, description = definition paragraph",
      [c.name for c in cs] == list(TRAITS[:2]) and "trait" in cs[0].description)
ev = validate_samples(bench.eval_samples())
check("eval_samples: n_eval held-out questions per trait",
      ev.height == 6 and ev.get_column("response").null_count() == 6)
check("unknown filter rejected at construction",
      raises(lambda: make_bench(filter="strict"), "filter"))
check("train_samples without dump raises with both remedies",
      raises(lambda: make_bench().train_samples(), "dump_dir")
      and raises(lambda: make_bench().train_samples(), "bind_generator"))
check("filter='local' (removed) rejected at construction",
      raises(lambda: make_bench(filter="local"), "filter"))


class FakeGenerator:
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.ref = GeneratorRef(model_id=model_id)
        self.gen_calls = []

    def chat_prompt(self, text):
        return f"<chat>{text}"

    def generate(self, prompts, gen):
        self.gen_calls.append(gen)
        out = []
        for p in prompts:
            if "helpful assistant" in p:
                out.append("NEG")
            elif "You are" in p:
                out.append("POS")
            else:
                out.append("PLAIN")
        return out


check("mismatched generator subject raises",
      raises(lambda: make_bench(dump_dir=str(TMP / "d1"))
             .bind_generator(FakeGenerator("other/model")).train_samples(),
             "must come from the subject"))

dump = TMP / "dump"
fake = FakeGenerator()
gen_bench = make_bench(dump_dir=str(dump)).bind_generator(fake)
train = validate_samples(gen_bench.train_samples())
label = pl.col("meta").struct.field("label")
# 2 traits x (2 questions x 5 pairs x 2 rollouts) = 20 pairs/trait;
# filter=none -> all effective. Plain arm stays single-rollout.
check("train: 20 positives per trait; -1 pool = plain (2) + neg (20) per trait",
      train.filter(label == "pos").height == 40
      and train.filter(pl.col("concept_id") == -1).height == 2 * (2 + 20))
check("extraction sampled at config temperature",
      all(g.temperature == 1.0 and g.do_sample for g in fake.gen_calls))
check("filter='none' marks every pair effective, fallback unused",
      all(not s["fallback_used"] and s["n_pairs"] == 20
          for s in json.loads((dump / "filter_stats.json").read_text()).values()))

polar = gen_bench.polar_train_samples()
check("polar: winning = pos response, losing = the question's plain response",
      polar.height == 40 and set(polar.get_column("winning").to_list()) == {"POS"}
      and set(polar.get_column("losing").to_list()) == {"PLAIN"})
latent = gen_bench.latent_samples()
check("latent: latent_n pos + latent_n neg per trait, sae_link = trait",
      latent.height == 8
      and set(latent.get_column("meta").struct.field("sae_link").to_list()) == set(TRAITS[:2]))

ext = validate_samples(gen_bench.extraction_samples())
mlabel = ext.get_column("meta").struct.field("label")
check("extraction rows: pos/neg per (pair x rollout) + plain per question, folded prompts",
      mlabel.value_counts().sort("label").rows() == [("neg", 40), ("plain", 4), ("pos", 40)]
      and all("\n\n" in p for p in
              ext.filter(mlabel == "pos").get_column("prompt").to_list()))
check("extraction meta carries the effective mask and (question, instruction, sample) pairing",
      set(ext.get_column("meta").struct.fields) >= {"label", "question", "instruction_idx",
                                                    "sample_idx", "trait_score", "coherence",
                                                    "effective"}
      and sorted(ext.filter(mlabel == "pos").get_column("meta").struct.field("sample_idx")
                 .unique().to_list()) == [0, 1])
greedy_dir = TMP / "dump-greedy"
greedy_fake = FakeGenerator()
make_bench(dump_dir=str(greedy_dir), n_per_question=1, extraction_temperature=0.0) \
    .bind_generator(greedy_fake).train_samples()
check("faithful single-sample override runs greedy (upstream n=1 -> T=0 coupling, explicit)",
      all(g.temperature == 0.0 and not g.do_sample for g in greedy_fake.gen_calls))

cached = make_bench(dump_dir=str(dump)).train_samples()  # no generator bound
check("populated dump reads back without a generator", cached.equals(train))

# legacy-style dump: everything but the extraction parquet
(dump / "extraction" / "extraction_data.parquet").rename(TMP / "stash.parquet")
check("legacy dump (no extraction parquet) raises the targeted error",
      raises(lambda: make_bench(dump_dir=str(dump)).extraction_samples(), "predates"))
(TMP / "stash.parquet").rename(dump / "extraction" / "extraction_data.parquet")

metadata = [json.loads(l) for l in (dump / "generate" / "metadata.jsonl").read_text().splitlines()]
check("metadata refs end in an int concept id",
      all(m["ref"].split("/")[-1] == str(m["concept_id"]) for m in metadata))


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
