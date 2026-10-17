import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..axbench.benchmark import LATENT_META_KEYS, TRAIN_META_KEYS, canonical_samples
from ..benchmark import LABEL_NEG, LABEL_POS, META_LABEL, BenchmarkConfig, PolarBenchmark, SteerBenchmark
from ..concept import Concept
from ..generator import GenParams
from ..steer import GeneratorRef
from .evals import CHECKER_META_KEY, check
from .types import STOLFO_DIR, build_packs, select_types


@dataclass
class IFEvalConfig(BenchmarkConfig):
    subject: GeneratorRef              # generates the pack data AND is steered; checked against any bound generator
    dump_dir: str | None = None        # gen_pairs-style dump: cached source, or the generative stage's cache target
    stolfo_dir: str | None = None      # Stolfo pool override; default = the vendored copy
    type_set: str = "format+language"  # "format" (12) | "format+language" (12+19, protocol default) | "all"
    n_types: int | None = None
    n_train: int = 120                 # pack rows per type (pre checker filter)
    n_eval: int = 10
    seed: int = 42
    gen_tokens: int = 192              # config of record
    gen_batch: int = 16
    n_pos: int = 72                    # checker-passing positives kept per type
    latent_n: int = 36                 # latent pos/neg rows per type


class IFEvalBenchmark(SteerBenchmark, PolarBenchmark):
    """IFEval verifiable constraints as steering concepts: one concept per
    Stolfo-pool type_id, concept text = build_description at the pack's modal
    effective kwargs. Concept.name doubles as the train rows' output_concept
    (the axbench fits filter on it) and Concept.description as the prompt-arm
    text — the eval-time steered prompt is base + "\\n" + desc, exactly
    PromptSteerModel(placement="append").

    The whole training side is GENERATIVE (there is nothing released): the
    subject model answers base prompts with and without the instruction, the
    official checker filters constrained responses to genuine positives, and
    plain responses double as the losing/negative side. Cached mode reads a
    gen_pairs-layout dump; generative mode needs bind_generator() AND dump_dir
    (the dump is the artifact — packs.json in it is the provenance record, and
    config type params apply only when generating). eval_samples/concepts need
    neither: packs build locally from the vendored pool. StolfoDiffMean
    training is StolfoDiffMeanModel.fit over stolfo_samples()."""

    config_cls = IFEvalConfig

    def __init__(self, config: IFEvalConfig):
        super().__init__(config)
        self._generator = None
        self._packs = None

    def bind_generator(self, generator):
        self._generator = generator
        return self

    def _stolfo_dir(self):
        return Path(self.config.stolfo_dir) if self.config.stolfo_dir else STOLFO_DIR

    def _dump(self) -> Path | None:
        return Path(self.config.dump_dir) if self.config.dump_dir else None

    def _dump_populated(self) -> bool:
        dump = self._dump()
        return dump is not None and (dump / "generate" / "train_data.parquet").exists()

    def packs(self) -> list[dict]:
        if self._packs is None:
            if self._dump_populated():
                self._packs = json.loads((self._dump() / "packs.json").read_text())
            else:
                types = select_types(self.config.type_set, self._stolfo_dir(), self.config.n_types)
                self._packs = build_packs(types, self.config.n_train, self.config.n_eval,
                                          self.config.seed, self._stolfo_dir())
        return self._packs

    def concepts(self) -> list[Concept]:
        return [Concept(id=p["concept_id"], name=p["concept"], description=p["concept"])
                for p in self.packs()]

    def eval_samples(self) -> pl.DataFrame:
        rows = []
        for p in self.packs():
            for e in p["eval"]:
                rows.append({
                    "concept_id": p["concept_id"],
                    "prompt": e["base"],
                    "response": None,
                    "meta": {
                        CHECKER_META_KEY: {
                            "instruction_id": p["instruction_id"],
                            "kwargs_json": json.dumps(e["kwargs"]),
                        },
                        "desc": e["desc"],
                        "type_id": p["type_id"],
                        "key": e.get("key"),
                    },
                })
        return pl.DataFrame(rows)

    def stolfo_samples(self) -> pl.DataFrame:
        # ALL pack train rows, not checker-filtered: the Stolfo diff-mean is
        # over instruction-present vs instruction-absent PROMPTS (gen_pairs
        # computed it over every train row) — StolfoDiffMeanModel.fit's input
        rows = []
        for p in self.packs():
            for e in p["train"]:
                rows.append({
                    "concept_id": p["concept_id"],
                    "prompt": e["base"],
                    "response": None,
                    "meta": {"desc": e["desc"], "type_id": p["type_id"]},
                })
        return pl.DataFrame(rows)

    def train_samples(self) -> pl.DataFrame:
        self._require_dump("train_samples")
        native = pl.read_parquet(self._dump() / "generate" / "train_data.parquet")
        return canonical_samples(native, TRAIN_META_KEYS)

    def latent_samples(self) -> pl.DataFrame:
        # axbench staging quirk kept: latent rows live under pregen/, NOT
        # generate/ (axbench's pre_compute_mean_activations globs the
        # inference dir and chokes on strays)
        self._require_dump("latent_samples")
        native = pl.read_parquet(self._dump() / "pregen" / "latent_eval_data.parquet")
        return canonical_samples(native, LATENT_META_KEYS)

    def polar_train_samples(self) -> pl.DataFrame:
        self._require_dump("polar_train_samples")
        native = pl.read_parquet(self._dump() / "generate" / "dpo_train_data.parquet")
        label = (pl.when(pl.col("category") == "positive").then(pl.lit(LABEL_POS))
                 .otherwise(pl.lit(LABEL_NEG)))
        return native.select(
            "concept_id",
            pl.col("input").alias("prompt"),
            pl.col("winning_output").alias("winning"),
            pl.col("losing_output").alias("losing"),
            pl.struct(label.alias(META_LABEL), *[pl.col(k) for k in TRAIN_META_KEYS]).alias("meta"),
        )

    def evals(self) -> list:
        # the paper-defined evals: official checkers (strict+loose) and
        # Stolfo's GPT-2 perplexity degeneracy gate. The two-stage GPT-4o
        # quality judge (StolfoQualityEval) is API spend and stays a
        # runscript opt-in, as does any generic coherency judge
        # (steergym.evals.judges).
        from .evals import IFEvalEval, StolfoPerplexityEval
        return [IFEvalEval(), StolfoPerplexityEval()]

    # ---------------------------------------------------------------- generative stage
    def _require_dump(self, method):
        if self._dump_populated():
            return
        dump = self._dump()
        if dump is None:
            raise RuntimeError(
                f"IFEvalBenchmark.{method} needs a generated dump and no dump_dir is set — "
                "point dump_dir at a gen_pairs-style dump, or set dump_dir AND "
                "bind_generator(GeneratorLM) to generate one there")
        if self._generator is None:
            raise RuntimeError(
                f"IFEvalBenchmark.{method}: dump_dir {str(dump)!r} has no "
                "generate/train_data.parquet and no generator is bound — "
                "bind_generator(GeneratorLM) to generate the pack there")
        if self._generator.ref.model_id != self.config.subject.model_id:
            raise RuntimeError(
                f"bound generator is {self._generator.ref.model_id!r} but config.subject "
                f"is {self.config.subject.model_id!r} — pack responses must come from the subject")
        self._generate_dump()

    def _generate_dump(self):
        # gen_pairs.generate, minus StolfoDiffMean weights (now a model fit).
        from transformers import set_seed

        cfg = self.config
        dump = self._dump()
        gen_dir, pregen_dir = dump / "generate", dump / "pregen"
        gen_dir.mkdir(parents=True, exist_ok=True)
        pregen_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed)
        gen = GenParams(max_new_tokens=cfg.gen_tokens, temperature=1.0, do_sample=True,
                        batch_size=cfg.gen_batch, seed=None)
        packs = self.packs()

        # one plain (unconstrained) response per UNIQUE base, shared across
        # types drawing the same Stolfo base (11242 rows sit on 541 bases) —
        # and the train_data global negative pool for free
        all_bases = sorted({e["base"] for p in packs for e in p["train"]})
        plain_responses = self._generator.generate(
            [self._generator.chat_prompt(b) for b in all_bases], gen)
        plain_by_base = dict(zip(all_bases, plain_responses))

        metadata_lines, train_rows, dpo_rows, latent_rows = [], [], [], []
        pass_rates = {}
        for p in packs:
            cid, tid, iid, concept = p["concept_id"], p["type_id"], p["instruction_id"], p["concept"]
            # train-time constrained prompt is SPACE-joined (eval-time steering
            # via PromptSteerModel "append" is newline-joined)
            constrained = self._generator.generate(
                [self._generator.chat_prompt(f'{e["base"]} {e["desc"]}') for e in p["train"]], gen)

            passing = [(e["base"], resp) for e, resp in zip(p["train"], constrained)
                       if check(iid, e["kwargs"], resp)[0]]
            pass_rates[tid] = len(passing) / max(1, len(p["train"]))

            for base, resp in passing[:cfg.n_pos]:
                common = {"output_concept": concept, "concept_genre": "text",
                          "dataset_category": "instruction", "concept_id": cid}
                train_rows.append({"input": base, "output": resp, "category": "positive", **common})
                dpo_rows.append({"input": base, "winning_output": resp,
                                 "losing_output": plain_by_base[base], "category": "positive", **common})
            for base, resp in passing[:cfg.latent_n]:
                latent_rows.append({"input": base, "output": resp, "output_concept": concept,
                                    "concept_genre": "text", "category": "positive",
                                    "dataset_category": "instruction", "concept_id": cid,
                                    "sae_link": tid, "sae_id": 0})
            for e in p["train"][:cfg.latent_n]:
                latent_rows.append({"input": e["base"], "output": plain_by_base[e["base"]],
                                    "output_concept": concept, "concept_genre": "text",
                                    "category": "negative", "dataset_category": "instruction",
                                    "concept_id": cid, "sae_link": tid, "sae_id": 0})
            # inference.py int-parses the ref's last path segment — keep the
            # type id in the path, END with the concept id
            metadata_lines.append(json.dumps({
                "concept_id": cid, "concept": concept, "ref": f"ifeval/{tid}/{cid}",
                "concept_genres_map": {concept: ["text"]}}))

        for base, resp in plain_by_base.items():
            train_rows.append({"input": base, "output": resp, "output_concept": "EEEEE",
                               "concept_genre": "text", "category": "negative",
                               "dataset_category": "instruction", "concept_id": -1})

        pl.DataFrame(train_rows).write_parquet(gen_dir / "train_data.parquet")
        pl.DataFrame(dpo_rows).write_parquet(gen_dir / "dpo_train_data.parquet")
        pl.DataFrame(latent_rows).write_parquet(pregen_dir / "latent_eval_data.parquet")
        (gen_dir / "metadata.jsonl").write_text("\n".join(metadata_lines) + "\n")
        (dump / "packs.json").write_text(json.dumps(packs, indent=2))
        (dump / "pass_rates.json").write_text(json.dumps(pass_rates, indent=2))
