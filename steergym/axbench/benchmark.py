import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..benchmark import LABEL_NEG, LABEL_POS, META_LABEL, BenchmarkConfig, PolarBenchmark, SteerBenchmark
from ..concept import Concept
from ..generator import GenParams
from ..steer import GeneratorRef

# columns of the native axbench train/latent parquets that ride verbatim in
# `meta` -- the canonical repack is lossless: input->prompt, output->response,
# everything else keeps its native name inside the struct, plus the derived
# `label` ("hard negative" rows in the latent frame keep that literal label)
TRAIN_META_KEYS = ("category", "output_concept", "concept_genre", "dataset_category")
LATENT_META_KEYS = TRAIN_META_KEYS + ("sae_link", "sae_id")

# factor grid of record; dose = factor x max_act. Reference list only --
# coefficient grids are a runscript concern, not a benchmark method
CONCEPT500_FACTORS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0)


def canonical_samples(native: pl.DataFrame, meta_keys: tuple) -> pl.DataFrame:
    label = (pl.when(pl.col("category") == "positive").then(pl.lit(LABEL_POS))
             .when(pl.col("category") == "negative").then(pl.lit(LABEL_NEG))
             .otherwise(pl.col("category")))
    return native.select(
        "concept_id",
        pl.col("input").alias("prompt"),
        pl.col("output").alias("response"),
        pl.struct(label.alias(META_LABEL), *[pl.col(k) for k in meta_keys]).alias("meta"),
    )


@dataclass
class Concept500Config(BenchmarkConfig):
    subject: GeneratorRef              # the dump is subject+layer specific; checked against any bound generator
    dump_dir: str | None = None        # prod_2b_l20_v1-style dump (generate/ + inference/)
    hf_repo: str | None = None         # lazy snapshot_download(repo_type="dataset") instead
    hf_subdir: str | None = None       # dump root inside the snapshot, when not at its top
    dpo_dir: str | None = None         # polar-pair cache; generated there when absent
    alpaca_path: str | None = None     # local alpaca_eval.json; else lazy hf download
    n_concepts: int | None = None      # head-subset of the concept list
    eval_prompts_per_concept: int = 10  # axbench steering_num_of_examples (paper: 10)
    dpo_output_length: int = 128       # config of record; losing gen = 1.5x this, greedy
    dpo_batch_size: int = 8


class Concept500Benchmark(SteerBenchmark, PolarBenchmark):
    """axbench concept500: gemmascope-SAE concepts with gpt-4o-mini-generated
    training data, shipped as a prod_2b_l20_v1-style dump. Concepts carry
    description="" (the dump has no definition text beyond the concept name;
    ExpressionEval falls back to its name-based prefix, and the axbench
    prompt-steering arm is AxbenchPromptSteerModel). The polar (DPO) pairs are
    NOT in the dump: losing responses are the subject's own unsteered greedy
    continuations (generate.py --mode dpo_training, keep_orig_axbench_format),
    a local-GPU stage -- bind_generator() to run it, dpo_dir to cache it."""

    config_cls = Concept500Config

    def __init__(self, config: Concept500Config):
        super().__init__(config)
        self._generator = None

    def bind_generator(self, generator):
        self._generator = generator
        return self

    def _root(self) -> Path:
        cfg = self.config
        if (cfg.dump_dir is None) == (cfg.hf_repo is None):
            raise ValueError("Concept500Config needs exactly one data source: "
                             "dump_dir (a local dump) or hf_repo (lazy download)")
        if cfg.dump_dir is not None:
            root = Path(cfg.dump_dir)
        else:
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(cfg.hf_repo, repo_type="dataset"))
            if cfg.hf_subdir:
                root = root / cfg.hf_subdir
        if not (root / "generate" / "train_data.parquet").exists():
            raise FileNotFoundError(f"{root} is not a concept500 dump "
                                    "(no generate/train_data.parquet)")
        return root

    def _keep(self, df: pl.DataFrame) -> pl.DataFrame:
        # head-subset: first n concepts, and the
        # global concept_id == -1 negative pool rides along (-1 < n)
        if self.config.n_concepts is None:
            return df
        return df.filter(pl.col("concept_id") < self.config.n_concepts)

    def concepts(self) -> list[Concept]:
        lines = (self._root() / "generate" / "metadata.jsonl").read_text().splitlines()
        metas = [json.loads(line) for line in lines if line.strip()]
        if self.config.n_concepts is not None:
            metas = metas[:self.config.n_concepts]
        return [Concept(id=m["concept_id"], name=m["concept"], description="") for m in metas]

    def train_samples(self) -> pl.DataFrame:
        native = pl.read_parquet(self._root() / "generate" / "train_data.parquet")
        return self._keep(canonical_samples(native, TRAIN_META_KEYS))

    def latent_samples(self) -> pl.DataFrame:
        # inference/latent_eval_data.parquet: positive/negative/hard-negative
        # completions per concept -- the input to the certified max-act pass
        # (protocol.latent_max_acts) and to latent metrics; NOT eval_samples
        native = pl.read_parquet(self._root() / "inference" / "latent_eval_data.parquet")
        return self._keep(canonical_samples(native, LATENT_META_KEYS))

    def eval_samples(self) -> pl.DataFrame:
        # axbench SteeringDatasetFactory.create_eval_df, AlpacaEval branch:
        # per-concept pandas .sample(n, random_state=CONCEPT_ID) -- pandas is
        # imported lazily for exact row-selection parity with those runs
        import pandas as pd

        cfg = self.config
        if cfg.alpaca_path is not None:
            path = cfg.alpaca_path
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json",
                                   repo_type="dataset")
        alpaca = pd.read_json(path)
        n = cfg.eval_prompts_per_concept
        ids, prompts = [], []
        for concept in self.concepts():
            sampled = alpaca.sample(n, random_state=int(concept.id))["instruction"].tolist()
            ids.extend([concept.id] * n)
            prompts.extend(sampled)
        return pl.DataFrame({"concept_id": ids, "prompt": prompts}).with_columns(
            pl.lit(None, dtype=pl.String).alias("response"),
            pl.lit(None).alias("meta"))

    def polar_train_samples(self) -> pl.DataFrame:
        cfg = self.config
        cache = Path(cfg.dpo_dir) / "polar_train_data.parquet" if cfg.dpo_dir else None
        if cache is not None and cache.exists():
            polar = self._keep(pl.read_parquet(cache))
            missing = (set(c.id for c in self.concepts())
                       - set(polar.get_column("concept_id").unique().to_list()))
            if missing:
                raise RuntimeError(
                    f"dpo cache {cache} covers only part of the requested concepts "
                    f"(missing ids {sorted(missing)}) -- it was generated for a smaller "
                    "subset; delete it (or point dpo_dir elsewhere) to regenerate")
            return polar
        if self._generator is None:
            state = (f"dpo_dir {cfg.dpo_dir!r} has no polar_train_data.parquet"
                     if cache is not None else "no dpo_dir is set")
            raise RuntimeError(
                "concept500 polar pairs are generated, not shipped (losing responses "
                f"come from the subject model) and {state} -- point dpo_dir at a "
                "populated cache, or bind_generator(GeneratorLM) to generate them"
                + (" (they will be cached to dpo_dir)" if cache is not None else ""))
        if self._generator.ref.model_id != cfg.subject.model_id:
            raise RuntimeError(
                f"bound generator is {self._generator.ref.model_id!r} but config.subject "
                f"is {cfg.subject.model_id!r} -- losing responses must come from the subject")
        polar = self._generate_polar()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            polar.write_parquet(cache)
        return polar

    def _generate_polar(self) -> pl.DataFrame:
        # create_dpo_df, keep_orig_axbench_format branch: winning = the
        # concept-positive output, losing = the subject's own unsteered GREEDY
        # continuation of the same prompt, max_new_tokens = 1.5 x output_length
        cfg = self.config
        gen = GenParams(max_new_tokens=int(cfg.dpo_output_length * 1.5),
                        do_sample=False, batch_size=cfg.dpo_batch_size, seed=None)
        train = self.train_samples()
        label = pl.col("meta").struct.field(META_LABEL)
        frames = []
        for concept in self.concepts():
            positive = train.filter((pl.col("concept_id") == concept.id) & (label == LABEL_POS))
            chat = [self._generator.chat_prompt(p) for p in positive.get_column("prompt").to_list()]
            losing = self._generator.generate(chat, gen)
            frames.append(positive.select(
                "concept_id", "prompt", pl.col("response").alias("winning"), "meta"
            ).with_columns(pl.Series("losing", losing)))
        return pl.concat(frames).select("concept_id", "prompt", "winning", "losing", "meta")

    def evals(self) -> list:
        # AxBench's own judge protocol (gpt-4o-mini, 0-2 scales, harmonic-mean
        # leaderboard aggregation) is not implemented here, and no substitute
        # judge is baked in: judge choice is a runscript decision
        raise NotImplementedError(
            "Concept500Benchmark defines data, not judges: AxBench's gpt-4o-mini "
            "leaderboard judges are not implemented. Assemble an eval set in the "
            "runscript -- steergym.evals.judges provides a local digit-judge panel "
            "(scores are not comparable to leaderboard numbers without calibration).")
