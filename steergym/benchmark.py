from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from .concept import Concept

if TYPE_CHECKING:  # avoid importing the evals aggregator (it imports every submodule)
    from .evals.base import Eval

SAMPLE_COLUMNS = ["concept_id", "prompt", "response", "meta"]  # response nullable
POLAR_COLUMNS = ["concept_id", "prompt", "winning", "losing", "meta"]

# the ONE meta key with cross-benchmark semantics (contrastive fits and the
# polar derivation read it); every other meta key is benchmark-private and
# defined in its own submodule (e.g. ifeval's CHECKER_META_KEY)
META_LABEL = "label"  # polarity/arm: "pos"/"neg" conventional, benchmarks may extend

LABEL_POS = "pos"
LABEL_NEG = "neg"


def validate_samples(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in SAMPLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"samples missing columns: {missing}")
    return df


def meta_col(df: pl.DataFrame, key: str) -> pl.Series:
    meta = df.get_column("meta")
    if meta.dtype != pl.Struct or key not in meta.struct.fields:
        return pl.Series(key, [None] * df.height)
    return meta.struct.field(key)


@dataclass
class BenchmarkConfig:
    pass


class Benchmark:
    """Config mirrors the model/eval pattern: one dataclass per benchmark
    (`config_cls`, defaulted when omitted); data sources are config, live
    resources (a loaded GeneratorLM for generative stages) arrive via bind_*.
    Coefficient grids and result slicing belong to runscripts, not here."""

    config_cls = BenchmarkConfig

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config if config is not None else self.config_cls()

    def concepts(self) -> list[Concept]:
        raise NotImplementedError

    def train_samples(self) -> pl.DataFrame:
        raise NotImplementedError

    def eval_samples(self) -> pl.DataFrame:
        raise NotImplementedError


class SteerBenchmark(Benchmark):
    def evals(self) -> list["Eval"]:
        raise NotImplementedError


class PolarBenchmark(Benchmark):
    losing_label = LABEL_NEG

    def polar_train_samples(self) -> pl.DataFrame:
        # winning = LABEL_POS response, losing = same-prompt losing_label response
        label = pl.col("meta").struct.field(META_LABEL)
        s = self.train_samples().filter(pl.col("response").is_not_null())
        pos = s.filter(label == LABEL_POS)
        los = s.filter(label == self.losing_label).select(
            "concept_id", "prompt", pl.col("response").alias("losing"))
        return (pos.join(los, on=["concept_id", "prompt"])
                .select("concept_id", "prompt",
                        pl.col("response").alias("winning"), "losing", "meta"))


class ProbeBenchmark(Benchmark):
    # detector metrics over labeled samples; deliberately thin until first adapter
    pass
