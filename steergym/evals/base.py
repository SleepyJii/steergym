from dataclasses import dataclass

import polars as pl

from ..concept import Concept


@dataclass
class EvalConfig:
    pass


class Eval:
    """Per-eval configuration is a dataclass mirror of SteerConfig: every Eval
    takes exactly one `config` (its `config_cls`, defaulted when omitted);
    live resources (a loaded GPU judge) arrive via bind_* methods, never
    config. Constructors are called wherever a run is assembled — a
    Benchmark's evals() or a driver — nothing hidden."""

    execution = "local"  # "local" | "gpu" | "api"
    config_cls = EvalConfig

    def __init__(self, config: EvalConfig | None = None):
        self.config = config if config is not None else self.config_cls()

    @property
    def name(self) -> str:
        return type(self).__name__

    def score(self, rows: pl.DataFrame, concepts: dict[int | str, Concept]) -> dict[str, list]:
        # rows: results-table slice (concept_id, prompt, response, ...; meta where
        # the eval needs it); returns {column_name: values aligned to rows}
        raise NotImplementedError
