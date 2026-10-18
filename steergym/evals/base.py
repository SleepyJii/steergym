import os
import warnings
from dataclasses import dataclass

import polars as pl

from ..concept import Concept


@dataclass
class EvalConfig:
    pass


class MissingAPIKeyWarning(UserWarning):
    """An Eval that reads an API key was constructed while that key's env var
    is unset — score() will fail. Raised at construction so the cost surface
    of an eval set is visible up front, not deep inside a run."""


class Eval:
    """Per-eval configuration is a dataclass mirror of SteerConfig: every Eval
    takes exactly one `config` (its `config_cls`, defaulted when omitted);
    live resources (a loaded GPU judge) arrive via bind_* methods, never
    config. Constructors are called wherever a run is assembled — a
    Benchmark's evals() or a driver — nothing hidden.

    Resource expectations are declared, not discovered: `execution` says what
    the eval runs on, and an eval that reads an API key names its env var in
    `api_key_env` — construction emits MissingAPIKeyWarning when that var is
    unset, so assembling an eval set reveals what it will cost to run."""

    execution = "local"  # "local" | "gpu" | "api"
    api_key_env: str | None = None  # env var an "api" eval reads at score() time
    config_cls = EvalConfig

    def __init__(self, config: EvalConfig | None = None):
        self.config = config if config is not None else self.config_cls()
        if self.api_key_env and not os.environ.get(self.api_key_env):
            warnings.warn(
                f"{type(self).__name__} expects ${self.api_key_env}, which is not set "
                "in the environment — score() will fail without it",
                MissingAPIKeyWarning, stacklevel=2)

    @property
    def name(self) -> str:
        return type(self).__name__

    def score(self, rows: pl.DataFrame, concepts: dict[int | str, Concept]) -> dict[str, list]:
        # rows: results-table slice (concept_id, prompt, response, ...; meta where
        # the eval needs it); returns {column_name: values aligned to rows}
        raise NotImplementedError
