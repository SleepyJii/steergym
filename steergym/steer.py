from dataclasses import dataclass

from .concept import Concept


@dataclass(frozen=True)
class GeneratorRef:
    model_id: str
    revision: str | None = None


@dataclass
class SteerConfig:
    subject: GeneratorRef
    layer: int | None = None
    site: str | None = None


class SteerContext:
    """Context manager over the subject model: hooks and/or prompt render,
    both defaulting to no-op. Coefficients are the owning model's vocabulary."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def render(self, prompt: str) -> str:
        # called on raw user text, before chat templating
        return prompt

    def diagnostics(self) -> dict:
        return {}


class SteerModel:
    config_cls = SteerConfig

    def __init__(self, config: SteerConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def fit(self, concepts: list[Concept], samples):
        # `samples` view per method: train_samples or polar_train_samples
        raise NotImplementedError(f"{type(self).__name__} does not train")

    def bind(self, model, tokenizer=None):
        self._model = model
        self._tokenizer = tokenizer
        return self

    def _require_bound(self):
        # fit() needs both the subject model and its tokenizer (unlike context(),
        # which only needs the model) -- raise a clear error rather than an
        # AttributeError deep inside the training loop.
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                f"{type(self).__name__} not bound to a model+tokenizer -- "
                "call bind(model, tokenizer) before fit()")
        return self._model, self._tokenizer

    def _layer_module(self):
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call bind(model) first")
        if self.config.site not in (None, "resid_post"):
            raise NotImplementedError(f"site {self.config.site!r}; v1 hooks resid_post only")
        return self._model.model.layers[self.config.layer]

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> "SteerModel":
        raise NotImplementedError

    def context(self, concept: Concept, **coeffs) -> SteerContext:
        raise NotImplementedError
