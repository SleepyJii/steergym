"""Transform steering: methods whose artifact is a map h -> f(h), not a row.

`TransformSteerContext` is the chassis — it hooks one layer module and
applies an arbitrary per-row transform at configurable positions.
`TransformSteerModel` is the owning-model base: per-concept fitted transform
objects (torch Modules with odesteer's `.steer(X, T)` calling convention),
torch-checkpoint persistence. Concrete methods live in
steergym/odesteer/transforms.py.
"""
from dataclasses import dataclass

from .concept import Concept
from .steer import SteerConfig, SteerContext, SteerModel


@dataclass
class TransformSteerConfig(SteerConfig):
    pass


class TransformSteerContext(SteerContext):
    """Apply `transform.steer(rows [N, d], T) -> [N, d]` on the layer
    module's output. positions="generated" (odesteer's convention,
    steer_position_idx=-1): only position -1 of each forward — with
    KV-cache generation that is the last prompt token at prefill, then
    every newly generated token (GeneratorLM left-pads, so -1 is never a
    pad). positions="all": every position, flattened through the transform
    (= VectorAdditionContext's site for row-addition transforms).

    Contexts are PER-CONCEPT, like every SteerContext in the library
    (SteerModel.context(concept, ...)): one concept's intervention is
    active at a time, and benchmark sweeps iterate concept x dose — the
    convention of every certified path and of odesteer's own harness.
    Batching is over SEQUENCES: a forward's [B] selected rows go through
    the transform in one call.

    The transform runs in its own parameter dtype (float32 fits) and the
    result is cast back to the stream dtype. diagnostics() reports the
    live-accumulated applied ||f(h) - h|| — transforms are state-dependent,
    so the dose cannot be precomputed the way a row addition's can."""

    def __init__(self, module, transform, T: float = 1.0, positions: str = "generated"):
        if positions not in ("all", "generated"):
            raise ValueError(f"positions {positions!r}; expected 'all' or 'generated'")
        self.module = module
        self.transform = transform
        self.T = float(T)
        self.positions = positions
        self._handle = None
        self._rows = 0
        self._delta_sum = 0.0
        self._delta_max = 0.0

    def _apply(self, rows):
        # rows: [N, d] in the stream's dtype/device
        import torch

        self.transform.to(rows.device)
        with torch.no_grad():
            x = rows.float()  # every fit in this family is float32
            out = self.transform.steer(x, T=self.T)
            deltas = (out - x).norm(dim=-1)
            self._rows += len(deltas)
            self._delta_sum += float(deltas.sum())
            self._delta_max = max(self._delta_max, float(deltas.max()))
        return out.to(rows.dtype)

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("context already entered (nested entry would leak a hook)")

        def hook(mod, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.positions == "all":
                B, T_, d = h.shape
                h = self._apply(h.reshape(B * T_, d)).reshape(B, T_, d)
            else:
                h = h.clone()
                h[:, -1, :] = self._apply(h[:, -1, :])
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def diagnostics(self) -> dict:
        return {
            "steered_rows": self._rows,
            "applied_delta_norm_mean": self._delta_sum / self._rows if self._rows else 0.0,
            "applied_delta_norm_max": self._delta_max,
        }


class TransformSteerModel(SteerModel):
    """Base for per-concept transform methods. Subclasses implement
    `_fit_transform(pos_X, neg_X) -> torch Module` (the fitted transform,
    exposing `.steer(X, T)`) and set `weight_name`. Persistence is one torch
    checkpoint: config + per-concept transform state_dicts, rebuilt through
    `_new_transform()` on load."""

    config_cls = TransformSteerConfig
    weight_name: str | None = None

    def __init__(self, config):
        super().__init__(config)
        self.transforms = {}  # concept.id -> fitted transform Module

    def _fit_transform(self, pos_X, neg_X):
        raise NotImplementedError

    def _new_transform(self):
        # an unfitted transform whose buffers load() can restore
        raise NotImplementedError

    def transform_for(self, concept: Concept):
        return self.transforms[concept.id]

    def context(self, concept: Concept, T: float = 1.0,
                positions: str = "generated") -> TransformSteerContext:
        # T is odesteer's vocabulary: the intervention strength (integration
        # time for ODESteer; LinAcT/MiMiC ignore it, matching their classes)
        return TransformSteerContext(self._layer_module(), self.transform_for(concept),
                                     T=T, positions=positions)

    def save(self, path):
        from dataclasses import asdict

        import torch

        torch.save({
            "cls": type(self).__name__,
            "config": {**asdict(self.config), "subject": asdict(self.config.subject)},
            "transforms": {cid: t.state_dict() for cid, t in self.transforms.items()},
        }, path)

    # the exact buffer-key set a fitted transform persists — load() checks
    # it both ways, so a wrong-class or truncated artifact fails loudly at
    # load time, not with an AttributeError mid-generation
    fitted_buffers: frozenset[str] = frozenset()

    def _check_loaded(self, transform):
        # subclass hook: validate restored state against config (shapes etc.)
        pass

    @classmethod
    def load(cls, path):
        import torch

        from .steer import GeneratorRef

        d = torch.load(path, map_location="cpu", weights_only=False)
        if d["cls"] != cls.__name__:
            raise ValueError(f"artifact was saved by {d['cls']}, not {cls.__name__}")
        cfg_d = dict(d["config"])
        cfg = cls.config_cls(**{**cfg_d, "subject": GeneratorRef(**cfg_d["subject"])})
        m = cls(cfg)
        for cid, state in d["transforms"].items():
            if set(state) != set(cls.fitted_buffers):
                raise ValueError(
                    f"concept {cid!r}: artifact buffers {sorted(state)} != "
                    f"expected {sorted(cls.fitted_buffers)}")
            t = m._new_transform()
            # fitted buffers don't exist on an unfitted module (they are
            # registered during fit), so restore by direct registration
            # rather than load_state_dict:
            for name, tensor in state.items():
                parent = t
                *prefix, leaf = name.split(".")
                for p in prefix:
                    parent = getattr(parent, p)
                parent.register_buffer(leaf, tensor)
            m._check_loaded(t)
            m.transforms[cid] = t
        return m
