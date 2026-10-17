from dataclasses import asdict, dataclass

from .concept import Concept
from .steer import SteerConfig, SteerContext, SteerModel, GeneratorRef


@dataclass
class RowVectorConfig(SteerConfig):
    unit_norm: bool = True


class VectorAdditionContext(SteerContext):
    """h += strength * steering_row on the layer module's output.
    steering_row is ONE [d] vector — the owning model's row for the single
    concept this context steers (contexts are per-concept library-wide; a
    forward's batch dimension is sequences, all steered by this one
    vector). positions="all" adds at every position (axbench
    intervention_positions "all", the certified baseline convention);
    positions="generated" adds only at position -1 of each forward — with
    KV-cache generation that is the last prompt token at prefill, then
    every newly generated token (odesteer's steer_hook_func,
    steer_position_idx=-1; GeneratorLM left-pads, so -1 is never a pad)."""

    def __init__(self, module, steering_row, strength: float, positions: str = "all"):
        if positions not in ("all", "generated"):
            raise ValueError(f"positions {positions!r}; expected 'all' or 'generated'")
        self.module = module
        self.steering_row = steering_row  # [d]
        self.strength = float(strength)
        self.positions = positions
        self._handle = None

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("context already entered (nested entry would leak a hook)")
        row, strength, positions = self.steering_row, self.strength, self.positions

        def hook(mod, args, output):
            h = output[0] if isinstance(output, tuple) else output
            delta = strength * row.to(device=h.device, dtype=h.dtype)
            if positions == "all":
                h = h + delta
            else:
                h = h.clone()
                h[:, -1, :] = h[:, -1, :] + delta
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def diagnostics(self) -> dict:
        return {"applied_delta_norm": abs(self.strength) * float(self.steering_row.norm())}


class RowVectorModel(SteerModel):
    """One steering vector per concept, held as a [C, d] tensor:
    C = number of concepts (row i steers concept_index-mapped concept i),
    d = the subject model's residual-stream width (e.g. 2304 for gemma-2-2b).
    Dose = factor * max_act[concept] (axbench convention) or explicit
    strength (upstream raw-coefficient paths). Persistence in the axbench
    dump format lives in AxbenchRowModel (steergym/axbench.py)."""

    config_cls = RowVectorConfig

    def __init__(self, config: RowVectorConfig):
        super().__init__(config)
        self.rows = None            # [C, d]
        self.bias = None            # [C] or None
        self.concept_index = {}     # concept.id -> row index in [0, C)
        self.max_acts = {}          # concept.id -> float, recomputed per artifact
        self.raw_norms = None       # pre-unit-norm row norms, when normalized on load

    @classmethod
    def from_rows(cls, config, rows, concept_ids=None, bias=None):
        m = cls(config)
        m.rows = rows.float()
        if config.unit_norm:
            norms = m.rows.norm(dim=-1, keepdim=True)
            m.raw_norms = norms.squeeze(-1).tolist()
            m.rows = m.rows / norms.clamp_min(1e-8)
        m.bias = bias
        ids = list(concept_ids) if concept_ids is not None else list(range(len(m.rows)))
        m.concept_index = {cid: i for i, cid in enumerate(ids)}
        return m

    def set_max_acts(self, max_acts: dict):
        self.max_acts = dict(max_acts)

    def row_for(self, concept: Concept):
        return self.rows[self.concept_index[concept.id]]

    def context(self, concept: Concept, factor: float | None = None,
                strength: float | None = None,
                positions: str = "all") -> VectorAdditionContext:
        # positions="generated" + strength=<raw norm or absolute T> is the
        # odesteer-native application of the same row; "all" stays the
        # certified axbench path.
        if (factor is None) == (strength is None):
            raise ValueError("pass exactly one of factor (× max_act) or strength (raw)")
        if strength is None:
            if concept.id not in self.max_acts:
                raise RuntimeError(f"no max_act for concept {concept.id!r} — "
                                   "set_max_acts from a latent pass first")
            strength = factor * self.max_acts[concept.id]
        return VectorAdditionContext(self._layer_module(), self.row_for(concept),
                                     strength, positions=positions)

    def save(self, path):
        import torch
        torch.save({
            "cls": type(self).__name__,
            "config": {**asdict(self.config), "subject": asdict(self.config.subject)},
            "rows": self.rows, "bias": self.bias,
            "concept_index": self.concept_index, "max_acts": self.max_acts,
            "raw_norms": self.raw_norms,
        }, path)

    @classmethod
    def load(cls, path):
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        cfg_d = dict(d["config"])
        cfg = cls.config_cls(**{**cfg_d, "subject": GeneratorRef(**cfg_d["subject"])})
        m = cls(cfg)
        m.rows, m.bias = d["rows"], d["bias"]
        m.concept_index, m.max_acts = d["concept_index"], d["max_acts"]
        m.raw_norms = d.get("raw_norms")
        return m
