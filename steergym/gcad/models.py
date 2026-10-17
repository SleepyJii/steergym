"""GCAD (arXiv 2605.10664, "Gated Cropped Attention-Delta"; upstream github
NihiI-obstat/Gated-Cropped-Attention-Delta @ 6c879f4): attention-level
steering with a per-token, data-dependent gate.

Fitted artifact per concept, per layer l in `config.layers`:

    v[l]          = Eq. 8: E_D+[cropped_attn_sys(h_pos)] - E_D-[cropped_attn_sys(h_neg)]
    k_pos[l]      = Eq. 9: post-RoPE K averaged over prompt (D+ rows only)
    avg_demand[l] = Eq. 10's centering constant d-bar (D+ rows only)

(see extract.py for the extraction math, and the package README for the
crop-window generalization this needs for steergym's canonical samples
schema). `layer`/`site` on the base SteerConfig are unused (multi-layer
method, like A-LQR/S-PID)."""
from dataclasses import dataclass

from ..steer import SteerConfig, SteerContext, SteerModel


@dataclass
class GCADConfig(SteerConfig):
    layers: tuple[int, ...] | None = None  # None = every decoder layer (upstream functions'
                                            # own default); paper's per-model restricted ranges
                                            # are a runscript knob -- see README table
    base_coef: float = 2.0                 # c_base, Eq. 10 -- paper Appendix A default (Llama)
    scale: float = 1.5                     # s, Eq. 10 (gate sharpness) -- shared by both paper models
    max_length: int = 1024


class GCADContext(SteerContext):
    """Upstream's `MultiLayerGatedSteerer`: three hooks per layer on
    `self_attn` --

        pre-hook on self_attn   -- captures `position_embeddings` (cos, sin)
        forward hook on q_proj  -- captures pre-RoPE Q
        forward hook on self_attn -- RoPEs the captured Q, computes the
                                      per-token gate (Eq. 9/10), and adds
                                      `coef * v` to the attention output
                                      (Eq. 11) -- BEFORE it is cached or
                                      added to the residual stream for this
                                      layer's own MLP

    `positions="generated"`: only the last position of each forward (the
    last prompt token at prefill, every new token under KV-cache decode --
    `transform.py`'s convention). `positions="all"`: every position.
    `base_coef=0` (dose knob `strength=0`, see `GCADModel.context`) is an
    EXACT identity regardless of v/k_pos/avg_demand/scale -- `coef` is
    `2*0*sigmoid(...) == 0` identically."""

    def __init__(self, layer_modules, layer_indices, v, k_pos, avg_demand, *,
                num_heads: int, num_kv_heads: int, head_dim: int,
                base_coef: float, scale: float, positions: str = "generated"):
        if positions not in ("generated", "all"):
            raise ValueError(f"positions {positions!r}; expected 'generated' or 'all'")
        n = len(layer_indices)
        if not (len(layer_modules) == n == v.shape[0] == k_pos.shape[0] == avg_demand.shape[0]):
            raise ValueError(
                f"layer/v/k_pos/avg_demand counts disagree: "
                f"{len(layer_modules)} modules, {n} indices, {v.shape[0]} v rows, "
                f"{k_pos.shape[0]} k_pos rows, {avg_demand.shape[0]} avg_demand rows")
        self.layer_modules = list(layer_modules)  # module_idx below indexes this list (and v/k_pos/avg_demand rows)
        self.layer_indices = list(layer_indices)  # the model's decoder-layer number for each module_idx
        self.v = v              # [Lc, d] float32
        self.k_pos = k_pos      # [Lc, num_kv_heads, head_dim] float32, post-RoPE
        self.avg_demand = avg_demand  # [Lc] float32
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.base_coef = float(base_coef)
        self.scale = float(scale)
        self.positions = positions
        self._handles = None
        self._q_captured = {}
        self._pos_emb = {}
        self._dev_rows = {}  # module_idx -> (v_row, k_row) on that layer's device (layers may be sharded)
        self._stats = {}     # device -> [rows, sum ||delta||, max ||delta||]

    def _rows_for(self, module_idx, device, dtype):
        cached = self._dev_rows.get(module_idx)
        if cached is None or cached[0].device != device or cached[0].dtype != dtype:
            v_row = self.v[module_idx].to(device=device, dtype=dtype)
            k_row = self.k_pos[module_idx].repeat_interleave(self.num_kv_groups, dim=0).to(
                device=device, dtype=dtype)
            cached = (v_row, k_row)
            self._dev_rows[module_idx] = cached
        return cached

    def _apply(self, module_idx, attn_out, q_captured, pos_emb):
        import torch

        from .rope import apply_rotary_pos_emb

        cos, sin = pos_emb
        q_bhtd = q_captured.transpose(1, 2)  # [B, H, T, D]
        cos_d = cos.to(device=q_bhtd.device, dtype=q_bhtd.dtype)
        sin_d = sin.to(device=q_bhtd.device, dtype=q_bhtd.dtype)
        q_rope, _ = apply_rotary_pos_emb(q_bhtd, q_bhtd, cos_d, sin_d)
        q_rope = q_rope.transpose(1, 2).float()  # [B, T, H, D]

        v_row, k_row = self._rows_for(module_idx, q_rope.device, q_rope.dtype)

        dot_per_head = (q_rope * k_row).sum(dim=-1)                     # [B, T, H]
        demand = dot_per_head.mean(dim=-1) / (self.head_dim ** 0.5)   # [B, T]
        centered = demand - float(self.avg_demand[module_idx])
        coef = 2.0 * self.base_coef * torch.sigmoid(self.scale * centered)  # [B, T]
        delta = coef.unsqueeze(-1) * v_row  # [B, T, d]

        attn_out = attn_out.clone()
        if self.positions == "generated":
            applied = delta[:, -1, :].to(attn_out.dtype)
            attn_out[:, -1, :] = attn_out[:, -1, :] + applied
        else:
            applied = delta.to(attn_out.dtype)
            attn_out = attn_out + applied
            applied = applied.reshape(-1, applied.shape[-1])

        norms = applied.float().norm(dim=-1)
        st = self._stats.get(norms.device)
        if st is None:
            self._stats[norms.device] = [len(norms), norms.sum(), norms.max()]
        else:
            st[0] += len(norms)
            st[1] = st[1] + norms.sum()
            st[2] = torch.maximum(st[2], norms.max())
        return attn_out

    def _make_pre_hook(self, module_idx):
        def hook(module, args, kwargs):
            pe = kwargs.get("position_embeddings")
            if pe is None and len(args) >= 2:
                pe = args[1]
            self._pos_emb[module_idx] = pe
        return hook

    def _make_q_hook(self, module_idx):
        def hook(module, inputs, output):
            bsz, seq_len, _ = output.shape
            self._q_captured[module_idx] = output.view(bsz, seq_len, self.num_heads, self.head_dim)
        return hook

    def _make_attn_hook(self, module_idx):
        def hook(module, inputs, output):
            attn_out = output[0] if isinstance(output, (tuple, list)) else output
            rest = output[1:] if isinstance(output, (tuple, list)) else None

            q = self._q_captured.pop(module_idx, None)
            pe = self._pos_emb.pop(module_idx, None)
            if q is None or pe is None:
                return output

            steered = self._apply(module_idx, attn_out, q, pe)
            if rest is not None:
                return (steered,) + tuple(rest)
            return steered
        return hook

    def __enter__(self):
        if self._handles is not None:
            raise RuntimeError("context already entered (nested entry would leak hooks)")
        if not self.layer_modules:
            raise ValueError("GCADContext needs at least one layer")
        self._handles = []
        for module_idx, module in enumerate(self.layer_modules):
            attn = module.self_attn
            self._handles.append(
                attn.register_forward_pre_hook(self._make_pre_hook(module_idx), with_kwargs=True))
            self._handles.append(attn.q_proj.register_forward_hook(self._make_q_hook(module_idx)))
            self._handles.append(attn.register_forward_hook(self._make_attn_hook(module_idx)))
        return self

    def __exit__(self, *exc):
        if self._handles is not None:
            for h in self._handles:
                h.remove()
            self._handles = None
        self._q_captured.clear()
        self._pos_emb.clear()
        return False

    def diagnostics(self) -> dict:
        rows = sum(s[0] for s in self._stats.values())
        total = sum(float(s[1]) for s in self._stats.values())
        return {
            "steered_rows": rows,
            "applied_delta_norm_mean": total / rows if rows else 0.0,
            "applied_delta_norm_max": max((float(s[2]) for s in self._stats.values()),
                                          default=0.0),
        }


class GCADModel(SteerModel):
    config_cls = GCADConfig
    weight_name = "GCAD"

    def __init__(self, config):
        super().__init__(config)
        self.concepts = {}   # concept.id -> {"v", "k_pos", "avg_demand"}
        self.layer_list = None  # realized layer indices (shared across concepts)

    def fit(self, concepts, samples, log=lambda msg: None):
        import time

        import torch

        from ..axbench.protocol import prefix_suffix_lengths
        from .extract import contrast_rows, cropped_attn_response_avg, k_pos_and_avg_demand

        model, tokenizer = self._require_bound()
        self.layer_list = (list(self.config.layers) if self.config.layers is not None
                          else list(range(model.config.num_hidden_layers)))
        _, suffix_length = prefix_suffix_lengths(tokenizer)

        self.concepts = {}
        for concept in concepts:
            t0 = time.time()
            pos_rows, neg_rows = contrast_rows(samples, concept, tokenizer, suffix_length)
            if not pos_rows or not neg_rows:
                raise ValueError(
                    f"concept {concept.id}: {len(pos_rows)} positive / "
                    f"{len(neg_rows)} negative rows -- GCAD needs both")

            pos_avg = cropped_attn_response_avg(model, tokenizer, pos_rows, self.layer_list)
            neg_avg = cropped_attn_response_avg(model, tokenizer, neg_rows, self.layer_list)
            v = torch.stack([pos_avg[l].mean(dim=0) - neg_avg[l].mean(dim=0)
                             for l in self.layer_list], dim=0).float()

            k_pos, avg_demand = k_pos_and_avg_demand(model, tokenizer, pos_rows, self.layer_list)

            self.concepts[concept.id] = {
                "v": v.cpu(), "k_pos": k_pos.float().cpu(), "avg_demand": avg_demand.float().cpu(),
            }
            log(f"{type(self).weight_name} concept {concept.id} ({concept.name!r}): "
                f"{len(pos_rows)}+{len(neg_rows)} rows, {time.time() - t0:.1f}s")
        return self

    def context(self, concept, strength: float = 1.0, positions: str = "all") -> GCADContext:
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call bind(model) first")
        c = self.concepts[concept.id]
        layers = [self._model.model.layers[l] for l in self.layer_list]
        from .extract import layer_dims

        num_heads, num_kv_heads, _, head_dim = layer_dims(self._model)
        return GCADContext(
            layers, self.layer_list, c["v"], c["k_pos"], c["avg_demand"],
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            base_coef=strength * self.config.base_coef, scale=self.config.scale,
            positions=positions)

    def save(self, path):
        from dataclasses import asdict

        import torch

        torch.save({
            "cls": type(self).__name__,
            "config": {**asdict(self.config), "subject": asdict(self.config.subject)},
            "layer_list": self.layer_list,
            "concepts": self.concepts,
        }, path)

    @classmethod
    def load(cls, path):
        import torch

        from ..steer import GeneratorRef

        d = torch.load(path, map_location="cpu", weights_only=False)
        if d["cls"] != cls.__name__:
            raise ValueError(f"artifact was saved by {d['cls']}, not {cls.__name__}")
        cfg_d = dict(d["config"])
        cfg = cls.config_cls(**{**cfg_d, "subject": GeneratorRef(**cfg_d["subject"])})
        m = cls(cfg)
        m.layer_list = list(d["layer_list"])
        for cid, c in d["concepts"].items():
            if set(c) != {"v", "k_pos", "avg_demand"}:
                raise ValueError(f"concept {cid!r}: artifact keys {sorted(c)} != "
                                 "expected ['avg_demand', 'k_pos', 'v']")
            n = len(m.layer_list)
            if not (c["v"].shape[0] == c["k_pos"].shape[0] == c["avg_demand"].shape[0] == n):
                raise ValueError(
                    f"concept {cid!r}: v/k_pos/avg_demand row counts "
                    f"{c['v'].shape[0]}/{c['k_pos'].shape[0]}/{c['avg_demand'].shape[0]} "
                    f"!= len(layer_list) {n}")
            m.concepts[cid] = c
        return m
