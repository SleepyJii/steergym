"""A-LQR (arXiv 2604.19018; upstream commit c2e0c84): time-varying LQR over
the layer index, tracking a contrastive setpoint at every decoder layer.
Fitted artifact per concept, k = 0..L-1:

    e_k  = mean_pos_k - mean_neg_k      (layer-INPUT activations, last token)
    mu_k = ||e_k||;  v_k = e_k / mu_k
    w_k  = K_k @ v_k                    (K_k from control.riccati_gain_rows)

(v, w, mu) is all that is stored -- never the [L, d, d] gains upstream keeps
resident (~2 GB at 8B); exact by the rank-1 identity in the README. No
multisteer in v1; `layer`/`site` unused (multi-layer method)."""
from dataclasses import dataclass

from ..steer import SteerConfig, SteerContext, SteerModel
from ..training_helpers import pad_stack


@dataclass
class ALQRConfig(SteerConfig):
    q: float = 10.0               # upstream LQRSteering defaults
    r: float = 10.0
    q_f: float = 1.0
    jacobian_prompts: int = 50    # their toxicity/refusal prompt count
    jacobian_max_ctx: int = 24    # their working setting for 8-9B models
    batch_size: int = 6           # extraction batching only, not a paper hparam
    max_length: int = 1024
    seed: int = 42                # jacobian-prompt subsample (upstream is unseeded)


class ALQRContext(SteerContext):
    """Upstream's hook_setpoint_tracking as one forward hook per decoder
    layer. Per layer k, with x_in = the layer's INPUT hidden states:

        alpha_k = lmbda * mu_k - v_k . x_in     (scalar setpoint error)
        u_k     = alpha_k * w_k                 (== upstream's K_k @ (alpha_k v_k))
        layer output at position -1  += u_k     positions="generated"
        layer output, every position += u_k     positions="all" ("A-LQR+")

    "generated" under KV-cache decode = the last prompt token at prefill,
    then every newly generated token (upstream's default branch)."""

    def __init__(self, layer_modules, v, w, mu, lmbda: float, positions: str = "generated"):
        if positions not in ("generated", "all"):
            raise ValueError(f"positions {positions!r}; expected 'generated' or 'all'")
        self.layer_modules = list(layer_modules)
        self.v = v  # [L, d] float32
        self.w = w  # [L, d] float32
        self.beta_star = [lmbda * m for m in mu.tolist()]
        self.positions = positions
        self._handles = None
        self._dev_rows = {}  # k -> (v_k, w_k) on that layer's device (layers may be sharded)
        self._stats = {}     # device -> [rows, sum ||u|| (0-d tensor), max ||u|| (0-d tensor)]

    def _rows_for(self, k, device):
        cached = self._dev_rows.get(k)
        if cached is None or cached[0].device != device:
            cached = (self.v[k].to(device), self.w[k].to(device))
            self._dev_rows[k] = cached
        return cached

    def _apply(self, k, x_in, h):
        import torch

        # x_in: the layer's input [B, T, d]; h: the layer's output [B, T, d];
        # both in the stream's device/dtype. Compute in float32; one rounding
        # on the add, matching upstream's in-place fp32-promoted assignment.
        v_k, w_k = self._rows_for(k, x_in.device)
        beta_k = self.beta_star[k]
        if self.positions == "generated":
            x = x_in[:, -1, :].float()
            alpha = beta_k - x @ v_k          # [B]
            u = alpha[:, None] * w_k          # [B, d]
            h = h.clone()  # never write the hooked output in place (transform.py idiom)
            h[:, -1, :] = (h[:, -1, :].float() + u).to(h.dtype)
            delta_norms = u.norm(dim=-1)
        else:
            x = x_in.float()                  # [B, T, d]
            alpha = beta_k - x @ v_k          # [B, T]
            u = alpha.unsqueeze(-1) * w_k      # [B, T, d]
            h = (h.float() + u).to(h.dtype)
            delta_norms = u.reshape(-1, u.shape[-1]).norm(dim=-1)
        # accumulate on-device; float() syncs are deferred to diagnostics()
        st = self._stats.get(delta_norms.device)
        if st is None:
            self._stats[delta_norms.device] = [len(delta_norms), delta_norms.sum(),
                                               delta_norms.max()]
        else:
            st[0] += len(delta_norms)
            st[1] = st[1] + delta_norms.sum()
            st[2] = torch.maximum(st[2], delta_norms.max())
        return h

    def __enter__(self):
        if self._handles is not None:
            raise RuntimeError("context already entered (nested entry would leak hooks)")
        self._handles = []
        for k, module in enumerate(self.layer_modules):
            def hook(mod, args, output, k=k):
                h = output[0] if isinstance(output, tuple) else output
                h = self._apply(k, args[0], h)
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        if self._handles is not None:
            for handle in self._handles:
                handle.remove()
            self._handles = None
        return False

    def diagnostics(self) -> dict:
        rows = sum(st[0] for st in self._stats.values())
        total = sum(float(st[1]) for st in self._stats.values())
        return {
            "steered_rows": rows,
            "applied_delta_norm_mean": total / rows if rows else 0.0,
            "applied_delta_norm_max": max((float(st[2]) for st in self._stats.values()),
                                          default=0.0),
        }


def _contrast_texts(samples, concept, tokenizer):
    # same pos/neg text derivation as odesteer's concept_contrast_activations,
    # minus the single-layer extraction call -- A-LQR needs every layer
    from ..axbench.protocol import binarize_text, positive_and_negative_rows, prefix_suffix_lengths

    _, suffix_length = prefix_suffix_lengths(tokenizer)
    positive, negative = positive_and_negative_rows(samples, concept)
    pos_texts = [binarize_text(tokenizer, r["prompt"], r["response"], suffix_length)
                 for r in positive.iter_rows(named=True)]
    neg_texts = [binarize_text(tokenizer, r["prompt"], r["response"], suffix_length)
                 for r in negative.iter_rows(named=True)]
    return pos_texts, neg_texts


def _all_layer_last_token(generator, texts, config):
    """[N, L, d] float32 CPU: every layer's INPUT activation at each text's
    final token (hidden_states[0..L-1]; row L, the post-final-norm state,
    unused). Right padding + length-1 indexing, activations.py's convention."""
    import torch

    tokenizer = generator.tokenizer
    tokenizer.padding_side = "right"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = generator.model

    encoded = [tokenizer(t, max_length=config.max_length, truncation=True)["input_ids"]
               for t in texts]
    out = []
    with torch.no_grad():
        for i in range(0, len(encoded), config.batch_size):
            batch_ids = [torch.tensor(e, dtype=torch.long) for e in encoded[i:i + config.batch_size]]
            if not batch_ids:
                continue
            input_ids = pad_stack(batch_ids, pad_id, torch.long, torch)
            lengths = torch.tensor([len(x) for x in batch_ids], dtype=torch.long)
            T = input_ids.shape[1]
            attention_mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)).long()
            result = model(input_ids=input_ids.to(model.device),
                          attention_mask=attention_mask.to(model.device),
                          output_hidden_states=True)
            hidden_states = result.hidden_states  # len L+1; [0..L-1] are layer inputs
            L = len(hidden_states) - 1
            idx = torch.arange(len(lengths))
            per_layer = [hidden_states[k][idx.to(hidden_states[k].device),
                                          (lengths - 1).to(hidden_states[k].device)].cpu()
                         for k in range(L)]  # each [B, d]; index on device, ship only [B, d]
            out.append(torch.stack(per_layer, dim=1))  # [B, L, d]
    return torch.cat(out, dim=0).float()


class ALQRModel(SteerModel):
    config_cls = ALQRConfig
    weight_name = "ALQR"

    def __init__(self, config):
        super().__init__(config)
        self.concepts = {}  # concept.id -> {"v": [L,d], "w": [L,d], "mu": [L]}

    def fit(self, concepts, samples, log=lambda msg: None):
        import random
        import time

        import torch
        from transformers import set_seed

        from .control import riccati_gain_rows
        from .linearize import averaged_block_jacobians
        from ..generator import GeneratorLM

        model, tokenizer = self._require_bound()
        generator = GeneratorLM(self.config.subject, model, tokenizer)
        set_seed(self.config.seed)

        self.concepts = {}
        for concept in concepts:
            t0 = time.time()
            pos_texts, neg_texts = _contrast_texts(samples, concept, tokenizer)
            if not pos_texts or not neg_texts:
                raise ValueError(
                    f"concept {concept.id}: {len(pos_texts)} positive / "
                    f"{len(neg_texts)} negative rows -- A-LQR needs both")
            pos_X = _all_layer_last_token(generator, pos_texts, self.config)  # [Np, L, d]
            neg_X = _all_layer_last_token(generator, neg_texts, self.config)  # [Nn, L, d]

            e = pos_X.mean(dim=0) - neg_X.mean(dim=0)  # [L, d]
            mu = e.norm(dim=-1)                         # [L]
            zero = mu == 0
            v = torch.zeros_like(e)
            v[~zero] = e[~zero] / mu[~zero].unsqueeze(-1)

            jac_texts = random.Random(self.config.seed).sample(
                pos_texts, min(self.config.jacobian_prompts, len(pos_texts)))
            A = averaged_block_jacobians(
                model, tokenizer, jac_texts, self.config.jacobian_max_ctx,
                log=lambda msg, cid=concept.id: log(f"{type(self).weight_name} concept {cid}: {msg}"))
            # recursion on the subject device (CPU is ~minutes/layer at d>=4096;
            # A is L*d^2 fp32, ~2 GB at 8B / ~5 GB at 14B)
            w = riccati_gain_rows(A.to(model.device), v.to(model.device),
                                  self.config.q, self.config.r, self.config.q_f).cpu()

            self.concepts[concept.id] = {
                "v": v.float().cpu(), "w": w.float().cpu(), "mu": mu.float().cpu(),
            }
            log(f"{type(self).weight_name} concept {concept.id} ({concept.name!r}): "
                f"{len(pos_texts)}+{len(neg_texts)} rows, {len(jac_texts)} jacobian prompts, "
                f"{time.time() - t0:.1f}s")
        return self

    def context(self, concept, lmbda: float = 1.0, positions: str = "generated") -> ALQRContext:
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call bind(model) first")
        c = self.concepts[concept.id]
        layers = self._model.model.layers
        if len(layers) != c["v"].shape[0]:
            raise ValueError(f"artifact has {c['v'].shape[0]} layer rows, bound model "
                             f"has {len(layers)} decoder layers")
        return ALQRContext(layers, c["v"], c["w"], c["mu"], lmbda, positions=positions)

    def save(self, path):
        from dataclasses import asdict

        import torch

        torch.save({
            "cls": type(self).__name__,
            "config": {**asdict(self.config), "subject": asdict(self.config.subject)},
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
        for cid, c in d["concepts"].items():
            if set(c) != {"v", "w", "mu"}:
                raise ValueError(f"concept {cid!r}: artifact keys {sorted(c)} != "
                                 "expected ['mu', 'v', 'w']")
            if c["v"].shape != c["w"].shape:
                raise ValueError(f"concept {cid!r}: v.shape {tuple(c['v'].shape)} != "
                                 f"w.shape {tuple(c['w'].shape)}")
            if len(c["mu"]) != c["v"].shape[0]:
                raise ValueError(f"concept {cid!r}: len(mu) {len(c['mu'])} != "
                                 f"v.shape[0] {c['v'].shape[0]}")
            m.concepts[cid] = c
        return m
