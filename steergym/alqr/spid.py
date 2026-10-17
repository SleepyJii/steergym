"""S-PID (Setpoint-PID; arXiv 2604.19018, upstream commit c2e0c84,
steer/PIDsteering.py's `track_setpoint`/`hook_setpoint_tracking`): the
paper's own closed-loop ablation of A-LQR -- same contrastive artifact
(v, mu), the Riccati gain K replaced by a per-layer PID over the live
setpoint residual. No jacobians, no Riccati -- fit is the DiffMean half of
ALQRModel.fit only. Sibling module to models.py (kept separate so A-LQR's
jacobian/Riccati machinery and S-PID's controller-state machinery don't
share a file); __init__.py re-exports both -- `import steergym` stays
torch-free, this module's own top level is dataclass + intra-package only."""
from dataclasses import dataclass

from ..steer import SteerConfig, SteerContext, SteerModel


@dataclass
class SPIDConfig(SteerConfig):
    batch_size: int = 6           # extraction batching only, not a paper hparam
    max_length: int = 1024
    seed: int = 42                 # no jacobian subsample here -- kept for ALQRConfig parity (fit still set_seed()s)
    # PIDSteering.__init__ defaults (PIDsteering.py:26-28). NO released script uses these --
    # tqa_evalPID.py:129-133 / tox_evalPID.py:128 both pin llama-3-8B to (kp, ki, kd) = (0.1, 0.1, 0.0).
    kp: float = 10.0
    ki: float = 10.0
    kd: float = 1.0
    integral_reset_every: int = 10   # upstream hard-codes `layer_idx % 10 == 0`; 0/None disables
    d_reset_between_steps: bool = False  # faithful default: e_prev leaks across decode steps


class SPIDContext(SteerContext):
    """hook_setpoint_tracking, one forward hook per decoder layer. Per layer
    k, x_in = the layer's INPUT hidden states, position -1 only (upstream
    S-PID has no positions='all' variant):

        alpha_k = lmbda*mu_k - v_k . x_in         [B]
        e_k     = alpha_k * v_k                    [B, d]
        e_sum  += e_k
        if integral_reset_every and k % integral_reset_every == 0: e_sum = 0   (AFTER accumulate)
        u_k     = Kp*e_k + Ki*e_sum + Kd*(e_k - e_prev)
        e_prev  = e_k
        layer output at position -1 += u_k

    e_sum/e_prev are controller state spanning the whole decode (upstream
    resets both once per track_setpoint call = once per generate()). Our
    context instead spans MANY generate() calls, so _state_for reconstructs
    the per-call reset from what's observable at the hook: a prefill (T>1
    at the first hooked layer) means a new call started."""

    def __init__(self, layer_modules, v, mu, kp, ki, kd, lmbda: float,
                 integral_reset_every: int = 10, d_reset_between_steps: bool = False,
                 positions: str = "generated"):
        if positions != "generated":
            raise ValueError(f"positions {positions!r}; S-PID only supports 'generated' "
                              "(upstream has no all-tokens variant)")
        self.layer_modules = list(layer_modules)
        self.v = v  # [L, d] float32
        self.beta_star = [lmbda * m for m in mu.tolist()]
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_reset_every = integral_reset_every
        self.d_reset_between_steps = d_reset_between_steps
        self._handles = None
        self._dev_rows = {}       # k -> v_k on that layer's device (layers may be sharded)
        self._state = None        # (e_sum, e_prev), both [B, d] fp32, on the CURRENT layer's device
        self._batch_size = None
        self._stats = {}          # device -> [rows, sum ||u||, max ||u||]
        self._integral_max = {}   # device -> max ||e_sum|| seen (windup telemetry)

    def _rows_for(self, k, device):
        cached = self._dev_rows.get(k)
        if cached is None or cached.device != device:
            cached = self.v[k].to(device)
            self._dev_rows[k] = cached
        return cached

    def _state_for(self, k, device, B, T):
        import torch

        if k == 0:
            # device is NOT a reset trigger: on a sharded model the last
            # layer's device differs from layer 0's at every decode step, and
            # keying the reset on it would silently zero e_prev each step
            # (= d_reset_between_steps without asking). Migration below covers
            # the device hop, same as the k > 0 case.
            new_pass = self._state is None or self._batch_size != B or T > 1
            if new_pass:
                d = self.v.shape[-1]
                self._state = (torch.zeros(B, d, dtype=torch.float32, device=device),
                                torch.zeros(B, d, dtype=torch.float32, device=device))
                self._batch_size = B
                return self._state
            if self.d_reset_between_steps:
                self._state = (self._state[0], torch.zeros_like(self._state[1]))
        if self._state[0].device != device:
            self._state = (self._state[0].to(device), self._state[1].to(device))
        return self._state

    def _apply(self, k, x_in, h):
        import torch

        v_k = self._rows_for(k, x_in.device)
        beta_k = self.beta_star[k]
        x = x_in[:, -1, :].float()
        e_sum, e_prev = self._state_for(k, x_in.device, x.shape[0], x_in.shape[1])

        alpha = beta_k - x @ v_k               # [B]
        e = alpha[:, None] * v_k               # [B, d]
        e_sum = e_sum + e
        if self.integral_reset_every and k % self.integral_reset_every == 0:
            e_sum = e_sum * 0
        u = self.kp * e + self.ki * e_sum + self.kd * (e - e_prev)
        if not torch.isfinite(u).all():
            raise RuntimeError(f"S-PID layer {k}: u contains NaN or Inf")
        self._state = (e_sum, e)

        h = h.clone()  # never write the hooked output in place (transform.py idiom)
        h[:, -1, :] = (h[:, -1, :].float() + u).to(h.dtype)
        delta_norms = u.norm(dim=-1)

        st = self._stats.get(delta_norms.device)
        if st is None:
            self._stats[delta_norms.device] = [len(delta_norms), delta_norms.sum(),
                                               delta_norms.max()]
        else:
            st[0] += len(delta_norms)
            st[1] = st[1] + delta_norms.sum()
            st[2] = torch.maximum(st[2], delta_norms.max())

        integral_norms = e_sum.norm(dim=-1)
        cur = self._integral_max.get(integral_norms.device)
        self._integral_max[integral_norms.device] = (
            integral_norms.max() if cur is None else torch.maximum(cur, integral_norms.max()))
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
            "integral_norm_max": max((float(v) for v in self._integral_max.values()),
                                     default=0.0),
        }


class SPIDModel(SteerModel):
    config_cls = SPIDConfig
    weight_name = "SPID"

    def __init__(self, config):
        super().__init__(config)
        self.concepts = {}  # concept.id -> {"v": [L,d], "mu": [L]}

    def fit(self, concepts, samples, log=lambda msg: None):
        import time

        import torch
        from transformers import set_seed

        from .models import _all_layer_last_token, _contrast_texts
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
                    f"{len(neg_texts)} negative rows -- S-PID needs both")
            pos_X = _all_layer_last_token(generator, pos_texts, self.config)  # [Np, L, d]
            neg_X = _all_layer_last_token(generator, neg_texts, self.config)  # [Nn, L, d]

            e = pos_X.mean(dim=0) - neg_X.mean(dim=0)  # [L, d]
            mu = e.norm(dim=-1)                         # [L]
            zero = mu == 0
            v = torch.zeros_like(e)
            v[~zero] = e[~zero] / mu[~zero].unsqueeze(-1)

            self.concepts[concept.id] = {"v": v.float().cpu(), "mu": mu.float().cpu()}
            log(f"{type(self).weight_name} concept {concept.id} ({concept.name!r}): "
                f"{len(pos_texts)}+{len(neg_texts)} rows, {time.time() - t0:.1f}s")
        return self

    @classmethod
    def from_alqr(cls, alqr_model, **overrides):
        # upstream itself feeds identical (v, mu) to both controllers (A unused by the PID
        # path) -- this is that same seam, never a recompute.
        fields = {
            "subject": alqr_model.config.subject,
            "batch_size": alqr_model.config.batch_size,
            "max_length": alqr_model.config.max_length,
            "seed": alqr_model.config.seed,
            **overrides,
        }
        m = cls(cls.config_cls(**fields))
        m.concepts = {cid: {"v": c["v"], "mu": c["mu"]} for cid, c in alqr_model.concepts.items()}
        return m

    def context(self, concept, lmbda: float = 1.0, positions: str = "generated") -> SPIDContext:
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call bind(model) first")
        c = self.concepts[concept.id]
        layers = self._model.model.layers
        if len(layers) != c["v"].shape[0]:
            raise ValueError(f"artifact has {c['v'].shape[0]} layer rows, bound model "
                             f"has {len(layers)} decoder layers")
        return SPIDContext(layers, c["v"], c["mu"], self.config.kp, self.config.ki,
                           self.config.kd, lmbda,
                           integral_reset_every=self.config.integral_reset_every,
                           d_reset_between_steps=self.config.d_reset_between_steps,
                           positions=positions)

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
            if set(c) != {"v", "mu"}:
                raise ValueError(f"concept {cid!r}: artifact keys {sorted(c)} != "
                                 "expected ['mu', 'v']")
            if len(c["mu"]) != c["v"].shape[0]:
                raise ValueError(f"concept {cid!r}: len(mu) {len(c['mu'])} != "
                                 f"v.shape[0] {c['v'].shape[0]}")
            m.concepts[cid] = c
        return m
