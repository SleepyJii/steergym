"""Per-unit transport maps for the AcT family (Apple ml-act, arXiv
2410.23054). Every map is a plain `nn.Module` exposing `.transport(z) ->
z'` (the raw, unmasked map) plus three buffers `mask`/`q_lo`/`q_hi` that
`MultiModuleTransformContext` (context.py) uses to decide, per unit, where
`transport` actually applies: strictly inside `q_lo < z < q_hi` (ml-act's
quantile support filter, hooks/transport.py:251-253/:486-488); the std-mask
gate (BOTH groups' std exceeding `std_eps`, :201-209's `std1 > eps & std2 >
eps`) applies to Mean only -- Linear-AcT's mask is wide-open (upstream's
LearnableOTHook gates only on the quantile support, transport.py:461-503);
outside either condition z passes through unmodified. Module level imports
torch/nn directly (nn.Module subclassing needs `nn` bound at
class-definition time) -- this package IS reachable from a bare `import
steergym` (steergym.models -> acttransport.models -> acttransport package
__init__), so `.maps`'s exports are re-exported lazily from the package
__init__ (PEP 562 module `__getattr__`) rather than imported eagerly there.
"""
import torch
from torch import nn

_QUANTILES = {
    # ml-act act/utils/quantiles.py compute_quantiles' keys, same (lo, hi)
    # fractions; "q_all" is ours (the hook's own [-1e6, 1e6] "no masking"
    # sentinel, done here with real +/-inf -- exact at any realistic scale).
    "q_0_100": (0.00, 1.00),
    "q_0.5_99.5": (0.005, 0.995),
    "q_1_99": (0.01, 0.99),
    "q_2_98": (0.02, 0.98),
    "q_5_95": (0.05, 0.95),
    "q_10_90": (0.10, 0.90),
    "q_20_80": (0.20, 0.80),
    "q_30_70": (0.30, 0.70),
    "q_40_60": (0.40, 0.60),
}


# torch.quantile refuses inputs over 2^24 elements; quantiles are per-unit
# (dim=0), so column chunks keep each call under the limit bit-identically
_QUANTILE_NUMEL_LIMIT = 2 ** 24


def support_bounds(x: torch.Tensor, support: str):
    """(q_lo [d], q_hi [d]) of x's own per-unit distribution, same dtype as
    x. "q_all" -> (-inf, inf): the support mask is then true everywhere."""
    if support == "q_all":
        d = x.shape[-1]
        inf = torch.full((d,), float("inf"), dtype=x.dtype, device=x.device)
        return -inf, inf
    if support not in _QUANTILES:
        raise ValueError(f"support {support!r}; expected 'q_all' or one of {sorted(_QUANTILES)}")
    lo, hi = _QUANTILES[support]
    if x.numel() <= _QUANTILE_NUMEL_LIMIT:
        return torch.quantile(x, lo, dim=0), torch.quantile(x, hi, dim=0)
    cols = max(1, _QUANTILE_NUMEL_LIMIT // max(1, x.shape[0]))
    los, his = [], []
    for i in range(0, x.shape[-1], cols):
        sl = x[:, i:i + cols]
        los.append(torch.quantile(sl, lo, dim=0))
        his.append(torch.quantile(sl, hi, dim=0))
    return torch.cat(los), torch.cat(his)


def std_mask(std_a: torch.Tensor, std_b: torch.Tensor, std_eps: float) -> torch.Tensor:
    """ml-act hooks/transport.py:201-209: a unit is live only when BOTH
    groups' std exceeds std_eps -- a near-constant unit on either side is
    never transported (0/0 and degenerate-OLS guard)."""
    return (std_a > std_eps) & (std_b > std_eps)


class MeanShiftMap(nn.Module):
    """Mean-AcT (ml-act OnlyMeanHook, hooks/transport.py:273-307; forward
    at :227-270 with hook_onlymean=True): per-unit pure mean shift,
    T(z) = z - mu1 + mu2. mu1/mu2/std fit in float64 (:185-190 upstream's
    responses.to(float64) to avoid overflow); buffers stored float32.
    Support = the SOURCE group's own quantiles at the configured key
    (:218-226 upstream computes quantiles_dict_src from the label=True
    "source" group only)."""

    def fit(self, source_X: torch.Tensor, target_X: torch.Tensor,
            support: str = "q_all", std_eps: float = 1e-4) -> "MeanShiftMap":
        src, tgt = source_X.double(), target_X.double()
        mu1, std1 = src.mean(dim=0), src.std(dim=0)
        mu2, std2 = tgt.mean(dim=0), tgt.std(dim=0)
        q_lo, q_hi = support_bounds(src, support)
        self.register_buffer("mu1", mu1.float())
        self.register_buffer("mu2", mu2.float())
        self.register_buffer("mask", std_mask(std1, std2, std_eps))
        self.register_buffer("q_lo", q_lo.float())
        self.register_buffer("q_hi", q_hi.float())
        return self

    def transport(self, z: torch.Tensor) -> torch.Tensor:
        return z - self.mu1 + self.mu2


class IndependentSortOTMap(nn.Module):
    """Linear-AcT (ml-act LinearOTHook, hooks/transport.py:506-528) + 1D OT
    pairing (optimal_transport/ot_maps.py solve_ot_1d:9-21, "OT 1D for same
    number of points amounts to sorting") + the closed-form line fit
    (optimal_transport/archs.py LinearProj.optimize:30-54): per-unit
    independent sort pairs the two empirical quantile functions (equal n
    via a deterministic prefix balance here -- upstream's balancing lives
    in its dataloader, unshipped), then OLS WITH an intercept:

        w = sum(x~ * y~) / sum(x~^2),   b = m_y - w * m_x

    x = sorted SOURCE (+ seeded 1e-8 noise, archs.py:37's "prevent
    divisions by 0" -- no other epsilon guards the denominator, matching
    upstream exactly), y = sorted TARGET; x~/y~ center on the PRE-noise
    means m_x/m_y (archs.py computes m_x/m_y BEFORE adding the noise --
    bitwise-faithful, not a re-centering after the noise is added).
    T(z) = w*z + b.

    NO std mask: upstream's LearnableOTHook (transport.py:461-503) gates
    its forward only on the quantile support, never on std -- the std-mask
    gate is specific to GaussianOTHook/Mean. `mask` here is wide-open
    (all-true); only `q_lo`/`q_hi` restrict where `transport` applies.

    THE PAPER'S Def 3.1 prints this OLS denominator as sum(y~^2) -- a
    typo; archs.py:41-43's sum(x~^2) is the correct closed-form OLS slope
    and is what this ports (steergym/tests/test_acttransport.py pins the
    direction: the typo formula gives a measurably different w on any
    fixture with std(target) != std(source)).

    Deliberately NOT odesteer's LinearTransportMap (transforms.py's
    LinearTransportMap: truncate-pair, MEAN-CENTERED sort, no noise, a flat
    1e-10 eps on the denominator) -- see the package README's delta
    table."""

    def fit(self, source_X: torch.Tensor, target_X: torch.Tensor,
            support: str = "q_0_100",
            seed: int = 42) -> "IndependentSortOTMap":
        n = min(len(source_X), len(target_X))
        src, tgt = source_X[:n].double(), target_X[:n].double()
        q_lo, q_hi = support_bounds(src, support)

        x_sorted = src.sort(dim=0).values
        y_sorted = tgt.sort(dim=0).values
        m_x = x_sorted.mean(dim=0)
        m_y = y_sorted.mean(dim=0)
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(x_sorted.shape, generator=gen, dtype=torch.float64) * 1e-8
        x_tilde = (x_sorted + noise) - m_x
        y_tilde = y_sorted - m_y
        w = (x_tilde * y_tilde).sum(dim=0) / (x_tilde ** 2).sum(dim=0)
        b = m_y - w * m_x

        self.register_buffer("w", w.float())
        self.register_buffer("b", b.float())
        self.register_buffer("mask", torch.ones(src.shape[-1], dtype=torch.bool))
        self.register_buffer("q_lo", q_lo.float())
        self.register_buffer("q_hi", q_hi.float())
        return self

    def transport(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.w + self.b


class PIDShiftMap(nn.Module):
    """PID-AcT's per-module open-loop correction (Nguyen et al., the AcT
    fork): T_k(z) = z + alpha * u_k, u_k precomputed at fit time (see
    models.py's `_pid_fit_map` / `_pid_repo_compat_u`). support is
    unconditional (q_lo=-inf, q_hi=inf, every unit live) unlike the OT
    maps' quantile gating. CANONICAL mode's `mask` is also wide-open
    (the paper's law has no std gate); `repo_compat` mode carries the
    fork's own std mask (BOTH groups' std > std_eps, same `std_mask` rule
    as Mean -- models.py's `_pid_fit_map` computes and passes it when
    `repo_compat`). `alpha` (the dose) is NOT persisted -- PIDAcTModel.
    context() rebuilds a fresh instance per call from the saved (u, r,
    mask) buffers."""

    def __init__(self, u: torch.Tensor, r: torch.Tensor, alpha: float = 1.0,
                mask: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("u", u)
        self.register_buffer("r", r)
        self.register_buffer("mask", mask if mask is not None else torch.ones_like(u, dtype=torch.bool))
        self.register_buffer("q_lo", torch.full_like(u, float("-inf")))
        self.register_buffer("q_hi", torch.full_like(u, float("inf")))
        self.alpha = float(alpha)

    def transport(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.alpha * self.u.to(dtype=z.dtype)
