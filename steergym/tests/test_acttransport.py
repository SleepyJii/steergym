#!/usr/bin/env python3
# Offline tests for steergym.acttransport (Activation Transport: MeanAcT,
# LinearAcT, PIDAcT): the per-unit maps (maps.py), the multi-module
# context + find_module_names (context.py), the sequential/atonce fit loop
# and the PID law (models.py). CPU torch is enough. Run:
#   PYTHONPATH=<pylibs> python3 steergym/tests/test_acttransport.py
# An optional ACT_DIR-gated leg replays upstream ml-act's own OT fit/hook
# math from the vendored source repo (github.com/apple/ml-act).
import os
import sys
import types
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from steergym.acttransport.context import (
    MODULE_PRESETS,
    ModuleReached,
    MultiModuleTransformContext,
    find_module_names,
    resolve_module_regexes,
)
from steergym.acttransport.maps import (
    IndependentSortOTMap,
    MeanShiftMap,
    PIDShiftMap,
    std_mask,
    support_bounds,
)
from steergym.acttransport.models import (
    LinearAcTConfig,
    LinearAcTModel,
    MeanAcTConfig,
    MeanAcTModel,
    PIDAcTConfig,
    PIDAcTModel,
    _pid_repo_compat_u,
)
from steergym.concept import Concept
from steergym.generator import GeneratorRef

# Resolved now, before section 4 below installs a fake-LM-only `transformers`
# stub (it only defines set_seed) -- the mixed-width real-HF leg (L1/L4(a))
# needs the genuine package's model/config classes regardless of that later
# stubbing. See test_alqr.py's test_acttransport.py-identical precedent for
# why a bare `import transformers` reference isn't enough once the stub is
# installed (transformers is a lazy-attribute module keyed off sys.modules).
try:
    import transformers as _real_transformers
    _RealLlamaConfig = _real_transformers.LlamaConfig
    _RealLlamaForCausalLM = _real_transformers.LlamaForCausalLM
except ImportError:
    _real_transformers = None

FAILURES = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


SUBJECT = GeneratorRef("fake/acttransport-subject")


# ================================================================== 1. maps.py
g1 = torch.Generator().manual_seed(0)
src1 = torch.randn(300, 5, generator=g1) * 2.0 + 1.0
tgt1 = torch.randn(300, 5, generator=g1) * 3.0 + 5.0

mean_map = MeanShiftMap().fit(src1, tgt1, support="q_all", std_eps=1e-4)
check("MeanShiftMap.fit: mu1/mu2 match hand-computed per-unit means (1)",
      torch.allclose(mean_map.mu1, src1.double().mean(dim=0).float(), atol=1e-5)
      and torch.allclose(mean_map.mu2, tgt1.double().mean(dim=0).float(), atol=1e-5))
z1 = torch.randn(7, 5)
check("MeanShiftMap.transport: T(z) = z - mu1 + mu2 (1)",
      torch.allclose(mean_map.transport(z1), z1 - mean_map.mu1 + mean_map.mu2))
check("MeanShiftMap: support='q_all' -> q_lo/q_hi are +-inf (1)",
      bool(torch.all(mean_map.q_lo == float("-inf"))) and bool(torch.all(mean_map.q_hi == float("inf"))))

# (1b) std mask: a constant unit on the source side is never live, even
# though its "mean" is well defined
src_const = src1.clone()
src_const[:, 2] = 3.0  # unit 2 now has std=0 on the source side
mean_map_const = MeanShiftMap().fit(src_const, tgt1, support="q_all", std_eps=1e-4)
check("std_mask: a std=0 source unit is masked off (1b)",
      bool(mean_map_const.mask[2].item() is False) and bool(mean_map_const.mask.sum() == 4))
check("std_mask: matches the standalone std_mask() helper (1b)",
      torch.equal(mean_map_const.mask,
                  std_mask(src_const.double().std(dim=0), tgt1.double().std(dim=0), 1e-4)))

# (1c) support strictness: boundary z == q_lo/q_hi is EXCLUDED (strict <),
# constructed by hand so the context (section 2) can probe it precisely
hand_map = MeanShiftMap()
hand_map.register_buffer("mu1", torch.zeros(3))
hand_map.register_buffer("mu2", torch.ones(3) * 10.0)
hand_map.register_buffer("mask", torch.tensor([True, True, True]))
hand_map.register_buffer("q_lo", torch.tensor([-1.0, -1.0, -1.0]))
hand_map.register_buffer("q_hi", torch.tensor([1.0, 1.0, 1.0]))
boundary_in = torch.tensor([[0.0, -0.999, 0.999]])   # all strictly inside
boundary_out = torch.tensor([[1.0, -1.0, 2.0]])      # unit0 at q_hi, unit1 at q_lo, unit2 outside
inside_pool_mask = hand_map.mask & (hand_map.q_lo < boundary_in) & (boundary_in < hand_map.q_hi)
outside_pool_mask = hand_map.mask & (hand_map.q_lo < boundary_out) & (boundary_out < hand_map.q_hi)
check("support mask: strictly-inside point is live on every unit (1c)",
      bool(torch.all(inside_pool_mask)))
check("support mask: boundary-exact points are EXCLUDED (strict <) (1c)",
      not bool(torch.any(outside_pool_mask)))

# (1d) IndependentSortOTMap: OLS-denominator direction pinned -- the
# paper's printed Def 3.1 denominator (sum(y~^2)) differs measurably from
# the code's correct sum(x~^2) on data with std(target) != std(source);
# this test FAILS if the typo formula were implemented instead.
lin_map = IndependentSortOTMap().fit(src1, tgt1, support="q_0_100", seed=42)
x_sorted = src1.double().sort(dim=0).values
y_sorted = tgt1.double().sort(dim=0).values
m_x, m_y = x_sorted.mean(dim=0), y_sorted.mean(dim=0)
noise_gen = torch.Generator().manual_seed(42)
noise = torch.randn(x_sorted.shape, generator=noise_gen, dtype=torch.float64) * 1e-8
x_tilde = (x_sorted + noise) - m_x
y_tilde = y_sorted - m_y
w_code = ((x_tilde * y_tilde).sum(dim=0) / (x_tilde ** 2).sum(dim=0)).float()
w_typo = ((x_tilde * y_tilde).sum(dim=0) / (y_tilde ** 2).sum(dim=0)).float()
check("IndependentSortOTMap: w matches the CODE's sum(x~^2) denominator (1d)",
      torch.allclose(lin_map.w, w_code, atol=1e-4))
check("IndependentSortOTMap: w does NOT match the paper-typo sum(y~^2) denominator (1d)",
      not torch.allclose(lin_map.w, w_typo, rtol=0.2))
b_code = m_y - w_code.double() * m_x
check("IndependentSortOTMap: b = m_y - w*m_x on the PRE-noise means (1d)",
      torch.allclose(lin_map.b, b_code.float(), atol=1e-4))
check("IndependentSortOTMap.transport: T(z) = w*z + b (1d)",
      torch.allclose(lin_map.transport(z1[:, :5]), z1[:, :5] * lin_map.w + lin_map.b))

# (1e) seeded noise determinism: same seed => bitwise-identical fit. The
# noise itself (1e-8 scale) is seed-dependent -- a different seed draws a
# different noise tensor -- but that difference is BELOW float32
# precision once w is cast down (archs.py's own "prevent division by 0"
# purpose, not a signal-carrying perturbation), so the seed-dependence
# check compares the noise draws directly, the same way `fit` draws them.
lin_map_b = IndependentSortOTMap().fit(src1, tgt1, support="q_0_100", seed=42)
lin_map_c = IndependentSortOTMap().fit(src1, tgt1, support="q_0_100", seed=43)
check("IndependentSortOTMap: same seed => bitwise identical fit (1e)",
      torch.equal(lin_map.w, lin_map_b.w) and torch.equal(lin_map.b, lin_map_b.b))
noise_42 = torch.randn((300, 5), generator=torch.Generator().manual_seed(42), dtype=torch.float64)
noise_43 = torch.randn((300, 5), generator=torch.Generator().manual_seed(43), dtype=torch.float64)
check("IndependentSortOTMap: different seed => different noise draw (1e)",
      not torch.equal(noise_42, noise_43))

# (1f) balanced prefix truncation: unequal n uses min(n_src, n_tgt) rows of each
src_big = torch.randn(50, 3, generator=torch.Generator().manual_seed(5))
tgt_small = torch.randn(20, 3, generator=torch.Generator().manual_seed(6))
balanced_map = IndependentSortOTMap().fit(src_big, tgt_small, support="q_all", seed=1)
hand_balanced = IndependentSortOTMap().fit(src_big[:20], tgt_small, support="q_all", seed=1)
check("IndependentSortOTMap: unequal n truncates the larger side to the smaller (1f)",
      torch.equal(balanced_map.w, hand_balanced.w) and torch.equal(balanced_map.b, hand_balanced.b))

# (1g) PIDShiftMap: no masking, T(z) = z + alpha*u
u_g = torch.randn(4)
r_g = torch.randn(4)
pid_map = PIDShiftMap(u_g, r_g, alpha=2.0)
z1g = torch.randn(6, 4)
check("PIDShiftMap.transport: z + alpha*u (1g)",
      torch.allclose(pid_map.transport(z1g), z1g + 2.0 * u_g))
check("PIDShiftMap: mask is wide-open, q_lo/q_hi are +-inf (1g)",
      bool(torch.all(pid_map.mask)) and bool(torch.all(pid_map.q_lo == float("-inf")))
      and bool(torch.all(pid_map.q_hi == float("inf"))))

# (1h) support_bounds: bad key raises
try:
    support_bounds(torch.randn(5, 3), "q_bogus")
    bad_support_raised = False
except ValueError:
    bad_support_raised = True
check("support_bounds: unknown support key raises ValueError (1h)", bad_support_raised)

# (1h2) support_bounds: the column-chunked large-input path (torch.quantile
# refuses > 2^24 elements) is bit-identical to the direct path
from steergym.acttransport import maps as _maps_mod

x_1h2 = torch.randn(64, 37, dtype=torch.float64)
lo_direct, hi_direct = support_bounds(x_1h2, "q_5_95")
_saved_limit = _maps_mod._QUANTILE_NUMEL_LIMIT
_maps_mod._QUANTILE_NUMEL_LIMIT = 64 * 5  # forces ~8-column chunks
try:
    lo_chunked, hi_chunked = support_bounds(x_1h2, "q_5_95")
finally:
    _maps_mod._QUANTILE_NUMEL_LIMIT = _saved_limit
check("support_bounds: chunked path bit-identical to direct torch.quantile (1h2)",
      bool(torch.equal(lo_direct, lo_chunked)) and bool(torch.equal(hi_direct, hi_chunked)))

# (1i) ModuleReached: a plain Exception subclass, catchable on its own
check("ModuleReached is an Exception subclass (1i)", issubclass(ModuleReached, Exception))
try:
    raise ModuleReached()
except ModuleReached:
    module_reached_catchable = True
except Exception:
    module_reached_catchable = False
check("ModuleReached: raised and caught by its own type (1i)", module_reached_catchable)


# ================================================================== 2. context.py
class FakeBlock(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, h):
        return (h + torch.tanh(h @ self.M),)


class FakeStack(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.layers = nn.ModuleList(layer_list)


def run_stack(layers, h):
    for layer in layers:
        out = layer(h)
        h = out[0] if isinstance(out, tuple) else out
    return h


g2 = torch.Generator().manual_seed(1)
stack2 = FakeStack([FakeBlock(torch.randn(4, 4, generator=g2) * 0.2) for _ in range(3)])

check("find_module_names: registration order, fullmatch, root skipped (2)",
      find_module_names(stack2, [r"layers\.\d+"]) == ["layers.0", "layers.1", "layers.2"])
check("find_module_names: no match -> empty list (2)",
      find_module_names(stack2, [r"nonexistent"]) == [])
check("find_module_names: de-duplicated across overlapping regexes (2)",
      find_module_names(stack2, [r"layers\.\d+", r".*\.0"]) == ["layers.0", "layers.1", "layers.2"])

check("MODULE_PRESETS: postln/all_ln/icml24 are all present (2)",
      set(MODULE_PRESETS) == {"postln", "all_ln", "icml24"})
cfg_preset = MeanAcTConfig(subject=SUBJECT, module_preset="postln")
check("resolve_module_regexes: preset resolves to MODULE_PRESETS entry (2)",
      resolve_module_regexes(cfg_preset) == MODULE_PRESETS["postln"])
cfg_explicit = MeanAcTConfig(subject=SUBJECT, module_regexes=(r"foo",))
check("resolve_module_regexes: explicit module_regexes wins over preset (2)",
      resolve_module_regexes(cfg_explicit) == (r"foo",))
cfg_bad_preset = MeanAcTConfig(subject=SUBJECT, module_preset="bogus")
try:
    resolve_module_regexes(cfg_bad_preset)
    bad_preset_raised = False
except ValueError:
    bad_preset_raised = True
check("resolve_module_regexes: unknown preset raises ValueError (2)", bad_preset_raised)

maps2 = [MeanShiftMap().fit(torch.randn(60, 4, generator=g2) + 1.0,
                            torch.randn(60, 4, generator=g2) + 3.0) for _ in range(2)]
mods2 = [stack2.layers[0], stack2.layers[1]]
h0 = torch.randn(2, 5, 4, generator=g2)
baseline2 = run_stack(stack2.layers, h0.clone())


def hand_multimodule(positions, lmbda):
    x = h0.clone()
    for k, layer in enumerate(stack2.layers[:2]):
        out = layer(x)[0]
        m = maps2[k]
        if positions == "all":
            flat = out.reshape(-1, 4).float()
            t = m.transport(flat)
            mask = m.mask & (m.q_lo < flat) & (flat < m.q_hi)
            blended = lmbda * t + (1 - lmbda) * flat
            res = torch.where(mask, blended, flat).reshape(out.shape)
        else:
            last = out[:, -1, :].float()
            t = m.transport(last)
            mask = m.mask & (m.q_lo < last) & (last < m.q_hi)
            blended = lmbda * t + (1 - lmbda) * last
            res = out.clone()
            res[:, -1, :] = torch.where(mask, blended, last)
        x = res
    x = stack2.layers[2](x)[0]
    return x


for positions in ("all", "generated", "last"):
    ctx2 = MultiModuleTransformContext(mods2, maps2, lmbda=1.0, positions=positions)
    with ctx2:
        steered2 = run_stack(stack2.layers, h0.clone())
    hand2 = hand_multimodule(positions, 1.0)
    check(f"MultiModuleTransformContext positions={positions!r}: matches hand recompute (2)",
          torch.allclose(steered2, hand2, atol=1e-5))

# lmbda interpolation: 0.5 blend matches the hand formula too (not just 0/1)
ctx2_half = MultiModuleTransformContext(mods2, maps2, lmbda=0.5, positions="all")
with ctx2_half:
    steered2_half = run_stack(stack2.layers, h0.clone())
check("MultiModuleTransformContext: lmbda=0.5 matches the hand-blended formula (2)",
      torch.allclose(steered2_half, hand_multimodule("all", 0.5), atol=1e-5))

ctx2_zero = MultiModuleTransformContext(mods2, maps2, lmbda=0.0, positions="all")
with ctx2_zero:
    steered2_zero = run_stack(stack2.layers, h0.clone())
check("MultiModuleTransformContext: lmbda=0.0 is a no-op (within the support mask) (2)",
      torch.allclose(steered2_zero, baseline2, atol=1e-5))

# diagnostics: per-module breakdown + total, accumulates across forwards
ctx2_diag = MultiModuleTransformContext(mods2, maps2, lmbda=1.0, positions="all")
with ctx2_diag:
    run_stack(stack2.layers, h0.clone())
    d1 = ctx2_diag.diagnostics()
    run_stack(stack2.layers, h0.clone())
    d2 = ctx2_diag.diagnostics()
check("diagnostics: per_module has one entry per installed module (2)",
      set(d1["per_module"]) == {0, 1})
check("diagnostics: steered_rows == modules x batch x positions (2)",
      d1["steered_rows"] == 2 * 2 * 5)
check("diagnostics: a second forward doubles steered_rows, leaves mean unchanged (2)",
      d2["steered_rows"] == 2 * d1["steered_rows"]
      and abs(d2["applied_delta_norm_mean"] - d1["applied_delta_norm_mean"]) < 1e-5)
check("diagnostics: idempotent when called again with no new forward (2)",
      ctx2_diag.diagnostics() == d2)

# L3: in_support_frac -- maps2 are wide-open (support="q_all" default, every
# unit's std > std_eps -- mask also wide-open), so every selected unit-cell
# is in-support: overall AND every per-module entry read exactly 1.0.
check("in_support_frac: wide-open map (q_all support) -> overall == 1.0 (2, L3)",
      d1["in_support_frac"] == 1.0)
check("in_support_frac: wide-open map -> every per_module entry == 1.0 (2, L3)",
      all(v["in_support_frac"] == 1.0 for v in d1["per_module"].values()))
check("in_support_frac: a second forward with the same (wide-open) data leaves it at 1.0 (2, L3)",
      d2["in_support_frac"] == 1.0)


class _PassThroughL3(nn.Module):
    def forward(self, h):
        return h


# a TIGHT quantile support window (q_40_60: only the central 20% of the
# source's own per-unit distribution) applied to the source distribution
# ITSELF must leave a measurable (not 0, not 1) fraction of unit-cells
# in-support -- this fails under an always-true or always-false mask bug.
g_l3 = torch.Generator().manual_seed(61)
tight_src = torch.randn(2000, 4, generator=g_l3)
tight_tgt = torch.randn(2000, 4, generator=g_l3) + 1.0
tight_map = MeanShiftMap().fit(tight_src, tight_tgt, support="q_40_60", std_eps=1e-4)
tight_module = _PassThroughL3()
probe_rows = tight_src.unsqueeze(1)  # [N, T=1, d]
ctx_tight = MultiModuleTransformContext([tight_module], [tight_map], lmbda=1.0, positions="all")
with ctx_tight:
    tight_module(probe_rows)
tight_diag = ctx_tight.diagnostics()
check("in_support_frac: tight q_40_60 support window on its own fit distribution "
      "is measurably <1.0 and >0.0 (2, L3)",
      0.0 < tight_diag["in_support_frac"] < 1.0)
check("in_support_frac: single-module run -> per_module entry equals the overall value (2, L3)",
      tight_diag["per_module"][0]["in_support_frac"] == tight_diag["in_support_frac"])
# close to the nominal 20% (q_40_60's own window), loosely (sampling noise)
check("in_support_frac: tight window lands near its nominal 20% band (2, L3)",
      abs(tight_diag["in_support_frac"] - 0.20) < 0.05)

# re-entry + post-exit + bad-args guards
ctx2_re = MultiModuleTransformContext(mods2, maps2, lmbda=1.0, positions="all")
ctx2_re.__enter__()
try:
    ctx2_re.__enter__()
    reentry_raised = False
except RuntimeError:
    reentry_raised = True
ctx2_re.__exit__(None, None, None)
check("MultiModuleTransformContext: re-entry raises RuntimeError (2)", reentry_raised)
check("MultiModuleTransformContext: after __exit__ a forward is unmodified (2)",
      torch.allclose(run_stack(stack2.layers, h0.clone()), baseline2, atol=1e-6))
try:
    MultiModuleTransformContext(mods2, maps2, positions="bogus")
    bad_positions_raised = False
except ValueError:
    bad_positions_raised = True
check("MultiModuleTransformContext: bad positions raises ValueError (2)", bad_positions_raised)
try:
    MultiModuleTransformContext(mods2, maps2[:1])
    length_mismatch_raised = False
except ValueError:
    length_mismatch_raised = True
check("MultiModuleTransformContext: len(modules) != len(maps) raises ValueError (2)",
      length_mismatch_raised)

# hook leak guard: handle count returns to 0 after __exit__
handles_before = len(stack2.layers[0]._forward_hooks)
with MultiModuleTransformContext(mods2, maps2, positions="all"):
    pass
check("MultiModuleTransformContext: no leaked forward hooks after __exit__ (2)",
      len(stack2.layers[0]._forward_hooks) == handles_before)


# ================================================================== 3. PID law
def mk_r(gen, d=4):
    return torch.randn(d, generator=gen)


g3 = torch.Generator().manual_seed(9)
r_hist3 = [mk_r(g3) for _ in range(4)]  # r(0)..r(3)


def hand_pid(r_hist, kp, ki, kd, inclusive_sum):
    r_k = r_hist[-1]
    prev = r_hist[:-1]
    terms = r_hist if inclusive_sum else prev
    i_k = sum(terms, torch.zeros_like(r_k)) if terms else torch.zeros_like(r_k)
    r_prev = prev[-1] if prev else torch.zeros_like(r_k)
    d_k = r_k - r_prev
    return kp * r_k + ki * i_k + kd * d_k


class _MapStub:
    def __init__(self, r):
        self.r = r


def canonical_u(r_hist_prefix, kp, ki, kd, inclusive_sum):
    maps_so_far = [_MapStub(r) for r in r_hist_prefix[:-1]]
    from steergym.acttransport.models import _pid_fit_map

    pos_pool = torch.zeros(10, 4)
    neg_pool = -r_hist_prefix[-1].unsqueeze(0).expand(10, 4).clone()
    # pos_pool - neg_pool averages to exactly r_hist_prefix[-1]
    fitted = _pid_fit_map(pos_pool, neg_pool, maps_so_far, kp, ki, kd,
                          repo_compat=False, inclusive_sum=inclusive_sum)
    return fitted.u, fitted.r


for k in range(1, 4):
    u_excl, r_got = canonical_u(r_hist3[:k + 1], 1.0, 0.5, 0.2, inclusive_sum=False)
    u_hand_excl = hand_pid(r_hist3[:k + 1], 1.0, 0.5, 0.2, inclusive_sum=False)
    check(f"PID law k={k}: sum EXCLUSIVE of k matches hand formula (3)",
          torch.allclose(u_excl, u_hand_excl, atol=1e-5))
    check(f"PID law k={k}: r(k) captured correctly (3)",
          torch.allclose(r_got, r_hist3[k], atol=1e-5))

    u_incl, _ = canonical_u(r_hist3[:k + 1], 1.0, 0.5, 0.2, inclusive_sum=True)
    u_hand_incl = hand_pid(r_hist3[:k + 1], 1.0, 0.5, 0.2, inclusive_sum=True)
    check(f"PID law k={k}: sum INCLUSIVE of k matches hand formula (3)",
          torch.allclose(u_incl, u_hand_incl, atol=1e-5))
    if k >= 1:
        check(f"PID law k={k}: inclusive_sum differs from exclusive (nonzero Ki) (3)",
              not torch.allclose(u_excl, u_incl))

# k=0 edge: D-term treats r(-1)=0, so d_0 = r(0) (NOT zero); kp=0 isolates
# the D-term alone
u0, r0 = canonical_u(r_hist3[:1], 0.0, 0.0, 1.0, inclusive_sum=False)
check("PID law k=0: D-term with r(-1):=0 gives d_0 = r(0) (3)",
      torch.allclose(u0, r_hist3[0], atol=1e-5))
check("PID law k=0: integral term is zero (no earlier layers) (3)",
      torch.allclose(canonical_u(r_hist3[:1], 0.0, 5.0, 0.0, inclusive_sum=False)[0],
                     torch.zeros(4), atol=1e-5))

# repo_compat: hand-derived 0.7035*r(k) + 0.0035*s_k formula
r_hist_rc = [mk_r(torch.Generator().manual_seed(20 + i)) for i in range(5)]
for k in range(1, 5):
    prefix = r_hist_rc[:k + 1]
    got = _pid_repo_compat_u(prefix)
    prev = prefix[:-1]
    s = torch.cat(prev).mean()
    hand = 0.7035 * prefix[-1] + 0.0035 * s
    check(f"PID repo_compat k={k}: matches hand-derived 0.7035 r(k) + 0.0035 s_k (3)",
          torch.allclose(got, hand, atol=1e-5))
got0 = _pid_repo_compat_u(r_hist_rc[:1])
check("PID repo_compat k=0: no earlier .statedict files on disk -> u = 0.7*r(0) EXACTLY (3)",
      torch.allclose(got0, 0.7 * r_hist_rc[0], atol=1e-6))
check("PID repo_compat k=0: does NOT match the 0.7035*r(0)+0.0035*s_k formula later modules use (3)",
      not torch.allclose(got0, 0.7035 * r_hist_rc[0], atol=1e-4))
rc_u = _pid_repo_compat_u(r_hist_rc[:3])
s_check = torch.cat(r_hist_rc[:2]).mean()
check("PID repo_compat: the SAME scalar s_k is added to every coordinate (3)",
      torch.allclose(rc_u - 0.7035 * r_hist_rc[2], 0.0035 * s_check * torch.ones(4), atol=1e-5))

# M5(b): repo_compat carries the fork's own std mask (same rule as Mean);
# canonical mode's mask stays wide-open (the paper's law has no std gate).
from steergym.acttransport.models import _pid_fit_map  # noqa: E402

pos_pool_std = torch.randn(10, 4, generator=torch.Generator().manual_seed(77)) + 3.0
neg_pool_std = torch.randn(10, 4, generator=torch.Generator().manual_seed(78)) + 1.0
pos_pool_std[:, 1] = 5.0  # unit 1 constant on the pos side -> std=0 there
canonical_map = _pid_fit_map(pos_pool_std, neg_pool_std, [], 1.0, 0.056, 0.01,
                             repo_compat=False, inclusive_sum=False)
check("PID canonical: mask stays wide-open, no std gate (3)",
      bool(torch.all(canonical_map.mask)))
repo_compat_map = _pid_fit_map(pos_pool_std, neg_pool_std, [], 1.0, 0.056, 0.01,
                               repo_compat=True, inclusive_sum=False, std_eps=1e-4)
check("PID repo_compat: carries the std mask -- the constant unit is masked off (3)",
      bool(repo_compat_map.mask[1].item() is False) and bool(repo_compat_map.mask.sum() == 3))

# L2: canonical PID's per-coordinate law needs a UNIFORM-WIDTH module set --
# a mixed-width fake module set (module 0 at d=3, module 1 at d=5) must raise
# under canonical mode (repo_compat=False) but NOT under repo_compat=True
# (its scalar integral survives torch.cat-ing mixed widths along dim=0).
pos_pool_mw0 = torch.randn(5, 3, generator=torch.Generator().manual_seed(81))
neg_pool_mw0 = torch.randn(5, 3, generator=torch.Generator().manual_seed(82))
map0_mw = _pid_fit_map(pos_pool_mw0, neg_pool_mw0, [], 1.0, 0.056, 0.01,
                       repo_compat=False, inclusive_sum=False)
pos_pool_mw1 = torch.randn(5, 5, generator=torch.Generator().manual_seed(83))
neg_pool_mw1 = torch.randn(5, 5, generator=torch.Generator().manual_seed(84))
try:
    _pid_fit_map(pos_pool_mw1, neg_pool_mw1, [map0_mw], 1.0, 0.056, 0.01,
                repo_compat=False, inclusive_sum=False)
    canonical_mixed_width_raised = False
    canonical_mixed_width_msg = ""
except ValueError as e:
    canonical_mixed_width_raised = True
    canonical_mixed_width_msg = str(e)
check("L2: canonical PID (repo_compat=False) on a mixed-width module set raises "
      "ValueError naming the uniform-width requirement (3)",
      canonical_mixed_width_raised and "uniform-width" in canonical_mixed_width_msg)

repo_compat_map0_mw = _pid_fit_map(pos_pool_mw0, neg_pool_mw0, [], 1.0, 0.056, 0.01,
                                   repo_compat=True, inclusive_sum=False)
try:
    _pid_fit_map(pos_pool_mw1, neg_pool_mw1, [repo_compat_map0_mw], 1.0, 0.056, 0.01,
                repo_compat=True, inclusive_sum=False)
    repo_compat_mixed_width_raised = False
except ValueError:
    repo_compat_mixed_width_raised = True
check("L2: repo_compat=True's scalar integral survives the SAME mixed-width "
      "module set that canonical mode rejects (3)",
      not repo_compat_mixed_width_raised)


# ================================================================== 4/5. sequential fit + end-to-end on a fake LM
tf = types.ModuleType("transformers")
tf.set_seed = lambda s: torch.manual_seed(s)
sys.modules["transformers"] = tf

D4, V4, L4, BOS, EOS, PAD = 8, 300, 4, 1, 2, 0


class FakeTok:
    padding_side = "right"
    bos_token = "<bos>"
    eos_token_id = EOS
    pad_token_id = PAD

    def _t(self, c):
        return 3 + ord(c)

    def decode(self, ids):
        return "".join(chr(int(i) - 3) for i in ids if int(i) > 2)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        s = "«" + "".join(m["content"] for m in messages) + "»"
        if not tokenize:
            return self.bos_token + s
        return [BOS] + [self._t(c) for c in s]

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        single = isinstance(texts, str)
        rows_in = [texts] if single else texts
        rows = []
        for t in rows_in:
            ids = [BOS] + [self._t(c) for c in t]
            if truncation and max_length:
                ids = ids[:max_length]
            rows.append(ids)
        return {"input_ids": rows[0] if single else rows}


VISITED = []


class FakeLMLayer(nn.Module):
    def __init__(self, M, idx):
        super().__init__()
        self.M = M
        self.idx = idx

    def forward(self, hidden, attention_mask=None, position_ids=None, position_embeddings=None):
        VISITED.append(self.idx)
        return hidden + torch.tanh(hidden @ self.M)


class FakeLMInner(nn.Module):
    def __init__(self, D, V, L, gen):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.layers = nn.ModuleList(
            [FakeLMLayer(torch.randn(D, D, generator=gen) * 0.1, i) for i in range(L)])


class FakeLM(nn.Module):
    def __init__(self, D, V, L, gen):
        super().__init__()
        self.model = FakeLMInner(D, V, L, gen)
        self.config = types.SimpleNamespace(hidden_size=D)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return types.SimpleNamespace(hidden_states=(h,))


g4 = torch.Generator().manual_seed(3)
lm = FakeLM(D4, V4, L4, g4)
for p in lm.parameters():
    p.requires_grad_(False)
fake_tok = FakeTok()
subject4 = GeneratorRef("fake/acttransport-fit-subject")

import polars as pl  # noqa: E402


def mk_row(concept_id, prompt, response, output_concept, category):
    return {"concept_id": concept_id, "prompt": prompt, "response": response,
            "meta": {"output_concept": output_concept, "category": category, "concept_genre": "text"}}


rows4 = []
for i in range(8):
    rows4.append(mk_row(0, f"tell me about A{i}", "A" * (i + 3) + "!", "conceptA", "positive"))
for i in range(10):
    rows4.append(mk_row(-1, f"neutral prompt {i}", f"a plain neutral reply {i}", "", "negative"))
samples4 = pl.DataFrame(rows4)
concepts4 = [Concept(id=0, name="conceptA", description="")]
REGEX4 = (r"model\.layers\.\d+",)

# ---- (4a) registration order
model_mean = MeanAcTModel(MeanAcTConfig(subject=subject4, module_regexes=REGEX4,
                                        batch_size=4, max_length=32, seed=7)).bind(lm, fake_tok)
model_mean.fit(concepts4, samples4)
check("MeanAcTModel.fit: module_names in registration order (4a)",
      model_mean.module_names == ["model.layers.0", "model.layers.1", "model.layers.2", "model.layers.3"])
check("MeanAcTModel.fit: one map per module (4a)", len(model_mean.maps[0]) == L4)

# ---- (4b) maps<k are active during k's capture -- installing module 0's
# (non-identity) map changes what gets captured at module 1
from steergym.acttransport.models import _module_pool_batch  # noqa: E402

from steergym.acttransport.models import _contrast_texts  # noqa: E402

pos_texts4, neg_texts4 = _contrast_texts(samples4, concepts4[0], fake_tok)
by_name4 = dict(lm.named_modules())
mods4 = [by_name4[n] for n in model_mean.module_names]

unsteered_pool = _module_pool_batch(lm, fake_tok, neg_texts4, 4, 32, "mean",
                                    [], [], mods4[1], 1.0, steered=True)
steered_pool = _module_pool_batch(lm, fake_tok, neg_texts4, 4, 32, "mean",
                                  mods4[:1], [model_mean.maps[0][0]], mods4[1], 1.0, steered=True)
check("sequential fit: installing map<k changes module k's capture (4b)",
      not torch.allclose(unsteered_pool, steered_pool, atol=1e-6))

# ---- (4c) both-branches-steered (Mean/Linear): the TARGET branch is also
# affected by prior maps (installed regardless of label)
unsteered_pos = _module_pool_batch(lm, fake_tok, pos_texts4, 4, 32, "mean",
                                   [], [], mods4[1], 1.0, steered=True)
steered_pos = _module_pool_batch(lm, fake_tok, pos_texts4, 4, 32, "mean",
                                 mods4[:1], [model_mean.maps[0][0]], mods4[1], 1.0, steered=True)
check("sequential fit: Mean/LinearAcT also steers the TARGET branch during capture (4c)",
      not torch.allclose(unsteered_pos, steered_pos, atol=1e-6))
# ... but target_steered=False (PID source_only semantics) leaves it raw
unsteered_pos_b = _module_pool_batch(lm, fake_tok, pos_texts4, 4, 32, "mean",
                                     mods4[:1], [model_mean.maps[0][0]], mods4[1], 1.0, steered=False)
check("sequential fit: steered=False leaves the branch's capture untouched (4c)",
      torch.allclose(unsteered_pos_b, unsteered_pos, atol=1e-6))

# ---- (4d) early abort: capturing module 1 never runs modules 2/3
VISITED.clear()
_module_pool_batch(lm, fake_tok, neg_texts4[:4], 4, 32, "mean", [], [], mods4[1], 1.0, steered=True)
check("sequential fit: early abort -- modules after the target never run (4d)",
      set(VISITED) == {0, 1})
VISITED.clear()
_module_pool_batch(lm, fake_tok, neg_texts4[:4], 4, 32, "mean", [], [], mods4[3], 1.0, steered=True)
check("sequential fit: capturing the LAST module runs every module exactly once (4d)",
      VISITED == [0, 1, 2, 3])

# ---- (4e) atonce ablation == sequential when maps are identity (mu1==mu2
# everywhere: achieved by fitting on pos_texts == neg_texts)
same_rows = []
for i in range(8):
    same_rows.append(mk_row(0, f"x{i}", f"body{i}", "conceptSame", "positive"))
    same_rows.append(mk_row(-1, f"x{i}", f"body{i}", "", "negative"))
samples_same = pl.DataFrame(same_rows)
concepts_same = [Concept(id=0, name="conceptSame", description="")]

model_seq = MeanAcTModel(MeanAcTConfig(subject=subject4, module_regexes=REGEX4,
                                       batch_size=4, max_length=32, seed=11, sequential=True)
                        ).bind(lm, fake_tok)
model_seq.fit(concepts_same, samples_same)
model_atonce = MeanAcTModel(MeanAcTConfig(subject=subject4, module_regexes=REGEX4,
                                          batch_size=4, max_length=32, seed=11, sequential=False)
                           ).bind(lm, fake_tok)
model_atonce.fit(concepts_same, samples_same)
check("MeanShiftMap on identical pos/neg texts is the identity map (mu1==mu2) (4e)",
      all(torch.allclose(m.mu1, m.mu2, atol=1e-5) for m in model_seq.maps[0]))
check("atonce == sequential when installed maps are identity (4e)",
      all(torch.allclose(a.mu1, b.mu1, atol=1e-5) and torch.allclose(a.mu2, b.mu2, atol=1e-5)
          for a, b in zip(model_seq.maps[0], model_atonce.maps[0])))

# ---- (4f) PID source_only vs both: differ once prior corrections are non-trivial
model_pid_src = PIDAcTModel(PIDAcTConfig(subject=subject4, module_regexes=REGEX4,
                                         batch_size=4, max_length=32, seed=7,
                                         estimation="source_only")).bind(lm, fake_tok)
model_pid_src.fit(concepts4, samples4)
model_pid_both = PIDAcTModel(PIDAcTConfig(subject=subject4, module_regexes=REGEX4,
                                          batch_size=4, max_length=32, seed=7,
                                          estimation="both")).bind(lm, fake_tok)
model_pid_both.fit(concepts4, samples4)
check("PIDAcTModel: source_only and both agree at module 0 (no prior correction yet) (4f)",
      torch.allclose(model_pid_src.maps[0][0].r, model_pid_both.maps[0][0].r, atol=1e-5))
check("PIDAcTModel: source_only and both DIFFER from module 1 on (nontrivial prior law) (4f)",
      not torch.allclose(model_pid_src.maps[0][1].r, model_pid_both.maps[0][1].r, atol=1e-6))
try:
    PIDAcTModel(PIDAcTConfig(subject=subject4, module_regexes=REGEX4, estimation="bogus")
              ).bind(lm, fake_tok).fit(concepts4, samples4)
    bad_estimation_raised = False
except ValueError:
    bad_estimation_raised = True
check("PIDAcTModel.fit: bad estimation value raises ValueError (4f)", bad_estimation_raised)

# R6: repo_compat=True with estimation != "both" raises -- repo_compat
# reproduces the fork's shared incremental loop, which always steers both
# branches.
try:
    PIDAcTModel(PIDAcTConfig(subject=subject4, module_regexes=REGEX4,
                             repo_compat=True, estimation="source_only")
              ).bind(lm, fake_tok).fit(concepts4, samples4)
    bad_repo_compat_raised = False
except ValueError:
    bad_repo_compat_raised = True
check("PIDAcTModel.fit: repo_compat=True with estimation!='both' raises ValueError (4f)",
      bad_repo_compat_raised)

# ---- (4g) LinearAcTModel + PIDAcTModel end-to-end shapes/determinism
model_lin = LinearAcTModel(LinearAcTConfig(subject=subject4, module_regexes=REGEX4,
                                          batch_size=4, max_length=32, seed=7)).bind(lm, fake_tok)
model_lin.fit(concepts4, samples4)
check("LinearAcTModel.fit: one map per module, each with w/b of shape [D] (4g)",
      len(model_lin.maps[0]) == L4
      and all(tuple(m.w.shape) == (D4,) for m in model_lin.maps[0]))

model_lin_b = LinearAcTModel(LinearAcTConfig(subject=subject4, module_regexes=REGEX4,
                                            batch_size=4, max_length=32, seed=7)).bind(lm, fake_tok)
model_lin_b.fit(concepts4, samples4)
check("LinearAcTModel.fit: deterministic across two fits, same seed (4g)",
      all(torch.equal(a.w, b.w) and torch.equal(a.b, b.b)
          for a, b in zip(model_lin.maps[0], model_lin_b.maps[0])))

model_mean_b = MeanAcTModel(MeanAcTConfig(subject=subject4, module_regexes=REGEX4,
                                         batch_size=4, max_length=32, seed=7)).bind(lm, fake_tok)
model_mean_b.fit(concepts4, samples4)
check("MeanAcTModel.fit: deterministic across two fits, same seed (4g)",
      all(torch.equal(a.mu1, b.mu1) for a, b in zip(model_mean.maps[0], model_mean_b.maps[0])))

check("PIDAcTModel.fit: one map per module, u/r shape [D] (4g)",
      len(model_pid_src.maps[0]) == L4
      and all(tuple(m.u.shape) == (D4,) for m in model_pid_src.maps[0]))
model_pid_src_b = PIDAcTModel(PIDAcTConfig(subject=subject4, module_regexes=REGEX4,
                                           batch_size=4, max_length=32, seed=7,
                                           estimation="source_only")).bind(lm, fake_tok)
model_pid_src_b.fit(concepts4, samples4)
check("PIDAcTModel.fit: deterministic across two fits, same seed (4g)",
      all(torch.equal(a.u, b.u) for a, b in zip(model_pid_src.maps[0], model_pid_src_b.maps[0])))

# ---- (4h) context() steers a real forward; diagnostics; not-bound raises
from steergym.training_helpers import pad_stack  # noqa: E402

ids4_rows = [torch.tensor([BOS] + [fake_tok._t(c) for c in t]) for t in pos_texts4[:3]]
ids4 = pad_stack(ids4_rows, PAD, torch.long, torch)
base4 = lm(input_ids=ids4).hidden_states[0]
with model_mean.context(concepts4[0], lmbda=1.0, positions="all") as ctxm:
    steered4 = lm(input_ids=ids4).hidden_states[0]
check("MeanAcTModel.context: steers a real forward (4h)", not torch.allclose(steered4, base4))
check("MeanAcTModel.context: diagnostics cover every module, keyed by NAME (4h)",
      set(ctxm.diagnostics()["per_module"]) == set(model_mean.module_names))

with model_lin.context(concepts4[0], lmbda=1.0, positions="all"):
    steered4_lin = lm(input_ids=ids4).hidden_states[0]
check("LinearAcTModel.context: steers a real forward (4h)", not torch.allclose(steered4_lin, base4))

with model_pid_src.context(concepts4[0], alpha=1.0, positions="all"):
    steered4_pid = lm(input_ids=ids4).hidden_states[0]
check("PIDAcTModel.context: steers a real forward (4h)", not torch.allclose(steered4_pid, base4))
with model_pid_src.context(concepts4[0], alpha=0.0, positions="all"):
    steered4_pid0 = lm(input_ids=ids4).hidden_states[0]
check("PIDAcTModel.context: alpha=0.0 is a no-op (4h)",
      torch.allclose(steered4_pid0, base4, atol=1e-6))

unbound_mean = MeanAcTModel(MeanAcTConfig(subject=subject4, module_regexes=REGEX4))
try:
    unbound_mean.context(concepts4[0])
    not_bound_raised = False
except RuntimeError:
    not_bound_raised = True
check("MeanAcTModel.context: not bound raises RuntimeError (4h)", not_bound_raised)

# ---- (4i) empty pos/neg raise ValueError
no_match_concept = Concept(id=99, name="no-such-concept", description="")
model_empty = MeanAcTModel(MeanAcTConfig(subject=subject4, module_regexes=REGEX4,
                                        batch_size=4, max_length=32)).bind(lm, fake_tok)
try:
    model_empty.fit([no_match_concept], samples4)
    empty_pos_raised = False
except ValueError:
    empty_pos_raised = True
check("MeanAcTModel.fit: concept with zero positive rows raises ValueError (4i)", empty_pos_raised)

rows4_no_neg = [r for r in rows4 if r["meta"]["category"] != "negative"]
samples4_no_neg = pl.DataFrame(rows4_no_neg)
model_empty_neg = MeanAcTModel(MeanAcTConfig(subject=subject4, module_regexes=REGEX4,
                                             batch_size=4, max_length=32)).bind(lm, fake_tok)
try:
    model_empty_neg.fit(concepts4, samples4_no_neg)
    empty_neg_raised = False
except ValueError:
    empty_neg_raised = True
check("MeanAcTModel.fit: concept with zero negative rows raises ValueError (4i)", empty_neg_raised)

# no modules matched at all
model_no_modules = MeanAcTModel(MeanAcTConfig(subject=subject4, module_regexes=(r"nonexistent\.module",))
                               ).bind(lm, fake_tok)
try:
    model_no_modules.fit(concepts4, samples4)
    no_modules_raised = False
except ValueError:
    no_modules_raised = True
check("MeanAcTModel.fit: module_regexes matching nothing raises ValueError (4i)", no_modules_raised)


# ---- (4j) M7(i): branch-swap pin -- a PID-style sequential fit with
# target_steered=False (what PIDAcTModel.estimation="source_only" passes to
# _sequential_fit) must leave the POS(target) pool RAW at k>=1 while the
# NEG(source) pool is the one actually steered by the so-far-fitted maps;
# this fails if the steered= arguments to pos_pool/neg_pool are ever
# swapped inside _sequential_fit.
from steergym.acttransport.models import _sequential_fit  # noqa: E402

captured_branch_pools = {}


def _record_fit_map(k, name, pos_pool, neg_pool, maps_so_far):
    captured_branch_pools[k] = (pos_pool.clone(), neg_pool.clone())
    return MeanShiftMap().fit(neg_pool, pos_pool, support="q_all", std_eps=1e-4)


_sequential_fit(lm, fake_tok, model_mean.module_names, pos_texts4, neg_texts4,
                model_mean.config, _record_fit_map, target_steered=False)
pos_pool_k1, neg_pool_k1 = captured_branch_pools[1]
raw_pos_k1 = _module_pool_batch(lm, fake_tok, pos_texts4, 4, 32, "mean", [], [], mods4[1], 1.0, steered=True)
raw_neg_k1 = _module_pool_batch(lm, fake_tok, neg_texts4, 4, 32, "mean", [], [], mods4[1], 1.0, steered=True)
check("branch-swap pin: target_steered=False leaves the POS pool RAW at k=1 (4j)",
      torch.allclose(pos_pool_k1, raw_pos_k1, atol=1e-6))
check("branch-swap pin: the NEG pool IS steered (differs from raw) at k=1 (4j)",
      not torch.allclose(neg_pool_k1, raw_neg_k1, atol=1e-6))

# ---- (4k) M7(ii): direction pin for Mean -- the steered SOURCE(neg) pool
# must move toward the INDEPENDENTLY (ground-truth) computed target mean
# mu2, not toward the source's own mean; this fails under a fit(pos, neg)
# direction flip inside MeanAcTModel._fit_map (comparing against the map's
# OWN mu1/mu2 buffers would not catch a flip -- they'd just be
# mislabeled consistently). model_mean.fit() balanced (shuffled+truncated,
# M6) its own copies of pos_texts4/neg_texts4 with config.seed before
# fitting -- reproduce that exact input here so the "ground truth" is
# computed over the SAME rows the map was actually fit on.
import random as _random4k  # noqa: E402

_rng4k = _random4k.Random(model_mean.config.seed)
pos_texts4_bal, neg_texts4_bal = list(pos_texts4), list(neg_texts4)
_rng4k.shuffle(pos_texts4_bal)
_rng4k.shuffle(neg_texts4_bal)
_n4k = min(len(pos_texts4_bal), len(neg_texts4_bal))
pos_texts4_bal, neg_texts4_bal = pos_texts4_bal[:_n4k], neg_texts4_bal[:_n4k]

raw_neg_pool0 = _module_pool_batch(lm, fake_tok, neg_texts4_bal, 4, 32, "mean", [], [], mods4[0], 1.0, steered=True)
raw_pos_pool0 = _module_pool_batch(lm, fake_tok, pos_texts4_bal, 4, 32, "mean", [], [], mods4[0], 1.0, steered=True)
mu1_gt = raw_neg_pool0.double().mean(dim=0)
mu2_gt = raw_pos_pool0.double().mean(dim=0)
ctx_direction = model_mean.context(concepts4[0], lmbda=1.0, positions="all")
steered_neg_pool0 = ctx_direction._apply(0, raw_neg_pool0.float())
mask0 = model_mean.maps[0][0].mask
expected_neg_pool0 = (raw_neg_pool0.double() - mu1_gt + mu2_gt).float()
check("direction pin (Mean): steered SOURCE pool == raw - GT mu1 + GT mu2 on masked units (4k)",
      torch.allclose(steered_neg_pool0[:, mask0], expected_neg_pool0[:, mask0], atol=1e-4))
check("direction pin (Mean): masked units actually moved (sanity, not a no-op) (4k)",
      not torch.allclose(steered_neg_pool0[:, mask0], raw_neg_pool0[:, mask0], atol=1e-6))

# ---- (4l) M7(iii): strict-support boundary pin THROUGH THE CONTEXT (not by
# recomputing the mask expression by hand, unlike (1c)/(2)) -- fails under
# a strict-< -> <= typo in MultiModuleTransformContext._apply.
class _PassThrough(nn.Module):
    def forward(self, h):
        return h


boundary_map = MeanShiftMap()
boundary_map.register_buffer("mu1", torch.zeros(3))
boundary_map.register_buffer("mu2", torch.ones(3) * 10.0)
boundary_map.register_buffer("mask", torch.tensor([True, True, True]))
boundary_map.register_buffer("q_lo", torch.tensor([-1.0, -1.0, -1.0]))
boundary_map.register_buffer("q_hi", torch.tensor([1.0, 1.0, 1.0]))
boundary_module = _PassThrough()
# unit0 sits exactly AT q_hi, unit1 exactly AT q_lo, unit2 strictly inside
rows_boundary = torch.tensor([[1.0, -1.0, 0.999]]).unsqueeze(1)  # [B=1, T=1, d=3]
ctx_boundary = MultiModuleTransformContext([boundary_module], [boundary_map], lmbda=1.0, positions="all")
with ctx_boundary:
    out_boundary = boundary_module(rows_boundary)
out_boundary = out_boundary.squeeze(1)
check("boundary pin THROUGH THE CONTEXT: unit exactly at q_hi passes through unchanged (4l)",
      torch.allclose(out_boundary[0, 0], torch.tensor(1.0)))
check("boundary pin THROUGH THE CONTEXT: unit exactly at q_lo passes through unchanged (4l)",
      torch.allclose(out_boundary[0, 1], torch.tensor(-1.0)))
check("boundary pin THROUGH THE CONTEXT: unit strictly inside IS transported to z-mu1+mu2 (4l)",
      torch.allclose(out_boundary[0, 2], torch.tensor(0.999 - 0.0 + 10.0), atol=1e-5))

# ---- (4m) M7(iv): pooling batch invariance -- ragged-length texts must give
# an IDENTICAL [N, d] pooled output at batch_size=1 vs batch_size=4 (fails
# if the attention mask is ignored in mean pooling, letting padding leak
# into the per-row mean).
pool_bs1 = _module_pool_batch(lm, fake_tok, pos_texts4, 1, 32, "mean", [], [], mods4[0], 1.0, steered=True)
pool_bs4 = _module_pool_batch(lm, fake_tok, pos_texts4, 4, 32, "mean", [], [], mods4[0], 1.0, steered=True)
check("pooling batch invariance: batch_size=1 vs 4 give identical pooled output (4m)",
      torch.allclose(pool_bs1, pool_bs4, atol=1e-5))

# ---- (4n) R1: hook-leak guard -- an exception raised between the capture
# hook's registration and ctx.__enter__ (here, a length-mismatched
# prior_modules/prior_maps pair raising inside MultiModuleTransformContext's
# constructor) must not leak the capture hook on target_module.
before_hooks_4n = len(mods4[1]._forward_hooks)
try:
    _module_pool_batch(lm, fake_tok, neg_texts4[:4], 4, 32, "mean",
                       mods4[:1], [model_mean.maps[0][0], model_mean.maps[0][0]],
                       mods4[1], 1.0, steered=True)
    r1_mismatch_raised = False
except ValueError:
    r1_mismatch_raised = True
check("R1: mismatched prior_modules/prior_maps raises ValueError (4n)", r1_mismatch_raised)
check("R1: no leaked capture hook on target_module after that failure (4n)",
      len(mods4[1]._forward_hooks) == before_hooks_4n)


# ================================================================== 5. persistence
import tempfile  # noqa: E402

SP = os.environ.get("ACTTRANSPORT_SCRATCH") or tempfile.mkdtemp(prefix="test_acttransport_")

for name, model_obj, cls in (
    ("mean", model_mean, MeanAcTModel),
    ("linear", model_lin, LinearAcTModel),
    ("pid", model_pid_src, PIDAcTModel),
):
    ckpt = f"{SP}/{name}.pt"
    model_obj.save(ckpt)
    loaded = cls.load(ckpt)
    check(f"{name}: save/load roundtrip preserves module_names (5)",
          loaded.module_names == model_obj.module_names)
    keys_match = set(loaded.maps) == set(model_obj.maps)
    buffers_match = all(
        torch.equal(getattr(lm_, k), getattr(mm_, k))
        for lm_, mm_ in zip(loaded.maps[0], model_obj.maps[0])
        for k in cls.fitted_buffers)
    check(f"{name}: save/load roundtrip preserves concept ids + every fitted buffer (5)",
          keys_match and buffers_match)
    check(f"{name}: save/load roundtrip preserves config (5)",
          loaded.config.seed == model_obj.config.seed
          and loaded.config.module_regexes == model_obj.config.module_regexes)

# wrong-cls artifact raises
bad_cls_path = f"{SP}/bad_cls.pt"
torch.save({"cls": "NotAModel", "config": {}, "module_names": [], "maps": {}}, bad_cls_path)
try:
    MeanAcTModel.load(bad_cls_path)
    wrong_cls_raised = False
except ValueError:
    wrong_cls_raised = True
check("save/load: wrong-cls artifact raises ValueError (5)", wrong_cls_raised)

# wrong-buffer-keys artifact raises (missing "mu2")
bad_keys_path = f"{SP}/bad_keys.pt"
cfg_d = {**{"module_regexes": REGEX4, "module_preset": "postln", "pooling": "mean",
           "positions": "all", "sequential": True, "fit_strength": 1.0, "support": "q_all",
           "std_eps": 1e-4, "batch_size": 4, "max_length": 32, "seed": 1,
           "layer": None, "site": None},
         "subject": {"model_id": "x", "revision": None}}
torch.save({
    "cls": "MeanAcTModel", "config": cfg_d, "module_names": ["a"],
    "maps": {0: [{"mu1": torch.zeros(3), "mask": torch.ones(3, dtype=torch.bool),
                 "q_lo": torch.zeros(3), "q_hi": torch.ones(3)}]},  # missing "mu2"
}, bad_keys_path)
try:
    MeanAcTModel.load(bad_keys_path)
    bad_keys_raised = False
except ValueError:
    bad_keys_raised = True
check("save/load: missing-buffer-key artifact raises ValueError (5)", bad_keys_raised)

# L1: per-MODULE-INDEX validation -- DIFFERENT module indices in the SAME
# artifact/concept MAY have different trailing dims (mixed-width module
# sets, e.g. llama icml24's gate/up_proj @ 14336 vs down_proj @ 4096): module
# 0 at d=3, module 1 at d=4, same concept -- this must NOT raise any more.
mixed_width_dims_path = f"{SP}/mixed_width_dims.pt"
torch.save({
    "cls": "MeanAcTModel", "config": cfg_d, "module_names": ["a", "b"],
    "maps": {0: [
        {"mu1": torch.zeros(3), "mu2": torch.ones(3), "mask": torch.ones(3, dtype=torch.bool),
         "q_lo": torch.zeros(3), "q_hi": torch.ones(3)},
        {"mu1": torch.zeros(4), "mu2": torch.ones(4), "mask": torch.ones(4, dtype=torch.bool),
         "q_lo": torch.zeros(4), "q_hi": torch.ones(4)},
    ]},
}, mixed_width_dims_path)
try:
    MeanAcTModel.load(mixed_width_dims_path)
    mixed_width_dims_ok = True
except ValueError as e:
    mixed_width_dims_ok = False
    print(f"  (unexpected) mixed-width-across-modules load failed: {e}")
check("L1: different module INDICES with different trailing dims, same concept, "
      "no longer raises (5)", mixed_width_dims_ok)

# L1 (still an error): the SAME module index disagreeing on its trailing dim
# ACROSS CONCEPTS still means a corrupt/mixed-up save -- module index 0 is
# d=3 under concept 0 but d=5 under concept 1.
bad_dims_path = f"{SP}/bad_dims.pt"
torch.save({
    "cls": "MeanAcTModel", "config": cfg_d, "module_names": ["a"],
    "maps": {
        0: [{"mu1": torch.zeros(3), "mu2": torch.ones(3), "mask": torch.ones(3, dtype=torch.bool),
             "q_lo": torch.zeros(3), "q_hi": torch.ones(3)}],
        1: [{"mu1": torch.zeros(5), "mu2": torch.ones(5), "mask": torch.ones(5, dtype=torch.bool),
             "q_lo": torch.zeros(5), "q_hi": torch.ones(5)}],
    },
}, bad_dims_path)
try:
    MeanAcTModel.load(bad_dims_path)
    bad_dims_raised = False
except ValueError:
    bad_dims_raised = True
check("L1: the SAME module index disagreeing on its trailing dim ACROSS CONCEPTS "
      "still raises ValueError (5)", bad_dims_raised)

# unaffected by L1: a mismatch WITHIN one map's own buffers (mu1 d=3 vs mu2
# d=4) still raises -- this is a per-map, not a per-module-index, check.
bad_intra_dims_path = f"{SP}/bad_intra_dims.pt"
torch.save({
    "cls": "MeanAcTModel", "config": cfg_d, "module_names": ["a"],
    "maps": {0: [
        {"mu1": torch.zeros(3), "mu2": torch.ones(4), "mask": torch.ones(3, dtype=torch.bool),
         "q_lo": torch.zeros(3), "q_hi": torch.ones(3)},
    ]},
}, bad_intra_dims_path)
try:
    MeanAcTModel.load(bad_intra_dims_path)
    bad_intra_dims_raised = False
except ValueError:
    bad_intra_dims_raised = True
check("unaffected by L1: mismatched buffer trailing dims WITHIN one map still "
      "raises (5)", bad_intra_dims_raised)

# wrong-module-names (regex-set mismatch) artifact rejected at context()-time:
# bind a model whose find_module_names() output disagrees with the artifact's
# recorded module_names (a different-sized layer stack under the same regex)
g5 = torch.Generator().manual_seed(44)
lm_small = FakeLM(D4, V4, 2, g5)  # only 2 layers, not 4
for p in lm_small.parameters():
    p.requires_grad_(False)
mismatched = MeanAcTModel.load(f"{SP}/mean.pt").bind(lm_small, fake_tok)
try:
    mismatched.context(concepts4[0])
    mismatch_raised = False
except ValueError:
    mismatch_raised = True
check("context(): artifact module_names != bound model's find_module_names raises ValueError (5)",
      mismatch_raised)


# ---- L1/L4(a): mixed-width artifact save/load round-trip, adapted from
# scratchpad/acttqa_probe/probe_mixed_load.py -- a REAL tiny Llama whose
# "icml24" module set spans TWO distinct widths (mlp up_proj/gate_proj are
# [intermediate_size]-wide, down_proj is [hidden_size]-wide) must fit AND
# round-trip through save()/load() now that load() validates per MODULE
# INDEX rather than once across the whole artifact. This is the exact
# mixed-width shape (llama icml24) the spec calls out, fit through the real
# sequential-fit path end to end (not a hand-built state dict).
if _real_transformers is None:
    print("SKIP mixed-width real-HF save/load round-trip: transformers unavailable")
else:
    mw_cfg = _RealLlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=40,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, pad_token_id=0,
                              attn_implementation="eager")
    mw_model = _RealLlamaForCausalLM(mw_cfg).eval()
    for p in mw_model.parameters():
        p.requires_grad_(False)

    class _MixedWidthTok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, t, max_length=None, truncation=False):
            ids = [2] + [3 + ord(c) % 60 for c in t]
            if truncation and max_length:
                ids = ids[:max_length]
            return {"input_ids": ids}

    mw_tok = _MixedWidthTok()
    mw_pos = [f"yes {i} true" for i in range(6)]
    mw_neg = [f"no {i} false!" for i in range(6)]
    mixed_width_rt_ok = {}
    for mw_cls, mw_ccls in ((MeanAcTModel, MeanAcTConfig), (LinearAcTModel, LinearAcTConfig)):
        mw = mw_cls(mw_ccls(subject=SUBJECT, module_preset="icml24")).bind(mw_model, mw_tok)
        mw_names = find_module_names(mw_model, resolve_module_regexes(mw.config))
        check(f"mixed-width ({mw_cls.__name__}): icml24 resolves modules of >1 distinct "
              f"width on this tiny Llama (hidden=16, intermediate=40) (5, L1)",
              len({dict(mw_model.named_modules())[n].weight.shape[0] for n in mw_names}) > 1)
        mw.module_names = mw_names

        def mw_fit_map(k, n, p, q, maps_so_far, _m=mw):
            return _m._fit_map(k, n, p, q, maps_so_far)

        _, mw_maps = _sequential_fit(mw_model, mw_tok, mw_names, mw_pos, mw_neg,
                                     mw.config, mw_fit_map, True)
        mw.maps["k-s0"] = mw_maps
        mw_ckpt = f"{SP}/mixed_width_{mw_cls.__name__}.pt"
        mw.save(mw_ckpt)
        try:
            mw_loaded = mw_cls.load(mw_ckpt)
            mixed_width_rt_ok[mw_cls.__name__] = (
                mw_loaded.module_names == mw.module_names
                and set(mw_loaded.maps) == set(mw.maps))
        except ValueError as e:
            mixed_width_rt_ok[mw_cls.__name__] = False
            print(f"  (unexpected) {mw_cls.__name__} mixed-width load failed: {e}")
    check("mixed-width real-HF save/load round-trip passes for MeanAcT + LinearAcT "
          "(5, L1/L4(a))",
          set(mixed_width_rt_ok) == {"MeanAcTModel", "LinearAcTModel"}
          and all(mixed_width_rt_ok.values()))


# ================================================================== 6. ACT_DIR-gated upstream source parity
ACT_DIR = os.environ.get("ACT_DIR")
if not ACT_DIR:
    print("SKIP upstream source-parity leg (1 group skipped): ACT_DIR not set")
else:
    sys.path.insert(0, ACT_DIR)
    try:
        del sys.modules["transformers"]  # drop our fake-LM stub for the real import below
    except KeyError:
        pass
    try:
        from act.hooks.transport import OnlyMeanHook
        from act.optimal_transport.archs import LinearProj
        from act.utils.quantiles import compute_quantiles
    except Exception as exc:  # omegaconf / transformers version skew etc.
        print(f"SKIP upstream source-parity leg 6a-6c: import failed ({exc!r})")
        OnlyMeanHook = None
    try:
        from act.models.model_with_hooks import ModelWithHooks
    except Exception as exc:  # the heavier dep chain (diffusers/wandb/...)
        print(f"SKIP upstream source-parity leg 6d: import failed ({exc!r})")
        ModelWithHooks = None

    if OnlyMeanHook is not None:
        gp = torch.Generator().manual_seed(55)
        src_p = torch.randn(80, 6, generator=gp) * 1.5 + 0.5
        tgt_p = torch.randn(80, 6, generator=gp) * 2.5 + 3.0

        # 6a. LinearProj.optimize vs ours, on IDENTICAL (sorted, pre-noise)
        # tensors -- both draw their own 1e-8 noise from numpy's global RNG
        # vs our seeded torch.Generator, so this compares the FORMULA (set
        # numpy's seed right before calling theirs, torch's right before
        # calling ours, same seed value) rather than bit-identical noise.
        import numpy as np

        x_np = src_p.sort(dim=0).values.numpy()
        y_np = tgt_p.sort(dim=0).values.numpy()
        np.random.seed(123)
        proj = LinearProj(dim=6)
        params_theirs, _ = proj.optimize(x_np.copy(), y_np.copy())
        w_theirs = torch.tensor(params_theirs[0], dtype=torch.float32)   # beta, per-unit
        b_theirs = torch.tensor(params_theirs[1], dtype=torch.float32)  # alpha, per-unit

        ours = IndependentSortOTMap()
        torch.manual_seed(123)  # best-effort alignment; formula-level check, not bitwise
        ours.fit(src_p, tgt_p, support="q_all", seed=123)
        check("upstream parity: LinearProj.optimize slope/intercept close to ours (6a)",
              torch.allclose(ours.w, w_theirs, atol=1e-6)
              and torch.allclose(ours.b, b_theirs, atol=5e-2))

        # 6b. OnlyMeanHook stats (mu1/mu2/std1/std2/mask) vs ours
        hook = OnlyMeanHook(module_name="dummy", strength=1.0, quantiles_src="q_all")
        labels = torch.ones(80 + 80, dtype=torch.bool)
        labels[80:] = False
        z_cat = torch.cat([src_p, tgt_p], dim=0)
        hook.fit(responses=z_cat, labels=labels)
        ours_mean = MeanShiftMap().fit(src_p, tgt_p, support="q_all", std_eps=1e-4)
        check("upstream parity: OnlyMeanHook mu1/mu2 match MeanShiftMap (6b)",
              torch.allclose(hook.mu1, ours_mean.mu1, atol=1e-4)
              and torch.allclose(hook.mu2, ours_mean.mu2, atol=1e-4))
        check("upstream parity: OnlyMeanHook mask matches our std_mask (6b)",
              torch.equal(hook.mask, ours_mean.mask))

        # 6c. compute_quantiles vs our support_bounds, for every shared key
        qdict = compute_quantiles(src_p.double())
        for key in ("q_0_100", "q_5_95", "q_30_70"):
            lo, hi = support_bounds(src_p, key)
            check(f"upstream parity: support_bounds[{key}] matches compute_quantiles (6c)",
                  torch.allclose(lo, qdict[key][0].float(), atol=1e-4)
                  and torch.allclose(hi, qdict[key][1].float(), atol=1e-4))

    # 6d. find_module_names vs ModelWithHooks.find_module_names, on a tiny
    # real HF config -- independent of the 6a-6c import group above (its
    # own, heavier dep chain)
    if ModelWithHooks is not None:
        try:
            from transformers import LlamaConfig, LlamaForCausalLM

            cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=2, num_attention_heads=2,
                              num_key_value_heads=2, max_position_embeddings=32)
            hf_model = LlamaForCausalLM(cfg).eval()
            regexes = [r".*self_attn\.o_proj"]
            theirs = ModelWithHooks.find_module_names(hf_model, regexes)
            ours_names = find_module_names(hf_model, regexes)
            check("upstream parity: find_module_names matches ModelWithHooks.find_module_names (6d)",
                  ours_names == theirs)
        except Exception as exc:
            print(f"SKIP find_module_names parity (6d): {exc!r}")


# ============================================= 7. PID uniform-width module check
from steergym.acttransport.models import PIDAcTModel, MeanAcTModel, module_output_width  # noqa: E402

check("module_output_width: nn.Linear -> out_features (7)",
      module_output_width(nn.Linear(8, 24)) == 24)
_norm = nn.LayerNorm(16)
check("module_output_width: LayerNorm -> normalized_shape (7)",
      module_output_width(_norm) == 16)


class _WeightOnlyNorm(nn.Module):  # RMSNorm-shaped: lone 1-D weight
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))


check("module_output_width: lone 1-D weight (RMSNorm shape) -> its dim (7)",
      module_output_width(_WeightOnlyNorm(12)) == 12)
check("module_output_width: unknown module -> None (7)",
      module_output_width(nn.ReLU()) is None)

_mixed_names = ["a.gate", "a.down", "b.gate"]
_mixed_mods = [nn.Linear(4, 10), nn.Linear(10, 4), nn.Linear(4, 10)]
_uniform_mods = [nn.Linear(4, 10), nn.Linear(4, 10), nn.Linear(4, 10)]


def _pid(repo_compat=False):
    cfg = PIDAcTModel.config_cls(subject=SUBJECT, repo_compat=repo_compat,
                                 estimation="both" if repo_compat else "source_only")
    return PIDAcTModel(cfg)


try:
    _pid()._check_modules(_mixed_names, _mixed_mods)
    mixed_raised = False
except ValueError as e:
    mixed_raised = "uniform-width" in str(e) and "d=4" in str(e) and "d=10" in str(e)
check("PIDAcT canonical: mixed-width module set raises naming both widths (7)", mixed_raised)

try:
    _pid()._check_modules(_mixed_names, _uniform_mods)
    _pid(repo_compat=True)._check_modules(_mixed_names, _mixed_mods)
    MeanAcTModel(MeanAcTModel.config_cls(subject=SUBJECT))._check_modules(
        _mixed_names, _mixed_mods)
    ok_paths = True
except ValueError:
    ok_paths = False
check("uniform set, repo_compat mixed, and MeanAcT mixed all pass _check_modules (7)", ok_paths)


if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("\nall checks passed")
