"""AcT-family SteerModels: MeanAcTModel, LinearAcTModel, PIDAcTModel.

Fit is a SEQUENTIAL (causal) loop over the resolved module set, port of
ml-act scripts/learn_intervention.py's `learn_intervention_all_incremental`
(:337-392): for module k in registration order, install the already-fitted
maps for modules < k (lmbda=config.fit_strength, positions="all", BOTH
source and target sentences pushed through -- paper Def 3.2), capture each
sentence's ATTENTION-MASKED pooled activation at module k (pooling="mean"
default, paper App. D; early forward abort right after capture -- ml-act's
TargetModuleReached trick, context.py's `ModuleReached`), fit module k's
map, continue. `config.sequential=False` is their `atonce` ablation
(:420-437's `incremental == "atonce"` branch): one pass per branch, every
module's own RAW activations captured simultaneously, no incremental
coupling at all.

PID-AcT (`PIDAcTModel`) reuses the exact same sequential loop (the
released fork literally patches ml-act's incremental script) but fits a
`PIDShiftMap` per module from a running r(k) = E_dst[x(k)] - E_src[x~(k)]
history instead of an OT map; `config.estimation` controls whether the
TARGET (positive) branch is also pushed through the so-far-fitted PID
corrections ("both", what the fork's shared loop actually does) or left
raw ("source_only", the paper's Eqs. 4-5/19 -- canonical default). The
SOURCE branch is always pushed through (the whole point of the sequential
estimate is "how much of the gap is left after what's already been
corrected").
"""
from dataclasses import dataclass

from ..concept import Concept
from ..steer import SteerConfig, SteerModel
from .context import ModuleReached, MultiModuleTransformContext, find_module_names, resolve_module_regexes

# .maps (module-level torch, nn.Module subclasses) is imported lazily at
# point of use so the steergym.models aggregator stays torch-lazy


# ---------------------------------------------------------------- configs
@dataclass
class AcTTransportConfig(SteerConfig):
    module_regexes: tuple[str, ...] = ()
    module_preset: str = "postln"       # "postln" | "all_ln" | "icml24" (context.py MODULE_PRESETS)
    pooling: str = "mean"                # "mean" (paper App. D: best) | "last"
    positions: str = "all"               # inference-time default; context() can override
    sequential: bool = True              # False = the "atonce" ablation (no incremental coupling)
    fit_strength: float = 1.0            # lmbda for installing maps<k DURING the sequential fit
    support: str = "q_all"               # overridden per method below
    std_eps: float = 1e-4
    batch_size: int = 6
    max_length: int = 1024
    seed: int = 42


@dataclass
class MeanAcTConfig(AcTTransportConfig):
    # shipped mean_ot.yaml default. NOTE (GAP, recorded in the README): the
    # paper's own mitigation rule calls for Q_o (a real quantile support),
    # not "q_all" -- yaml-vs-paper divergence, knob exposed either way.
    support: str = "q_all"


@dataclass
class LinearAcTConfig(AcTTransportConfig):
    support: str = "q_0_100"  # shipped linear_ot.yaml, matches the paper's mitigation rule
    # std_eps is inherited but INERT here: upstream's LinearOTHook/LearnableOTHook
    # (actup transport.py:461-503) gates only on the quantile support, never on
    # std -- the std mask is a Mean/GaussianOTHook-only rule (README).


@dataclass
class PIDAcTConfig(AcTTransportConfig):
    # No Table-1 values published for kp/ki/kd -- GAP (README): defaulted
    # to the paper's Fig. 6 sweet spot. kp/ki/kd are UNUSED when repo_compat=True
    # (the fork's released law is its own fixed 0.7035*r(k)+0.0035*s_k formula,
    # not a Kp/Ki/Kd law at all -- README).
    kp: float = 1.0
    ki: float = 0.056
    kd: float = 0.01
    estimation: str = "source_only"  # "source_only" (canonical) | "both" (the fork's loop)
    repo_compat: bool = False        # released-code degenerate law (README)
    inclusive_sum: bool = False      # True: integral sums j<=k (inclusive of the current layer)
    # support is inherited but INERT for PID in BOTH modes: the law is never
    # quantile-gated. std_eps is inert in CANONICAL mode (mask wide-open, the
    # paper's Def. 1 has no std gate) but IS used when repo_compat=True (the
    # fork's released law only ever fires on std-masked units, same rule as
    # Mean -- README).


# ---------------------------------------------------------------- shared fit-input helper
def _contrast_texts(samples, concept: Concept, tokenizer):
    # same pos/neg text derivation as odesteer/alqr's (binarize_text chat
    # formatting) -- duplicated submodule-local per the spec (no shared
    # helper falls out naturally here; this package's fit input is the
    # PER-MODULE pooled view, not a single-layer activation tensor)
    from ..axbench.protocol import binarize_text, positive_and_negative_rows, prefix_suffix_lengths

    _, suffix_length = prefix_suffix_lengths(tokenizer)
    positive, negative = positive_and_negative_rows(samples, concept)
    pos_texts = [binarize_text(tokenizer, r["prompt"], r["response"], suffix_length)
                 for r in positive.iter_rows(named=True)]
    neg_texts = [binarize_text(tokenizer, r["prompt"], r["response"], suffix_length)
                 for r in negative.iter_rows(named=True)]
    return pos_texts, neg_texts


def _pool(h, lengths, pooling: str):
    # h: [B, T, d] float32 CPU (right-padded); lengths: [B] long
    import torch

    if pooling == "mean":
        T = h.shape[1]
        mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1).to(h.dtype)
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if pooling == "last":
        idx = torch.arange(len(lengths))
        return h[idx, lengths - 1]
    raise ValueError(f"pooling {pooling!r}; expected 'mean' or 'last'")


def _encode_batches(tokenizer, texts, batch_size, max_length):
    import torch

    from ..training_helpers import pad_stack

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = [tokenizer(t, max_length=max_length, truncation=True)["input_ids"] for t in texts]
    for i in range(0, len(encoded), batch_size):
        batch_ids = [torch.tensor(e, dtype=torch.long) for e in encoded[i:i + batch_size]]
        if not batch_ids:
            continue
        input_ids = pad_stack(batch_ids, pad_id, torch.long, torch)
        lengths = torch.tensor([len(x) for x in batch_ids], dtype=torch.long)
        T = input_ids.shape[1]
        attention_mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)).long()
        yield input_ids, attention_mask, lengths


def _module_pool_batch(model, tokenizer, texts, batch_size, max_length, pooling,
                       prior_modules, prior_maps, target_module, fit_strength, steered: bool):
    """[N, d] float32 CPU: module `target_module`'s pooled activation per
    text, with the (already-fitted) `prior_maps` on `prior_modules`
    installed (lmbda=fit_strength, positions="all") iff `steered` and there
    are any -- the sequential fit's per-module, per-branch capture. Early
    abort: a hook on `target_module` raises `ModuleReached` right after
    capturing, so no compute runs past it."""
    import torch

    out = []
    for input_ids, attention_mask, lengths in _encode_batches(tokenizer, texts, batch_size, max_length):
        captured = {}

        def capture_hook(mod, args, output, captured=captured):
            h = output[0] if isinstance(output, tuple) else output
            captured["h"] = h.detach().float().cpu()
            raise ModuleReached()

        handle = None
        ctx = None
        try:
            handle = target_module.register_forward_hook(capture_hook)
            if steered and prior_modules:
                ctx = MultiModuleTransformContext(prior_modules, prior_maps,
                                                  lmbda=fit_strength, positions="all")
                ctx.__enter__()
            with torch.no_grad():
                try:
                    model(input_ids=input_ids.to(model.device),
                         attention_mask=attention_mask.to(model.device))
                except ModuleReached:
                    pass
        finally:
            if handle is not None:
                handle.remove()
            if ctx is not None:
                ctx.__exit__(None, None, None)
        out.append(_pool(captured["h"], lengths, pooling))
    return torch.cat(out, dim=0) if out else torch.zeros(0, 0)


def _all_modules_pool_batch(model, tokenizer, texts, batch_size, max_length, pooling, modules):
    """[len(modules)] list of [N, d] float32 CPU: every module's pooled
    activation per text in ONE pass (no incremental coupling at all) --
    the `sequential=False` "atonce" ablation."""
    import torch

    per_module = [[] for _ in modules]
    for input_ids, attention_mask, lengths in _encode_batches(tokenizer, texts, batch_size, max_length):
        captured = {}

        def capture_hook(mod, args, output, k):
            o = output[0] if isinstance(output, tuple) else output
            if k not in captured:  # first write wins (setdefault-equivalent)
                captured[k] = o.detach().float().cpu()

        handles = [mod.register_forward_hook(
            lambda m, a, o, k=k: capture_hook(m, a, o, k))
            for k, mod in enumerate(modules)]
        try:
            with torch.no_grad():
                model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device))
        finally:
            for h in handles:
                h.remove()
        for k in range(len(modules)):
            per_module[k].append(_pool(captured[k], lengths, pooling))
    return [torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 0) for chunks in per_module]


def _sequential_fit(model, tokenizer, module_names, pos_texts, neg_texts, config, fit_map, target_steered):
    by_name = dict(model.named_modules())
    modules = [by_name[n] for n in module_names]
    maps = []
    for k in range(len(module_names)):
        pos_pool = _module_pool_batch(model, tokenizer, pos_texts, config.batch_size, config.max_length,
                                      config.pooling, modules[:k], maps, modules[k],
                                      config.fit_strength, steered=target_steered)
        neg_pool = _module_pool_batch(model, tokenizer, neg_texts, config.batch_size, config.max_length,
                                      config.pooling, modules[:k], maps, modules[k],
                                      config.fit_strength, steered=True)
        maps.append(fit_map(k, module_names[k], pos_pool, neg_pool, maps))
    return modules, maps


def _atonce_fit(model, tokenizer, module_names, pos_texts, neg_texts, config, fit_map):
    by_name = dict(model.named_modules())
    modules = [by_name[n] for n in module_names]
    pos_pools = _all_modules_pool_batch(model, tokenizer, pos_texts, config.batch_size,
                                        config.max_length, config.pooling, modules)
    neg_pools = _all_modules_pool_batch(model, tokenizer, neg_texts, config.batch_size,
                                        config.max_length, config.pooling, modules)
    maps = [fit_map(k, module_names[k], pos_pools[k], neg_pools[k], [])
            for k in range(len(module_names))]
    return modules, maps


# ---------------------------------------------------------------- models
class AcTTransportModel(SteerModel):
    """Shared chassis: module-set resolution, the sequential/atonce fit
    dispatch, and persistence (per-concept per-module fitted-map buffers).
    Subclasses implement `_new_map` (an unfitted map instance, for
    load()'s buffer registration), `_fit_map` (this method's per-module
    objective), `fitted_buffers` (the exact buffer key set a fitted map
    persists), and may override `_target_steered`."""

    config_cls = AcTTransportConfig
    weight_name: str | None = None
    fitted_buffers: frozenset[str] = frozenset()

    def __init__(self, config):
        super().__init__(config)
        self.module_names: list[str] = []
        self.maps: dict = {}  # concept.id -> list[map], aligned to module_names

    def _new_map(self):
        raise NotImplementedError

    def _fit_map(self, k, name, pos_pool, neg_pool, maps_so_far):
        raise NotImplementedError

    def _target_steered(self) -> bool:
        return True

    def _check_modules(self, names, modules):
        # subclass hook: validate the resolved module set BEFORE any capture
        # pass spends compute (PIDAcTModel requires uniform width)
        pass

    def fit(self, concepts, samples, log=lambda msg: None):
        import time

        from transformers import set_seed

        model, tokenizer = self._require_bound()
        set_seed(self.config.seed)
        target_steered = self._target_steered()  # validates config knobs before any compute
        regexes = resolve_module_regexes(self.config)
        names = find_module_names(model, regexes)
        if not names:
            raise ValueError(f"module_regexes/preset {regexes!r} matched no modules in the bound model")
        by_name = dict(model.named_modules())
        self._check_modules(names, [by_name[n] for n in names])
        self.module_names = names
        self.maps = {}

        for concept in concepts:
            t0 = time.time()
            pos_texts, neg_texts = _contrast_texts(samples, concept, tokenizer)
            if not pos_texts or not neg_texts:
                raise ValueError(f"concept {concept.id}: {len(pos_texts)} positive / "
                                 f"{len(neg_texts)} negative rows -- {type(self).__name__} needs both")

            # upstream's balanced subsample (learn_intervention.py:285-291 ->
            # responses_io.py label_src_dst_subsets, balanced=True): a seeded
            # permutation of BOTH groups, then truncate both to min-n -- shipped
            # for every hook type, not just Linear-AcT (README). RNG differs
            # from upstream's numpy rng.permutation (disclosed delta).
            import random

            rng = random.Random(self.config.seed)
            rng.shuffle(pos_texts)
            rng.shuffle(neg_texts)
            n_balanced = min(len(pos_texts), len(neg_texts))
            pos_texts, neg_texts = pos_texts[:n_balanced], neg_texts[:n_balanced]

            def fit_map(k, name, pos_pool, neg_pool, maps_so_far):
                return self._fit_map(k, name, pos_pool, neg_pool, maps_so_far)

            if self.config.sequential:
                _, maps = _sequential_fit(model, tokenizer, names, pos_texts, neg_texts,
                                          self.config, fit_map, target_steered)
            else:
                _, maps = _atonce_fit(model, tokenizer, names, pos_texts, neg_texts,
                                      self.config, fit_map)
            self.maps[concept.id] = maps
            log(f"{type(self).weight_name} concept {concept.id} ({concept.name!r}): "
                f"{len(pos_texts)}+{len(neg_texts)} rows, {len(names)} modules, "
                f"{time.time() - t0:.1f}s")
        return self

    def _resolved_modules(self):
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__} not bound — call bind(model) first")
        regexes = resolve_module_regexes(self.config)
        live = find_module_names(self._model, regexes)
        if live != self.module_names:
            raise ValueError(
                f"artifact module_names {self.module_names} != the bound model's "
                f"find_module_names({regexes!r}) = {live}")
        by_name = dict(self._model.named_modules())
        return [by_name[n] for n in self.module_names]

    def context(self, concept: Concept, lmbda: float = 1.0,
               positions: str | None = None) -> MultiModuleTransformContext:
        modules = self._resolved_modules()
        return MultiModuleTransformContext(modules, self.maps[concept.id], lmbda=lmbda,
                                           positions=positions or self.config.positions,
                                           names=self.module_names)

    def save(self, path):
        from dataclasses import asdict

        import torch

        torch.save({
            "cls": type(self).__name__,
            "config": {**asdict(self.config), "subject": asdict(self.config.subject)},
            "module_names": self.module_names,
            "maps": {cid: [m.state_dict() for m in maps] for cid, maps in self.maps.items()},
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
        m.module_names = list(d["module_names"])
        m.maps = {}
        # per-MODULE-INDEX expected trailing dim d_i: within one map, all its
        # non-scalar buffers share a trailing dim (checked below); across
        # concepts, the SAME module index i must keep agreeing on d_i, but
        # different indices may legitimately differ -- mixed-width module
        # sets (e.g. llama icml24's gate/up_proj at 14336 vs down_proj at
        # 4096) are exactly this: same index every concept, different index
        # across the module set.
        expected_d_by_module: dict = {}
        for cid, states in d["maps"].items():
            if len(states) != len(m.module_names):
                raise ValueError(f"concept {cid!r}: {len(states)} fitted maps != "
                                 f"{len(m.module_names)} module_names in the artifact")
            built = []
            for i, state in enumerate(states):
                if set(state) != set(cls.fitted_buffers):
                    raise ValueError(f"concept {cid!r} module {i}: artifact buffers "
                                     f"{sorted(state)} != expected {sorted(cls.fitted_buffers)}")
                mp = m._new_map()
                for name, tensor in state.items():
                    mp.register_buffer(name, tensor)
                # every non-scalar buffer THIS map carries must share the same
                # trailing dim d_i -- a mismatch means a corrupt/mixed-up save.
                dims = {tensor.shape[-1] for tensor in state.values() if tensor.dim() >= 1}
                if len(dims) > 1:
                    raise ValueError(
                        f"concept {cid!r} module {i} ({m.module_names[i]!r}): buffer "
                        f"trailing dims disagree: "
                        f"{ {n: tuple(t.shape) for n, t in state.items()} }")
                d_here = next(iter(dims), None)
                if d_here is not None:
                    if i not in expected_d_by_module:
                        expected_d_by_module[i] = d_here
                    elif d_here != expected_d_by_module[i]:
                        raise ValueError(
                            f"concept {cid!r} module {i} ({m.module_names[i]!r}): buffer "
                            f"dim {d_here} != {expected_d_by_module[i]} seen earlier in "
                            f"this artifact for module index {i}")
                built.append(mp)
            m.maps[cid] = built
        return m


class MeanAcTModel(AcTTransportModel):
    weight_name = "MeanAcT"
    config_cls = MeanAcTConfig
    fitted_buffers = frozenset({"mu1", "mu2", "mask", "q_lo", "q_hi"})

    def _new_map(self):
        from .maps import MeanShiftMap

        return MeanShiftMap()

    def _fit_map(self, k, name, pos_pool, neg_pool, maps_so_far):
        from .maps import MeanShiftMap

        return MeanShiftMap().fit(neg_pool, pos_pool, support=self.config.support,
                                  std_eps=self.config.std_eps)


class LinearAcTModel(AcTTransportModel):
    weight_name = "LinearAcT"
    config_cls = LinearAcTConfig
    fitted_buffers = frozenset({"w", "b", "mask", "q_lo", "q_hi"})

    def _new_map(self):
        from .maps import IndependentSortOTMap

        return IndependentSortOTMap()

    def _fit_map(self, k, name, pos_pool, neg_pool, maps_so_far):
        from .maps import IndependentSortOTMap

        return IndependentSortOTMap().fit(neg_pool, pos_pool, support=self.config.support,
                                          seed=self.config.seed)


def _pid_repo_compat_u(r_hist):
    """pidup/Mean-AcT act/hooks/transport.py's GaussianOTPIDHook, the
    RELEASED (degenerate) law -- fit():404-412 sets diff = mu2-mu1 = r(k);
    load_state_dict():354-383 globs every EARLIER module's saved `diff`
    (break at the current file: strictly exclusive of k, but in
    FILESYSTEM GLOB order over .statedict names, not fit-time incremental
    order -- a stale/reordering hazard on multi-digit layer indices we do
    NOT reproduce, see the README), concatenates those 1-D [d] vectors
    along dim=0 (flattening across BOTH coordinates and layers into one
    long vector) and takes ITS mean -- collapsing the "integral" to a
    single GLOBAL SCALAR s_k, no per-coordinate structure at all:
        diff_m = diff + 0.005 * (s_k + diff) = 1.005 * diff + 0.005 * s_k
    forward():484 (OnlyMeanPIDHook, hook_onlymean=True): z_ot = z + 0.7 *
    diff_m, i.e.
        u(k) = 0.7 * diff_m = 0.7035 * r(k) + 0.0035 * s_k * ones(d)
    "earlier layers" here = REGISTRATION order (j < k, exclusive) as a
    deterministic stand-in for the glob order -- the approximation this
    repo_compat flag discloses, not the literal bug. k=0 (no earlier
    .statedict files on disk yet): load_state_dict's glob finds nothing,
    so the fork's `diff_m` is never rebuilt from a concatenation at all --
    module 0 gets u = 0.7 * diff = 0.7 * r(0) EXACTLY, not 0.7035 * r(0)
    (that only applies once later modules concatenate r(0) into their own
    s_k -- :382-383's `if not prev: return`)."""
    import torch

    r_k = r_hist[-1]
    prev = r_hist[:-1]
    if not prev:
        return 0.7 * r_k
    s = torch.cat(prev).mean()
    diff_m = 1.005 * r_k + 0.005 * s
    return 0.7 * diff_m


def _pid_fit_map(pos_pool, neg_pool, maps_so_far, kp, ki, kd, repo_compat, inclusive_sum,
                 std_eps=1e-4):
    """Canonical PID-AcT law (paper Def. 1 / Eq. 18, Eqs. 4-5/19 for the
    estimation): r(k) = E_dst[x(k)] - E_src[x~(k)] (pos_pool/neg_pool are
    this module's capture, already reflecting whatever estimation branch
    the caller chose); u(k) = Kp*r(k) + Ki*sum_{j<k}r(j) + Kd*(r(k)-r(k-1)),
    PER-COORDINATE, the integral sum EXCLUSIVE of k by default
    (inclusive_sum=True sums j<=k instead); D-term at k=0 treats r(-1)=0.
    repo_compat carries the fork's own std mask (std1 AND std2 > std_eps,
    same rule as Mean) on the fitted map; canonical mode's mask stays
    wide-open (README)."""
    r_k = (pos_pool.double().mean(dim=0) - neg_pool.double().mean(dim=0)).float()
    r_hist = [m.r.to(r_k.device) for m in maps_so_far] + [r_k]

    if repo_compat:
        u_k = _pid_repo_compat_u(r_hist)
    else:
        # the canonical per-coordinate PID law (paper Def. 1) is only
        # well-defined on a UNIFORM-WIDTH module set -- Kp*r(k) +
        # Ki*sum(r(j)) + Kd*(r(k)-r(k-1)) requires every r(j) in the
        # history to share r(k)'s shape. Mixed-width sets (e.g. llama
        # icml24's gate/up_proj @ 14336 vs down_proj @ 4096) are
        # traversable only by the released code's degenerate scalar
        # integral (repo_compat's torch.cat of mixed widths along dim=0
        # still works, and the resulting scalar s_k broadcasts to any
        # width -- left working, see that path above). Pick a
        # uniform-width module set (e.g. all_ln) for canonical PID.
        for j, r_j in enumerate(r_hist):
            if r_j.shape != r_k.shape:
                raise ValueError(
                    f"canonical PID (repo_compat=False) needs a uniform-width "
                    f"module set: r(k) has shape {tuple(r_k.shape)} but r({j}) "
                    f"in the history has shape {tuple(r_j.shape)} -- mixed-width "
                    f"module sets (e.g. llama icml24) are only traversable via "
                    f"the released code's degenerate scalar integral "
                    f"(repo_compat=True); pick a uniform-width module set (e.g. "
                    f"all_ln) for canonical PID")
        prev = r_hist[:-1]
        integral_terms = r_hist if inclusive_sum else prev
        i_k = sum(integral_terms, 0.0 * r_k)
        r_prev = prev[-1] if prev else 0.0 * r_k  # r(-1) := 0
        d_k = r_k - r_prev
        u_k = kp * r_k + ki * i_k + kd * d_k
    from .maps import PIDShiftMap, std_mask

    mask = None
    if repo_compat:
        mask = std_mask(neg_pool.double().std(dim=0), pos_pool.double().std(dim=0), std_eps)
    return PIDShiftMap(u_k, r_k, mask=mask)


def module_output_width(module):
    """The hooked module's output width (= the space its map/residual lives
    in), or None when the module type is unrecognized: nn.Linear-style
    out_features, LayerNorm-style normalized_shape, or a lone weight's
    leading dim (1-D RMSNorm weight = width; 2-D Linear weight = [out, in])."""
    if hasattr(module, "out_features"):
        return int(module.out_features)
    if getattr(module, "normalized_shape", None):
        return int(module.normalized_shape[-1])
    w = getattr(module, "weight", None)
    if w is not None and w.ndim in (1, 2):
        return int(w.shape[0])
    return None


class PIDAcTModel(AcTTransportModel):
    weight_name = "PIDAcT"
    config_cls = PIDAcTConfig
    fitted_buffers = frozenset({"u", "r", "mask", "q_lo", "q_hi"})

    def _check_modules(self, names, modules):
        # The canonical per-coordinate law (Def. 1) sums residuals ACROSS
        # modules — only well-defined when every hooked module has the same
        # output width (a fundamental limit of the approach; fail loudly up
        # front rather than mid-fit). The
        # released code's repo_compat scalar integral broadcasts into any
        # width, so the parity mode stays runnable on mixed-width sets
        # (their published llama runs used exactly that).
        if self.config.repo_compat:
            return
        widths = {}
        for name, module in zip(names, modules):
            w = module_output_width(module)
            if w is not None:
                widths.setdefault(w, []).append(name)
        if len(widths) > 1:
            summary = "; ".join(
                f"d={w}: {ns[0]}{' (+' + str(len(ns) - 1) + ' more)' if len(ns) > 1 else ''}"
                for w, ns in sorted(widths.items()))
            raise ValueError(
                f"PIDAcT (canonical) needs a uniform-width module set — the "
                f"per-coordinate integral sums residuals across modules. Got "
                f"{len(widths)} widths: {summary}. Pick a uniform preset "
                f"(e.g. all_ln) or repo_compat=True (the released scalar law).")

    def _new_map(self):
        import torch

        from .maps import PIDShiftMap

        return PIDShiftMap(torch.empty(0), torch.empty(0))

    def _target_steered(self) -> bool:
        if self.config.estimation not in ("source_only", "both"):
            raise ValueError(f"estimation {self.config.estimation!r}; "
                             "expected 'source_only' or 'both'")
        if self.config.repo_compat and self.config.estimation != "both":
            raise ValueError(
                "repo_compat=True reproduces the fork's shared incremental loop, which "
                "steers both branches -- set estimation='both'")
        return self.config.estimation == "both" or self.config.repo_compat

    def _fit_map(self, k, name, pos_pool, neg_pool, maps_so_far):
        return _pid_fit_map(pos_pool, neg_pool, maps_so_far, self.config.kp, self.config.ki,
                           self.config.kd, self.config.repo_compat, self.config.inclusive_sum,
                           std_eps=self.config.std_eps)

    def context(self, concept: Concept, alpha: float = 1.0,
               positions: str | None = None) -> MultiModuleTransformContext:
        from .maps import PIDShiftMap

        modules = self._resolved_modules()
        dosed = [PIDShiftMap(m.u, m.r, alpha=alpha, mask=m.mask) for m in self.maps[concept.id]]
        return MultiModuleTransformContext(modules, dosed, lmbda=1.0,
                                           positions=positions or self.config.positions,
                                           names=self.module_names)
