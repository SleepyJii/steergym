"""Multi-module transform context: AcT targets N arbitrary modules (not a
single decoder layer) -- Gemma-2's "Post-LN" set is the 52
post_attention_layernorm/post_feedforward_layernorm modules (branch
outputs, pre-residual-add), Llama's default `icml24` set is mlp
up/down/gate projections. `find_module_names` is the registration-order
module-set resolver (port of ml-act model_with_hooks.py:143-166's
fullmatch semantics -- the ':N' sub-tensor-channel convention in
is_module_name_in_regex is out of scope, no LLM decoder module needs it).

`MultiModuleTransformContext` is `TransformSteerContext`/`ALQRContext`'s
discipline generalized to N independently-masked per-module maps: per
module k, `z' = lmbda * T_k(z) + (1 - lmbda) * z` for units INSIDE that
map's support+std mask (maps.py), z unchanged outside. positions: "all"
(ml-act's native default -- every prompt + generated position); "last"
(ml-act's own other `intervention_position` value -- the final position of
whatever is passed through in ONE forward, e.g. a non-cached fit-time
pass); "generated" (our KV-cache convention for cross-paradigm comparison
-- odesteer/vector.py's steer_position_idx=-1: the last prompt token at
prefill, then every newly generated token). "last" and "generated" hook
the same position (-1) -- the distinction is which generation protocol the
caller is in, not a code difference.

PID-AcT shares this exact class: PIDAcTModel.context() builds per-module
`PIDShiftMap`s (mask wide-open) and calls here with lmbda=1.0 fixed (open
loop, no interpolation) -- the dose "alpha" is baked into each map's
`.transport` closure, not a context constructor argument.
"""
import re

from ..steer import SteerContext

MODULE_PRESETS = {
    # Gemma-2 "Post-LN": the 2 norm modules per decoder layer whose OUTPUT
    # feeds the residual add (AcT paper's gemma-2-2b target set).
    "postln": (r".*post_attention_layernorm", r".*post_feedforward_layernorm"),
    "all_ln": (r".+layernorm",),
    # AcT paper's Llama default: mlp projections. GAP (spec, recorded in
    # the README): the paper's separate Llama "Attention" module set
    # (probably self_attn.o_proj) is unshipped -- not guessed here.
    "icml24": (r".*\.mlp\.up_proj", r".*\.mlp\.down_proj", r".*\.mlp\.gate_proj"),
}


def resolve_module_regexes(config) -> tuple[str, ...]:
    if config.module_regexes:
        return tuple(config.module_regexes)
    try:
        return MODULE_PRESETS[config.module_preset]
    except KeyError:
        raise ValueError(
            f"module_preset {config.module_preset!r}; expected one of "
            f"{sorted(MODULE_PRESETS)} or an explicit module_regexes") from None


def find_module_names(model, regexes) -> list[str]:
    """named_modules() REGISTRATION ORDER (= topological order, load-
    bearing for the sequential fit), any regex in `regexes` fullmatch,
    de-duplicated, root module ('') skipped."""
    names = []
    for name, _ in model.named_modules():
        if name == "":
            continue
        if name not in names and any(re.fullmatch(r, name) for r in regexes):
            names.append(name)
    return names


class ModuleReached(Exception):
    """ml-act scripts/learn_intervention.py's TargetModuleReached trick
    (`except hooks.custom_exceptions.TargetModuleReached: pass` around the
    forward call): the sequential fit's capture hook raises this right
    after module k's pooled stats are captured, aborting the rest of that
    forward. Each capture pass still runs every layer UP TO k (not O(1)
    work) -- the sequential fit over L modules is ~L/2 full-forwards-
    equivalent per branch (sum_{k} k layer evals = O(L^2) layer
    evaluations total), and that is PER CONCEPT (re-run from scratch for
    every concept, no caching across concepts); cost note in the package
    README."""


class MultiModuleTransformContext(SteerContext):
    def __init__(self, modules, maps, lmbda: float = 1.0, positions: str = "all", names=None):
        if positions not in ("all", "last", "generated"):
            raise ValueError(f"positions {positions!r}; expected 'all', 'last', or 'generated'")
        if len(modules) != len(maps):
            raise ValueError(f"{len(modules)} modules != {len(maps)} maps")
        if names is not None and len(names) != len(modules):
            raise ValueError(f"{len(names)} names != {len(modules)} modules")
        self.modules = list(modules)
        self.maps = list(maps)
        self.lmbda = float(lmbda)
        self.positions = positions
        self.names = list(names) if names is not None else None
        self._handles = None
        self._stats = {}  # (k, device) -> [rows, sum ||dz|| (0-d), max ||dz|| (0-d),
                          #                  sum in-support cells (0-d), total cells]

    def _apply(self, k, rows):
        import torch

        m = self.maps[k]
        m.to(rows.device)  # buffers may still sit on CPU after fit/load
        with torch.no_grad():
            x = rows.float()
            t = m.transport(x)
            mask = m.mask & (m.q_lo < x) & (x < m.q_hi)
            blended = self.lmbda * t + (1 - self.lmbda) * x
            out = torch.where(mask, blended, x)
            delta = (out - x).norm(dim=-1)
            in_support_cells = mask.sum()
            total_cells = mask.numel()
            key = (k, delta.device)
            st = self._stats.get(key)
            if st is None:
                self._stats[key] = [len(delta), delta.sum(), delta.max(),
                                    in_support_cells, total_cells]
            else:
                st[0] += len(delta)
                st[1] = st[1] + delta.sum()
                st[2] = torch.maximum(st[2], delta.max())
                st[3] = st[3] + in_support_cells
                st[4] += total_cells
        return out.to(rows.dtype)

    def __enter__(self):
        if self._handles is not None:
            raise RuntimeError("context already entered (nested entry would leak hooks)")
        self._handles = []
        for k, module in enumerate(self.modules):
            def hook(mod, args, output, k=k):
                h = output[0] if isinstance(output, tuple) else output
                if self.positions == "all":
                    B, T, d = h.shape
                    h2 = self._apply(k, h.reshape(B * T, d)).reshape(B, T, d)
                else:
                    h2 = h.clone()
                    h2[:, -1, :] = self._apply(k, h[:, -1, :])
                return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        if self._handles is not None:
            for handle in self._handles:
                handle.remove()
            self._handles = None
        return False

    def diagnostics(self) -> dict:
        per_module = {}
        by_k = {}
        for (k, _device), st in self._stats.items():
            agg = by_k.setdefault(k, [0, 0.0, 0.0, 0.0, 0])
            agg[0] += st[0]
            agg[1] += float(st[1])
            agg[2] = max(agg[2], float(st[2]))
            agg[3] += float(st[3])
            agg[4] += int(st[4])
        total_rows = total_sum = 0
        total_max = 0.0
        in_support_weighted_sum = 0.0  # sum_k(rows_k * in_support_frac_k) -- overall
                                       # in_support_frac is weighted BY ROWS, not by
                                       # cells, so modules with different trailing
                                       # dims d_k don't skew the aggregate
        for k, (rows, s, mx, in_support_cells, cells) in by_k.items():
            key = self.names[k] if self.names is not None else k
            frac = in_support_cells / cells if cells else 0.0
            per_module[key] = {
                "steered_rows": rows,
                "applied_delta_norm_mean": s / rows if rows else 0.0,
                "applied_delta_norm_max": mx,
                "in_support_frac": frac,
            }
            total_rows += rows
            total_sum += s
            total_max = max(total_max, mx)
            in_support_weighted_sum += rows * frac
        return {
            "per_module": per_module,
            "steered_rows": total_rows,
            "applied_delta_norm_mean": total_sum / total_rows if total_rows else 0.0,
            "applied_delta_norm_max": total_max,
            "in_support_frac": (in_support_weighted_sum / total_rows) if total_rows else 0.0,
        }
