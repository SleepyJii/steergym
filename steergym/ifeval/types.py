# Type-set construction: Stolfo's augmented pool -> per-type train/eval packs,
# and the type_id (pool id) <-> instruction_id (checker id) split needed to fit
# one vector per pool id while still scoring against the official checker.
# Param-driven: config drives, nothing reads the pool at import time.
#
# Protocol provenance: mimics Heyman & Vandeputte 2026 (arXiv:2605.03907) --
# Stolfo et al. 2025's augmented IFEval (12 format types + response-language
# per language: their paper says 14 languages, the Stolfo pool carries 19
# distinct ids and which 14 they used is not recoverable, so "format+language"
# and "all" use ALL 19 present rather than guess a subset). TYPES is derived
# from what the pool actually contains, never hardcoded from INSTRUCTION_DICT
# (which carries `detectable_format:multiple_sections_improved`, absent from
# all 11242 pool rows -- verified 2026-09-18).
import json
import random
from collections import Counter
from pathlib import Path

from .evals import _registry

STOLFO_DIR = Path(__file__).resolve().parent / "vendor" / "stolfo"
AUGMENTED_FILENAME = "ifeval_augmented_filtered.jsonl"
SINGLE_INSTR_FILENAME = "ifeval_single_instr_format.jsonl"

FORMAT_FAMILIES = ("detectable_format", "change_case", "punctuation", "startend")
LANGUAGE_PREFIX = "language:response_language_"
LANGUAGE_CHECKER_ID = "language:response_language"
_SUFFIXES = ("_at least", "_less than")  # order matters: "at least" before "less than"

TYPE_SETS = ("format", "format+language", "all")


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def checker_id(type_id):
    """Pool id (type_id, unique per concept) -> INSTRUCTION_DICT key (the
    official checker id, shared by e.g. both `keywords:frequency_at least`
    and `keywords:frequency_less than`, and by every `language:response_
    language_<xx>`)."""
    if type_id.startswith(LANGUAGE_PREFIX):
        return LANGUAGE_CHECKER_ID
    for suf in _SUFFIXES:
        if type_id.endswith(suf):
            return type_id[: -len(suf)]
    return type_id


def family(type_id):
    """Family prefix (before the first ':') -- for reporting, not selection."""
    return type_id.split(":", 1)[0]


def select_types(type_set: str, stolfo_dir=STOLFO_DIR, n_types: int | None = None):
    """The pool ids for a TYPE_SET, in canonical order:
    format types in INSTRUCTION_DICT declared order, then languages sorted by
    code, then the rest in INSTRUCTION_DICT order with `_at least` before
    `_less than` where a checker has both pool ids."""
    if type_set not in TYPE_SETS:
        raise ValueError(f"type_set {type_set!r} must be one of {TYPE_SETS}")
    registry = _registry()
    present = {r["single_instruction_id"]
               for r in _load_jsonl(Path(stolfo_dir) / AUGMENTED_FILENAME)}
    format_types = tuple(
        k for k in registry.INSTRUCTION_DICT
        if family(k) in FORMAT_FAMILIES and k in present)
    language_types = tuple(sorted(t for t in present if t.startswith(LANGUAGE_PREFIX)))
    rest = []
    for k in registry.INSTRUCTION_DICT:
        if family(k) in FORMAT_FAMILIES or k == LANGUAGE_CHECKER_ID:
            continue
        for suf in _SUFFIXES + ("",):
            cand = k + suf
            if cand in present and cand not in rest:
                rest.append(cand)

    if type_set == "format":
        types = format_types
    elif type_set == "format+language":
        types = format_types + language_types
    else:
        types = format_types + language_types + tuple(rest)
    return types[:n_types] if n_types is not None else types


def _hashable(v):
    """Recursively tuple-ify list values so a kwargs dict can be a Counter
    key (e.g. KeywordChecker's `keywords` kwarg is a list)."""
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    return v


def _row_kwargs(row):
    """Stolfo rows carry `kwargs` as a one-element list (verified: every
    sampled row in both pool files, mirroring IFEval's own per-instruction
    kwargs list for a single-instruction row)."""
    k = row.get("kwargs")
    return (k[0] if k else {}) or {}


def _valid_rows(rows, type_id):
    """Drop rows whose kwargs, filtered to this checker's accepted keys, are
    empty while the checker actually expects keys -- e.g. `keywords:existence`
    rows carrying only a stray `prompt_to_repeat`. Left in, these would
    silently build a checker with build_description's own randomised default
    instead of reflecting the row: a plausible-looking but fake
    positive/negative. Zero-kwarg checkers are never dropped (allowed empty,
    guard vacuous)."""
    registry = _registry()
    cid = checker_id(type_id)
    allowed = set(registry.INSTRUCTION_DICT[cid](cid).get_instruction_args_keys())
    valid, n_stray = [], 0
    for r in rows:
        call_kwargs = {k: v for k, v in _row_kwargs(r).items() if k in allowed}
        if allowed and not call_kwargs:
            n_stray += 1
            continue
        valid.append(r)
    return valid, n_stray


def build_instruction(type_id, kwargs):
    """(desc, effective_kwargs) for one type_id + row kwargs, filtered to the
    keys build_description actually accepts and read back via
    get_instruction_args() since unset kwargs get randomised. Builds against
    checker_id(type_id) -- the registry key -- never type_id itself, which for
    language/keyword/length pool ids is not a registry key."""
    registry = _registry()
    cid = checker_id(type_id)
    inst = registry.INSTRUCTION_DICT[cid](cid)
    allowed = set(inst.get_instruction_args_keys())
    call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    desc = inst.build_description(**call_kwargs)
    eff = inst.get_instruction_args() or {}
    return desc, eff


def _take_valid(candidates, type_id, n, with_key=False, errors=None):
    """Build entries from stray-filtered candidates in order, skipping rows
    whose kwargs fail the checker's own validation (observed: the pool carries
    `capital_relation: "at most"`, which instructions.py rejects -- a real
    data/checker mismatch) until n are collected or candidates run out."""
    out, used_bases = [], set()
    for r in candidates:
        if len(out) >= n:
            break
        try:
            desc, eff = build_instruction(type_id, _row_kwargs(r))
        except Exception as e:
            if errors is not None:
                errors.append((type_id, r.get("key"), str(e)))
            continue
        entry = {"base": r["prompt_without_instruction"], "desc": desc, "kwargs": eff}
        if with_key:
            entry["key"] = r.get("key")
        out.append(entry)
        used_bases.add(entry["base"])
    return out, used_bases


def build_packs(types, n_train, n_eval, seed, stolfo_dir=STOLFO_DIR, log=print):
    """One pack per type: {concept_id, type_id, instruction_id, concept (the
    canonical desc at the pack's most common effective kwargs), train
    [{base, desc, kwargs}], eval [{base, desc, kwargs, key}]}. Eval rows
    prefer bases from the 163-row single-instr set; train bases are disjoint
    from eval. Small types (languages sit at 9-49 rows) quarter-split instead
    of starving train."""
    stolfo_dir = Path(stolfo_dir)
    pool = _load_jsonl(stolfo_dir / AUGMENTED_FILENAME)
    single = _load_jsonl(stolfo_dir / SINGLE_INSTR_FILENAME)
    pool_by_type = {t: [r for r in pool if r["single_instruction_id"] == t] for t in types}
    skipped, stray_counts, small_types = [], {}, []

    packs = []
    for cid, tid in enumerate(types):
        rows, n_stray = _valid_rows(pool_by_type[tid], tid)
        stray_counts[tid] = n_stray
        rng = random.Random(seed + cid)

        count = len(rows)
        if count >= n_train + n_eval:
            n_eval_t, n_train_t = n_eval, n_train
        else:
            n_eval_t = min(max(3, count // 4), max(count - 1, 0))
            n_train_t = count - n_eval_t
            small_types.append((tid, count, n_train_t, n_eval_t))

        iid = checker_id(tid)
        eval_bases = {r["prompt_without_instruction"] for r in single
                      if iid in r.get("instruction_id_list", [])}
        preferred = [r for r in rows if r["prompt_without_instruction"] in eval_bases]
        rng.shuffle(preferred)
        eval_entries, eval_bases_used = _take_valid(preferred, tid, n_eval_t, with_key=True, errors=skipped)
        if len(eval_entries) < n_eval_t:
            rest = [r for r in rows if r["prompt_without_instruction"] not in eval_bases_used]
            rng.shuffle(rest)
            more, more_bases = _take_valid(rest, tid, n_eval_t - len(eval_entries), with_key=True, errors=skipped)
            eval_entries += more
            eval_bases_used |= more_bases

        train_candidates = [r for r in rows if r["prompt_without_instruction"] not in eval_bases_used]
        rng.shuffle(train_candidates)
        train_entries, _ = _take_valid(train_candidates, tid, n_train_t, errors=skipped)

        all_eff = [tuple(sorted((k, _hashable(v)) for k, v in e["kwargs"].items()))
                   for e in train_entries + eval_entries]
        mode_eff = {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in Counter(all_eff).most_common(1)[0][0]} if all_eff else {}
        concept_desc, _ = build_instruction(tid, mode_eff)

        packs.append({
            "concept_id": cid,
            "type_id": tid,
            "instruction_id": iid,
            "concept": concept_desc,
            "train": train_entries,
            "eval": eval_entries,
        })
        if not train_entries or not eval_entries:
            log(f"[ifeval.types] WARNING: {tid} ended up with "
                f"{len(train_entries)} train / {len(eval_entries)} eval rows "
                f"(pool had {count} valid rows)")

    if skipped:
        log(f"[ifeval.types] skipped {len(skipped)} rows with checker-rejected kwargs "
            f"(e.g. {skipped[0]})")
    n_dropped = sum(stray_counts.values())
    if n_dropped:
        detail = ", ".join(f"{t}={n}" for t, n in stray_counts.items() if n)
        log(f"[ifeval.types] dropped {n_dropped} stray rows (empty filtered kwargs): {detail}")
    if small_types:
        detail = ", ".join(f"{t} (pool={c}, train={ntr}, eval={ne})" for t, c, ntr, ne in small_types)
        log(f"[ifeval.types] {len(small_types)} small types quarter-split "
            f"(count < n_train+n_eval={n_train + n_eval}): {detail}")
    return packs
