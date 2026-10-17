# Trait pack construction: the vendored persona_vectors trait_data_{extract,
# eval} JSONs (schema: {instruction: [{pos,neg}]x5, questions: [str]x20,
# eval_prompt: str}) -> one pack per trait, param-driven (no env knobs).
#
# Protocol provenance: mimics persona_vectors steering eval (Chen et al.
# 2025, arXiv:2507.21509) -- their 7 published traits, 20 held-out eval
# questions/trait, trait+coherence judged over a 0-100 logprob-aggregated
# scale by gpt-4.1-mini. Standing deviations from upstream (disclosed, not
# silently applied): default subject gemma-2-2b-it (theirs Qwen2.5-7B/
# Llama-3.1-8B); no system role -- pos/neg system prompts folded into the
# user turn (fold()); steering = addition at ALL positions, dose = factor x
# max-act on UNIT rows (theirs: raw coefficient at response positions,
# ActivationSteerer layer_idx=layer-1; raw rows preserved for that path);
# layer 20 here is the axbench block-output convention = their hidden_states
# index 21; extraction generations 256 tokens T=1.0 (theirs up to 1000).
# Rollouts: upstream samples n_per_question responses per (question x
# instruction) prompt (eval_persona.py main() default 10 at T=1.0; T=0.0
# iff n=1) -- the adapter defaults to the upstream 10, with n_per_question/
# extraction_temperature explicit in PersonaVectorsConfig so test runs
# override deliberately.
import json
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "persona_vectors"

# the 7 traits persona_vectors ships (trait_data_extract/*.json filenames)
TRAITS = ("apathetic", "evil", "hallucinating", "humorous", "impolite",
          "optimistic", "sycophantic")


def a_or_an(word):
    """Verbatim from reference/eval/eval_persona.py:134-135."""
    return "an" if word[0].lower() in "aeiou" else "a"


def system_text(trait, instruction, sign):
    """Verbatim construction from reference/eval/eval_persona.py's
    load_persona_questions (:150-154): name = trait for the pos arm,
    "helpful" for the neg arm (their assistant_name default)."""
    name = trait if sign == "pos" else "helpful"
    return f"You are {a_or_an(name)} {name} assistant. {instruction}"


def fold(system, question):
    """The gemma no-system-role fold: their `system` message becomes a
    user-turn prefix instead (see the protocol note above)."""
    return f"{system}\n\n{question}"


def trait_description(eval_prompt):
    """The rubric's definition paragraph -- everything before the first
    '\\n\\nPrompt:' marker in the vendored eval_prompt."""
    marker = "\n\nPrompt:"
    idx = eval_prompt.find(marker)
    if idx < 0:
        raise ValueError(f"eval_prompt has no {marker!r} marker -- can't split "
                         "off the definition paragraph")
    return eval_prompt[:idx].strip()


def build_packs(traits=TRAITS, n_extract=20, n_eval=20, trait_data_dir=VENDOR_DIR, log=print):
    """One pack per trait: {concept_id, trait, description, eval_prompt,
    instruction (extract's own 5 pos/neg pairs), questions_extract,
    questions_eval} + *_differs flags. Extraction ALWAYS uses extract's own
    instruction pairs; eval_prompt comes from the eval file (what upstream's
    steering eval scores against) -- flagged rather than silently picked if
    the files ever disagree (verified identical across all 7 traits as
    vendored, 2026-09-18)."""
    trait_data_dir = Path(trait_data_dir)
    packs = []
    for cid, trait in enumerate(traits):
        extract = json.loads((trait_data_dir / "trait_data_extract" / f"{trait}.json").read_text())
        ev = json.loads((trait_data_dir / "trait_data_eval" / f"{trait}.json").read_text())

        assert len(extract["instruction"]) == 5, \
            f"{trait}: extract has {len(extract['instruction'])} instruction pairs, expected 5"
        assert len(ev["questions"]) >= n_eval, \
            f"{trait}: eval file has only {len(ev['questions'])} questions, need {n_eval}"
        assert len(extract["questions"]) >= n_extract, \
            f"{trait}: extract file has only {len(extract['questions'])} questions, need {n_extract}"

        eval_prompt = ev["eval_prompt"]
        assert "{question}" in eval_prompt and "{answer}" in eval_prompt, \
            f"{trait}: eval_prompt missing {{question}}/{{answer}} placeholders"
        eval_prompt_differs = extract["eval_prompt"] != eval_prompt
        instruction_differs = extract["instruction"] != ev["instruction"]

        packs.append({
            "concept_id": cid,
            "trait": trait,
            "description": trait_description(eval_prompt),
            "eval_prompt": eval_prompt,
            "eval_prompt_differs": eval_prompt_differs,
            "instruction": extract["instruction"],
            "instruction_differs": instruction_differs,
            "questions_extract": extract["questions"][:n_extract],
            "questions_eval": ev["questions"][:n_eval],
        })
        if eval_prompt_differs or instruction_differs:
            log(f"[personavec.traits] WARNING: {trait} extract/eval files disagree "
                f"(eval_prompt_differs={eval_prompt_differs} "
                f"instruction_differs={instruction_differs}) -- extraction uses "
                "extract's own instruction pairs; eval_prompt from the eval file")
    return packs
