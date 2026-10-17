import polars as pl

from .concept import Concept


def steer_sweep(generator, steer_model, concepts: list[Concept], eval_samples: pl.DataFrame,
                sweep: list[dict], gen) -> pl.DataFrame:
    """Long results table over concept x coeff-arm x prompt: render (the
    steer model's prompt seam) applied before chat templating, generation run
    inside the steer model's SteerContext (hooks armed for prefill + every decode
    step). One column per coeff key (struct `coeffs`) and per diagnostics key
    (struct `diag`, or a null column when every arm's diagnostics() is {})."""
    if gen.seed is not None:
        from transformers import set_seed
        set_seed(gen.seed)

    concept_ids, prompts, responses, coeffs_col, diag_col = [], [], [], [], []
    any_diag = False

    for concept in concepts:
        rows = eval_samples.filter(pl.col("concept_id") == concept.id)
        raw_prompts = rows.get_column("prompt").to_list()
        for coeffs in sweep:
            ctx = steer_model.context(concept, **coeffs)
            rendered = [ctx.render(p) for p in raw_prompts]
            chat = [generator.chat_prompt(r) for r in rendered]
            with ctx:
                arm_responses = generator.generate(chat, gen)
            diag = ctx.diagnostics()
            any_diag = any_diag or bool(diag)

            n = len(raw_prompts)
            concept_ids.extend([concept.id] * n)
            prompts.extend(raw_prompts)
            responses.extend(arm_responses)
            coeffs_col.extend([coeffs] * n)
            diag_col.extend([diag] * n)

    df = pl.DataFrame({
        "concept_id": concept_ids,
        "prompt": prompts,
        "response": responses,
        "coeffs": coeffs_col,
    })
    if any_diag:
        df = df.with_columns(pl.Series("diag", diag_col))
    else:
        df = df.with_columns(pl.lit(None).alias("diag"))
    return df
