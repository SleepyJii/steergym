from dataclasses import dataclass

from ..axbench import AxbenchRowConfig, AxbenchRowModel


@dataclass
class PersonaVectorConfig(AxbenchRowConfig):
    # config of record: gen_traits.py (LAYER=20, GEN_BATCH=16, SEED=42)
    batch_size: int = 16
    seed: int = 42


def _fit_persona_rows(generator, concepts, samples, config, log=lambda msg: None):
    """gen_traits.py's PersonaVector pass, replicated with a plain forward
    hook instead of axbench's gather_residual_activations — itself a batched
    port of reference/generate_vec.py's get_hidden_p_and_r at a single
    layer: per concept, response_avg_diff = mean over EFFECTIVE pos rows of
    the response-position resid average minus the same over effective neg
    rows. Re-forward text = chat_prompt + answer with
    add_special_tokens=False (their line 23); prompt_len from tokenizing the
    chat prompt ALONE (their line 24), never inferred from the joint
    tokenization; degenerate (empty-response) rows fall back to the last
    prompt token. Returns RAW rows — the caller applies the unit/raw storage
    convention. `samples` = PersonaVectorsBenchmark.extraction_samples()."""
    import time

    import polars as pl
    import torch
    from transformers import set_seed

    set_seed(config.seed)
    tokenizer = generator.tokenizer
    layer_module = generator.model.model.layers[config.layer]

    def response_avg_resid(texts, answers):
        tokenizer.padding_side = "right"
        prompt_lens = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
        joint = [t + a for t, a in zip(texts, answers)]
        vecs = []
        for i in range(0, len(joint), config.batch_size):
            chunk = joint[i:i + config.batch_size]
            plens = prompt_lens[i:i + config.batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            add_special_tokens=False)
            enc = {k: t.to(generator.model.device) for k, t in enc.items()}
            captured = {}

            def hook(mod, args, output, captured=captured):
                captured["h"] = output[0] if isinstance(output, tuple) else output

            handle = layer_module.register_forward_hook(hook)
            with torch.no_grad():
                generator.model(**enc)
            handle.remove()
            h = captured["h"]  # [B, T, d]
            total_lens = enc["attention_mask"].sum(dim=1).tolist()
            for b, plen in enumerate(plens):
                rlen = total_lens[b] - plen
                if rlen <= 0:
                    vecs.append(h[b, max(plen - 1, 0)].float().cpu())
                    continue
                vecs.append(h[b, plen:plen + rlen].mean(dim=0).float().cpu())
        return torch.stack(vecs)

    meta = pl.col("meta")
    rows = []
    for concept in concepts:
        t0 = time.time()
        eff = samples.filter(
            (pl.col("concept_id") == concept.id)
            & (meta.struct.field("effective") == True))  # noqa: E712 (polars expr)
        arms = {}
        for arm in ("pos", "neg"):
            arm_rows = eff.filter(meta.struct.field("label") == arm)
            if arm_rows.height == 0:
                raise ValueError(f"concept {concept.id} ({concept.name!r}) has no "
                                 f"effective {arm} extraction rows")
            texts = [generator.chat_prompt(t) for t in arm_rows.get_column("prompt").to_list()]
            arms[arm] = response_avg_resid(texts, arm_rows.get_column("response").to_list())
        rows.append(arms["pos"].mean(dim=0) - arms["neg"].mean(dim=0))
        log(f"PersonaVector concept {concept.id} ({concept.name!r}): "
            f"{eff.height} effective rows, {time.time() - t0:.1f}s")
    return torch.stack(rows)


class PersonaVectorModel(AxbenchRowModel):
    """persona_vectors rows (Chen et al. 2025, arXiv:2507.21509).
    AxbenchRowModel (not bare RowVectorModel) because the rows persist in
    the axbench dump format (PersonaVector_weight.pt/_weight_raw.pt), and
    the unit/raw dual-file convention is exactly the weight_filename hook. Steering/save/load behavior is otherwise the
    inherited RowVectorModel.

    Default config (unit_norm=True) loads/stores the unit rows -- axbench's
    max-act dosing already scales with ||v||, so unit rows are the correct
    context(factor=...) convention (raw rows overdose
    ~65-100x). With config.unit_norm=False the RAW rows are used (upstream's
    raw-coefficient convention) -- pair with context(strength=...), never
    context(factor=...). Native fit() consumes
    PersonaVectorsBenchmark.extraction_samples()."""

    weight_name = "PersonaVector"
    config_cls = PersonaVectorConfig

    @classmethod
    def weight_filename(cls, config) -> str:
        return "PersonaVector_weight.pt" if config.unit_norm else "PersonaVector_weight_raw.pt"

    def fit(self, concepts, samples):
        import torch

        from ..generator import GeneratorLM

        model, tokenizer = self._require_bound()
        generator = GeneratorLM(self.config.subject, model, tokenizer)
        raw = _fit_persona_rows(generator, concepts, samples, self.config)
        if self.config.unit_norm:
            norms = raw.norm(dim=-1, keepdim=True)
            self.raw_norms = norms.squeeze(-1).tolist()
            self.rows = raw / norms.clamp_min(1e-8)
        else:
            self.raw_norms = None
            self.rows = raw
        self.bias = torch.zeros(len(concepts), dtype=torch.float32)
        self.concept_index = {c.id: i for i, c in enumerate(concepts)}
        return self
