from dataclasses import dataclass

from ..axbench import AxbenchRowConfig, AxbenchRowModel


@dataclass
class StolfoDiffMeanConfig(AxbenchRowConfig):
    # config of record: gen_pairs.py (LAYER=20, GEN_BATCH=16, SEED=42)
    batch_size: int = 16
    seed: int = 42


def _fit_stolfo_rows(generator, concepts, samples, config, log=lambda msg: None):
    """gen_pairs.py's StolfoDiffMean pass, replicated with a plain forward
    hook instead of axbench's gather_residual_activations: per concept, mean
    over ALL pack train rows (not checker-filtered) of resid_post[layer] at
    the last non-pad prompt token of chat(base + " " + desc) minus
    chat(base), unit-normed. `samples` = IFEvalBenchmark.stolfo_samples()
    (prompt = base, meta.desc = the row's instruction text)."""
    import time

    import polars as pl
    import torch
    from transformers import set_seed

    set_seed(config.seed)
    tokenizer = generator.tokenizer
    layer_module = generator.model.model.layers[config.layer]

    def last_token_resid(texts, batch_size):
        tokenizer.padding_side = "right"  # right-pad: attention_mask.sum()-1 = last real token
        vecs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            add_special_tokens=True)
            enc = {k: t.to(generator.model.device) for k, t in enc.items()}
            captured = {}

            def hook(mod, args, output, captured=captured):
                captured["h"] = output[0] if isinstance(output, tuple) else output

            handle = layer_module.register_forward_hook(hook)
            with torch.no_grad():
                generator.model(**enc)
            handle.remove()
            h = captured["h"]  # [B, T, d]
            last = enc["attention_mask"].sum(dim=1) - 1
            for b in range(len(chunk)):
                vecs.append(h[b, last[b]].float().cpu())
        return torch.stack(vecs)

    desc = pl.col("meta").struct.field("desc")
    rows = []
    for concept in concepts:
        t0 = time.time()
        concept_rows = samples.filter(pl.col("concept_id") == concept.id)
        bases = concept_rows.get_column("prompt").to_list()
        descs = concept_rows.select(desc).to_series().to_list()
        constrained = [generator.chat_prompt(f"{b} {d}") for b, d in zip(bases, descs)]
        plain = [generator.chat_prompt(b) for b in bases]
        diff = (last_token_resid(constrained, config.batch_size)
                - last_token_resid(plain, config.batch_size)).mean(dim=0)
        rows.append(diff / diff.norm().clamp_min(1e-8))
        log(f"StolfoDiffMean concept {concept.id} ({concept.name!r}): "
            f"{len(bases)} pairs, {time.time() - t0:.1f}s")
    return torch.stack(rows), torch.zeros(len(rows), dtype=torch.float32)


class StolfoDiffMeanModel(AxbenchRowModel):
    """Stolfo-style last-prompt-token diff-mean rows; native fit() over
    IFEvalBenchmark.stolfo_samples(), or precomputed rows via config.dump_dir
    like every AxbenchRowModel. Open problem, no answer yet: flat dose
    response at fixed L20 under max-act scaling; their
    per-type layer search + projection-matched weight rule are unimplemented,
    and precomputed rows don't recover them."""

    weight_name = "StolfoDiffMean"
    config_cls = StolfoDiffMeanConfig

    def fit(self, concepts, samples):
        from ..generator import GeneratorLM

        model, tokenizer = self._require_bound()
        generator = GeneratorLM(self.config.subject, model, tokenizer)
        rows, biases = _fit_stolfo_rows(generator, concepts, samples, self.config)
        self.rows = rows
        self.bias = biases
        self.concept_index = {c.id: i for i, c in enumerate(concepts)}
        self.raw_norms = None
        return self
