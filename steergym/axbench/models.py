from dataclasses import dataclass
from pathlib import Path

from ..concept import Concept
from ..steer import SteerContext, SteerModel
from ..vector import RowVectorConfig, RowVectorModel


@dataclass
class AxbenchRowConfig(RowVectorConfig):
    dump_dir: str | None = None      # axbench train/ dump to load rows from
    hf_repo: str | None = None       # released HF dictionary instead of a dump
    hf_subfolder: str = "l20"


class AxbenchRowModel(RowVectorModel):
    """Abstract: RowVectorModel whose weights live in the axbench dump cache
    format — {weight_name}_weight.pt [C, d] (+ optional _bias.pt [C]) under a
    train/ dir, or a released HF dictionary with the same row layout. Whether
    the axbench cache is used is config (dump_dir / hf_repo), not baked into
    subclasses."""

    config_cls = AxbenchRowConfig
    weight_name: str | None = None  # filename stem in the dump cache

    @classmethod
    def weight_filename(cls, config) -> str:
        return f"{cls.weight_name or cls.__name__}_weight.pt"

    @classmethod
    def from_config_source(cls, config, concept_ids=None):
        if (config.dump_dir is None) == (config.hf_repo is None):
            raise ValueError("set exactly one of config.dump_dir or config.hf_repo")
        if config.dump_dir is not None:
            return cls.from_axbench_dump(config, config.dump_dir, concept_ids=concept_ids)
        return cls.from_released(config, concept_ids=concept_ids)

    @classmethod
    def from_axbench_dump(cls, config, dump_dir, concept_ids=None):
        import torch
        dump = Path(dump_dir)
        rows = torch.load(dump / cls.weight_filename(config), map_location="cpu")
        name = cls.weight_name or cls.__name__
        bias_path = dump / f"{name}_bias.pt"
        bias = torch.load(bias_path, map_location="cpu") if bias_path.exists() else None
        return cls.from_rows(config, rows, concept_ids=concept_ids, bias=bias)

    @classmethod
    def from_released(cls, config, concept_ids=None):
        import torch
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(config.hf_repo, "weight.pt", subfolder=config.hf_subfolder)
        return cls.from_rows(config, torch.load(path, map_location="cpu"), concept_ids=concept_ids)

    def save_axbench_dump(self, dump_dir):
        """Writes the axbench cache layout from_axbench_dump reads:
        {weight_name}_weight.pt [C, d] bf16, {weight_name}_bias.pt [C] bf16,
        config.json {model_name, layer, component}."""
        import json

        import torch

        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        name = type(self).weight_name or type(self).__name__
        torch.save(self.rows.to(torch.bfloat16), dump_dir / f"{name}_weight.pt")
        bias = self.bias if self.bias is not None else torch.zeros(self.rows.shape[0])
        torch.save(bias.to(torch.bfloat16), dump_dir / f"{name}_bias.pt")
        (dump_dir / "config.json").write_text(json.dumps({
            "model_name": self.config.subject.model_id,
            "layer": self.config.layer,
            "component": "res",
        }))


@dataclass
class DiffMeanConfig(AxbenchRowConfig):
    # config of record: sweep/wuzhengx/2b/l20/no_grad.yaml
    n_epochs: int = 1
    batch_size: int = 6
    max_length: int = 1024
    seed: int = 42


def _fit_diffmean_rows(generator, concepts, samples, config, log=lambda msg: None):
    """axbench mean.py DiffMean.train, replicated with a plain forward hook
    instead of pyvene: closed-form mean(positive activations) -
    mean(negative activations) at layer config.layer, pooled over
    [prefix_length:] non-pad token positions, unit-normalized. n_epochs=1 in
    the config of record so 'training' is a single pass over the data."""
    import time

    import torch
    from transformers import set_seed

    from ..activations import batched_layer_activations
    from .protocol import binarize_text, positive_and_negative_rows, prefix_suffix_lengths

    set_seed(config.seed)
    tokenizer = generator.tokenizer
    tokenizer.padding_side = "right"
    prefix_length, suffix_length = prefix_suffix_lengths(tokenizer)
    layer_module = generator.model.model.layers[config.layer]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rows, biases = [], []
    for concept in concepts:
        t0 = time.time()
        positive, negative = positive_and_negative_rows(samples, concept)

        texts, labels = [], []
        for row in positive.iter_rows(named=True):
            texts.append(binarize_text(tokenizer, row["prompt"], row["response"], suffix_length))
            labels.append(1)
        for row in negative.iter_rows(named=True):
            texts.append(binarize_text(tokenizer, row["prompt"], row["response"], suffix_length))
            labels.append(0)

        encoded = [tokenizer(t, max_length=1024, truncation=True)["input_ids"] for t in texts]
        n = len(encoded)
        order = torch.randperm(n).tolist() if n else []  # ambient global RNG, no explicit generator

        positive_activations, negative_activations = [], []
        start = 0
        for h_full, lengths in batched_layer_activations(
                generator.model, layer_module,
                [encoded[j] for j in order], config.batch_size, pad_id):
            B, T = len(lengths), h_full.shape[1]
            batch_labels = torch.tensor([labels[j] for j in order[start:start + B]],
                                        dtype=torch.long)
            start += B
            h = h_full[:, prefix_length:]  # [B, T-prefix, d]
            attention_mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)).long()
            nonbos_mask = attention_mask[:, prefix_length:].bool()
            activations = h[nonbos_mask]  # [N, d]
            per_token_labels = batch_labels.unsqueeze(1).repeat(1, T - prefix_length)[nonbos_mask]
            positive_activations.append(activations[per_token_labels == 1])
            negative_activations.append(activations[per_token_labels != 1])

        mean_pos = torch.cat(positive_activations, dim=0).float().mean(dim=0)
        mean_neg = torch.cat(negative_activations, dim=0).float().mean(dim=0)
        row = mean_pos - mean_neg
        norm = row.norm()
        eps = torch.finfo(row.dtype).eps
        row = row / (norm + eps)

        rows.append(row)
        biases.append(0.0)
        log(f"DiffMean concept {concept.id} ({concept.name!r}): "
            f"{n} rows, {time.time() - t0:.1f}s")

    return torch.stack(rows), torch.tensor(biases, dtype=torch.float32)


class DiffMeanModel(AxbenchRowModel):
    """Generic difference-of-means rows (trained by axbench's mean.py in our
    runs to date; released dictionary: pyvene/gemma-diffmean-2b-it-res —
    file layout assumed to mirror reft-r1, unverified)."""

    weight_name = "DiffMean"
    config_cls = DiffMeanConfig

    def fit(self, concepts, samples):
        from ..generator import GeneratorLM

        model, tokenizer = self._require_bound()
        generator = GeneratorLM(self.config.subject, model, tokenizer)
        rows, biases = _fit_diffmean_rows(generator, concepts, samples, self.config)
        self.rows = rows
        self.bias = biases
        self.concept_index = {c.id: i for i, c in enumerate(concepts)}
        self.raw_norms = None
        return self


# axbench dataset.py's non-API steering prompt (the has_prompt_steering=False
# branch; the gpt-4o "boosted" variant is an OpenAI call we never make)
T_PROMPT_STEERING = ("You must answer the question with content related to %s "
                     "even if it is not related to the question or it does not make sense.")


class AxbenchPromptSteerContext(SteerContext):
    def __init__(self, steering_prompt: str):
        self.steering_prompt = steering_prompt

    def render(self, prompt: str) -> str:
        return f"{self.steering_prompt}\n\nQuestion: {prompt}"


class AxbenchPromptSteerModel(SteerModel):
    """axbench's PromptSteering arm, verbatim: T_PROMPT_STEERING % concept.name
    joined as '{steering_prompt}\\n\\nQuestion: {prompt}' (dataset.py
    create_eval_df, AlpacaEval branch). A peer of models/prompt_steerer.py's
    generic PromptSteerModel, homed here because the template and joiner are
    axbench protocol facts."""

    def context(self, concept: Concept, **coeffs) -> AxbenchPromptSteerContext:
        if coeffs:
            raise ValueError(f"AxbenchPromptSteerModel takes no coefficients, got {coeffs}")
        return AxbenchPromptSteerContext(T_PROMPT_STEERING % concept.name)
