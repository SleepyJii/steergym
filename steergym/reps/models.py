import math
from dataclasses import dataclass

from ..axbench import AxbenchRowConfig, AxbenchRowModel


@dataclass
class LsReFTConfig(AxbenchRowConfig):
    # config of record: ab_config.yaml LsReFT block
    lr: float = 0.01
    weight_decay: float = 0.0
    n_epochs: int = 3
    batch_size: int = 6
    gradient_accumulation: int = 1
    topk: int = 8
    coeff_latent_l1_loss: float = 0.005
    max_length: int = 1024
    seed: int = 42


# ---------------------------------------------------------------- pure math (torch-free, tested)
def topk_mean_and_l1(latent_values: list[float], k: int) -> tuple[float, float]:
    """axbench interventions.TopKReLUIntervention forward + model_utils
    calculate_l1_losses' non_topk branch, for ONE sequence's intervened-
    position latents (already relu'd): (mean of the top-k values, mean of
    the non-top-k values with the top-k positions zeroed)."""
    n = len(latent_values)
    if n == 0:
        return 0.0, 0.0
    k = min(k, n)
    order = sorted(range(n), key=lambda i: latent_values[i], reverse=True)
    topk_idx = set(order[:k])
    topk_mean = sum(latent_values[i] for i in topk_idx) / k
    non_topk = [0.0 if i in topk_idx else latent_values[i] for i in range(n)]
    l1_mean = sum(non_topk) / n
    return topk_mean, l1_mean


def _fit_lsreft_rows(generator, concepts, samples, config, log=lambda msg: None):
    """axbench lsreft.py LsReFT.train, replicated with a plain forward hook
    (TopKReLUIntervention math applied at the intervened positions of the
    full hidden state) instead of pyvene's gathered sub-tensor. Rank-1 v
    with a constant-0.01 weight init and a dead (never-forwarded) bias,
    unit-norm + gradient-projection each step, linear-decay AdamW, a
    ramping top-k L1 penalty, HF's built-in causal-LM CE for the LM loss."""
    import time

    import torch
    import torch.nn as nn
    from transformers import set_seed

    from ..axbench.protocol import chat_prompt_text, positive_and_negative_rows, prefix_suffix_lengths
    from ..training_helpers import hidden_size, linear_decay_lr, lm_train_batches, tokenize_row

    set_seed(config.seed)
    tokenizer = generator.tokenizer
    tokenizer.padding_side = "right"
    prefix_length, _ = prefix_suffix_lengths(tokenizer)
    layer_module = generator.model.model.layers[config.layer]

    rows, biases = [], []
    for concept in concepts:
        t0 = time.time()
        positive, negative = positive_and_negative_rows(samples, concept)

        rows_data = []
        for src in (positive, negative):
            for row in src.iter_rows(named=True):
                prompt = chat_prompt_text(tokenizer, row["prompt"])
                # keep_orig_axbench_format=true in the config of record: no EOS/turn
                # suffix is appended to outputs (train.py:184-193 skips it)
                output = row["response"]
                rows_data.append(tokenize_row(tokenizer, prompt, output, config.max_length, torch))

        v = nn.Linear(hidden_size(generator.model, layer_module), 1, bias=True).to(
            device=generator.model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            v.weight.fill_(0.01)
            v.bias.fill_(0.0)
        v.bias.requires_grad_(False)  # never forwarded -> never receives a gradient (save zeros)

        optimizer = torch.optim.AdamW([v.weight], lr=config.lr, weight_decay=config.weight_decay)
        batches = lm_train_batches(tokenizer, rows_data, prefix_length, config.max_length,
                                    config.batch_size, config.seed)
        total_steps = config.n_epochs * len(batches)
        curr_step = 0

        for epoch in range(config.n_epochs):
            for batch in batches:
                v_row = v.weight[0]  # [d], bf16

                captured = {}

                def hook(mod, args, output, captured=captured):
                    # topk_mean_and_l1 gives the tested pure-python form of this identical
                    # per-row computation; here it must stay differentiable (grad -> v_row),
                    # so the topk/mask logic is redone with torch ops on a per-row slice --
                    # matching axbench's own in-place scatter_ on a clone (also autograd-safe).
                    h = output[0] if isinstance(output, tuple) else output
                    interv_mask = batch["interv_mask"].to(h.device)
                    latent_full = torch.relu(h @ v_row.to(device=h.device, dtype=h.dtype))  # [B, T]

                    mag_terms, l1_terms = [], []
                    for b in range(h.shape[0]):
                        pos = interv_mask[b].nonzero(as_tuple=True)[0]
                        n = pos.numel()
                        if n == 0:
                            mag_terms.append(latent_full.new_zeros(()))
                            l1_terms.append(latent_full.new_zeros(()))
                            continue
                        row_latent = latent_full[b, pos]  # [n], differentiable
                        k = min(config.topk, n)
                        topk_vals, topk_idx = row_latent.topk(k)
                        mag_terms.append(topk_vals.mean())
                        non_topk = row_latent.clone()
                        non_topk[topk_idx] = row_latent.new_zeros(k)
                        l1_terms.append(non_topk.mean())
                    mag = torch.stack(mag_terms).to(h.dtype)
                    captured["l1_mean"] = torch.stack(l1_terms).mean()

                    delta = interv_mask.unsqueeze(-1).to(h.dtype) * mag.view(-1, 1, 1) * v_row.to(device=h.device, dtype=h.dtype)
                    new_h = h + delta
                    return (new_h,) + tuple(output[1:]) if isinstance(output, tuple) else new_h

                handle = layer_module.register_forward_hook(hook)
                outputs = generator.model(
                    input_ids=batch["input_ids"].to(generator.model.device),
                    attention_mask=batch["attention_mask"].to(generator.model.device),
                    labels=batch["labels"].to(generator.model.device))
                handle.remove()

                coeff = curr_step / total_steps if total_steps else 0.0
                loss = outputs.loss + coeff * config.coeff_latent_l1_loss * captured["l1_mean"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([v.weight], 1.0)
                with torch.no_grad():
                    eps = torch.finfo(v.weight.dtype).eps
                    v.weight.data /= v.weight.data.norm(dim=1, keepdim=True) + eps
                    parallel = (v.weight.grad * v.weight.data).sum(dim=1, keepdim=True)
                    v.weight.grad -= parallel * v.weight.data
                for g in optimizer.param_groups:
                    g["lr"] = linear_decay_lr(config.lr, curr_step, total_steps)
                optimizer.step()
                curr_step += 1

        rows.append(v.weight.data[0].float().clone())
        biases.append(0.0)
        log(f"LsReFT concept {concept.id} ({concept.name!r}): "
            f"{len(rows_data)} rows, {total_steps} steps, {time.time() - t0:.1f}s")

    return torch.stack(rows), torch.tensor(biases, dtype=torch.float32)


class LsReFTModel(AxbenchRowModel):
    """ReFT-r1 steering path: addition of unit rows (bias/topk are latent-only,
    unused here). Released dictionary: pyvene/gemma-reft-r1-2b-it-res."""

    weight_name = "LsReFT"
    config_cls = LsReFTConfig

    def fit(self, concepts, samples):
        from ..generator import GeneratorLM

        model, tokenizer = self._require_bound()
        generator = GeneratorLM(self.config.subject, model, tokenizer)
        rows, biases = _fit_lsreft_rows(generator, concepts, samples, self.config)
        self.rows = rows
        self.bias = biases
        self.concept_index = {c.id: i for i, c in enumerate(concepts)}
        self.raw_norms = None
        return self


@dataclass
class PreferenceVectorConfig(AxbenchRowConfig):
    # config of record for the RePS training runs. n_epochs=18 is the canonical
    # training length; 8 is the faster "sanity" length used for smoke runs.
    lr: float = 0.04
    n_epochs: int = 18
    batch_size: int = 6
    steering_factors: tuple = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
    preference_pairs: tuple = ("orig_add", "orig_sub")
    substraction_type: str = "null_it_out"
    max_length: int = 512
    seed: int = 42


def scaled_simpo_terms(ref_lp_w: float, ref_lp_l: float, pol_lp_w: float, pol_lp_l: float,
                       len_w: float, len_l: float) -> float:
    """axbench preference_model.preference_loss's scaled_simpo branch
    (simpo_scaler fixed at 1.0 -- dead knob in the config of record), as a
    pure scalar function: -logsigmoid(scaled_chosen - scaled_rejected)
    computed via a numerically stable softplus(-x)."""
    ref_logratios_reverse = ref_lp_l - ref_lp_w
    scaled_chosen = (max(ref_logratios_reverse, 1.0) / len_w) * pol_lp_w
    scaled_rejected = (1.0 / len_l) * pol_lp_l
    x = scaled_chosen - scaled_rejected
    if x >= 0:
        return math.log1p(math.exp(-x))
    return -x + math.log1p(math.exp(x))


def _fit_preference_rows(generator, concepts, polar_samples, config, log=lambda msg: None):
    """axbench preference_model.py PreferenceModel.train (PreferenceVectorIntervention
    variant), replicated with plain forward hooks instead of pyvene's dual
    ref/policy forward. Positives-only polar pairs (winning=concept text,
    losing=the subject's own unsteered continuation); two arms per row
    (orig_add, orig_sub / null_it_out); scaled_simpo preference loss; NO
    unit-norm constraint -- the saved row (and its trained bias) are exactly
    the AdamW-optimized nn.Linear(hd, 1)."""
    import random as pyrandom
    import time

    import polars as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import set_seed

    from ..axbench.protocol import chat_prompt_text, prefix_suffix_lengths
    from ..training_helpers import hidden_size, linear_decay_lr, pad_stack

    set_seed(config.seed)
    tokenizer = generator.tokenizer
    tokenizer.padding_side = "right"
    prefix_length, _ = prefix_suffix_lengths(tokenizer)
    layer_module = generator.model.model.layers[config.layer]
    pairs = list(config.preference_pairs)

    def _side_tokenize(prompt, text):
        prompt_ids = tokenizer(prompt, max_length=config.max_length, truncation=True)["input_ids"]
        full_ids = tokenizer(prompt + text, max_length=config.max_length, truncation=True)["input_ids"]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "prompt_length": len(prompt_ids),
        }

    def _forward_batch(examples, v, bias_param, hook_on, steering_factors):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids = pad_stack([e["input_ids"] for e in examples], pad_id, torch.long, torch)
        lengths = torch.tensor([len(e["input_ids"]) for e in examples], dtype=torch.long)
        T = input_ids.shape[1]
        attention_mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)).long()
        labels = torch.full_like(input_ids, -100)
        for b, e in enumerate(examples):
            seq_len = len(e["input_ids"])
            labels[b, e["prompt_length"]:seq_len] = input_ids[b, e["prompt_length"]:seq_len]
        pos = torch.arange(T).unsqueeze(0)
        interv_mask = (pos >= prefix_length) & (pos < lengths.unsqueeze(1))

        handle = None
        if hook_on:
            factors = torch.tensor(steering_factors, dtype=torch.bfloat16)

            def hook(mod, args, output):
                h = output[0] if isinstance(output, tuple) else output
                v_row = v.weight[0].to(device=h.device, dtype=h.dtype)
                bias = bias_param.to(device=h.device, dtype=h.dtype)
                v_norm_sq = (v_row * v_row).sum().clamp_min(torch.finfo(h.dtype).eps)
                latent = torch.relu((h @ v_row) + bias)  # [B, T]
                sf = factors.to(h.device).view(-1, 1)
                zero_mask = (sf == 0.0).to(h.dtype)
                nonzero_mask = 1.0 - zero_mask
                null_term = -(latent / v_norm_sq) * zero_mask
                add_term = sf.to(h.dtype) + bias * nonzero_mask
                combined = null_term + add_term  # [B, T]
                imask = interv_mask.to(h.device).to(h.dtype)
                delta = (combined * imask).unsqueeze(-1) * v_row
                new_h = h + delta
                return (new_h,) + tuple(output[1:]) if isinstance(output, tuple) else new_h

            handle = layer_module.register_forward_hook(hook)

        ctx = torch.enable_grad() if hook_on else torch.no_grad()
        with ctx:
            outputs = generator.model(
                input_ids=input_ids.to(generator.model.device),
                attention_mask=attention_mask.to(generator.model.device))
        if handle is not None:
            handle.remove()

        logits = outputs.logits
        shifted_labels = labels[:, 1:].to(logits.device)
        shifted_logits = logits[:, :-1, :]
        mask = shifted_labels != -100
        safe_labels = shifted_labels.clone()
        safe_labels[~mask] = 0
        logp = torch.gather(shifted_logits.log_softmax(-1), 2, safe_labels.unsqueeze(-1)).squeeze(-1)
        logps = (logp * mask).sum(-1)
        return logps.float(), attention_mask.sum(-1).float().to(logps.device)

    rows, biases = [], []
    for concept in concepts:
        t0 = time.time()
        concept_rows = polar_samples.filter(pl.col("concept_id") == concept.id)

        row_pairs = []
        for r in concept_rows.iter_rows(named=True):
            for pair in pairs:
                if pair == "orig_add":
                    win_text, los_text = r["winning"], r["losing"]
                elif pair == "orig_sub":
                    win_text, los_text = r["losing"], r["winning"]
                else:
                    raise NotImplementedError(f"preference pair {pair!r} not implemented")
                row_pairs.append({
                    "pair": pair,
                    "winning": _side_tokenize(chat_prompt_text(tokenizer, r["prompt"]), win_text),
                    "losing": _side_tokenize(chat_prompt_text(tokenizer, r["prompt"]), los_text),
                })

        v = nn.Linear(hidden_size(generator.model, layer_module), 1, bias=True).to(
            torch.bfloat16)  # default kaiming-uniform init (CPU RNG)
        v = v.to(generator.model.device)
        with torch.no_grad():
            v.bias.fill_(0.0)

        optimizer = torch.optim.AdamW(v.parameters(), lr=config.lr, weight_decay=0.0)
        n_outer = concept_rows.height
        total_steps = config.n_epochs * math.ceil(n_outer / config.batch_size) if n_outer else 0
        curr_step = 0
        gen = torch.Generator().manual_seed(config.seed)  # fresh per concept, persists across its epochs

        for epoch in range(config.n_epochs):
            order = torch.randperm(n_outer, generator=gen).tolist() if n_outer else []
            for i in range(0, n_outer, config.batch_size):
                outer_idx = order[i:i + config.batch_size]
                if not outer_idx:
                    continue

                winning_examples, losing_examples, winning_factors, losing_factors = [], [], [], []
                for oi in outer_idx:
                    for pair in pairs:
                        rp = row_pairs[oi * len(pairs) + pairs.index(pair)]
                        winning_examples.append(rp["winning"])
                        losing_examples.append(rp["losing"])
                        if "_add" in pair:
                            winning_factors.append(pyrandom.choice(config.steering_factors))
                            losing_factors.append(pyrandom.choice(config.steering_factors))
                        else:
                            winning_factors.append(0.0)
                            losing_factors.append(0.0)

                expanded = len(winning_examples)
                minibatch_size = config.batch_size
                num_minibatches = math.ceil(expanded / minibatch_size)

                optimizer.zero_grad()
                for mb in range(num_minibatches):
                    s, e = mb * minibatch_size, min((mb + 1) * minibatch_size, expanded)
                    if s >= expanded:
                        break
                    mb_win, mb_los = winning_examples[s:e], losing_examples[s:e]
                    mb_win_f, mb_los_f = winning_factors[s:e], losing_factors[s:e]

                    ref_win_lp, win_lens = _forward_batch(mb_win, v, v.bias, False, mb_win_f)
                    ref_los_lp, los_lens = _forward_batch(mb_los, v, v.bias, False, mb_los_f)
                    pol_win_lp, _ = _forward_batch(mb_win, v, v.bias, True, mb_win_f)
                    pol_los_lp, _ = _forward_batch(mb_los, v, v.bias, True, mb_los_f)

                    # scaled_simpo (see scaled_simpo_terms for the pure/tested scalar form of
                    # this identical formula; simpo_scaler fixed at 1.0, per the config of record)
                    ref_logratios_reverse = ref_los_lp - ref_win_lp
                    scaled_chosen = (torch.clamp(ref_logratios_reverse, min=1.0) / win_lens) * pol_win_lp
                    scaled_rejected = (1.0 / los_lens) * pol_los_lp
                    mb_loss = F.softplus(-(scaled_chosen - scaled_rejected)).mean()
                    (mb_loss / num_minibatches).backward()

                torch.nn.utils.clip_grad_norm_(v.parameters(), 1.0)
                for g in optimizer.param_groups:
                    g["lr"] = linear_decay_lr(config.lr, curr_step, total_steps)
                optimizer.step()
                curr_step += 1

        rows.append(v.weight.data[0].float().clone())
        biases.append(float(v.bias.data[0]))
        log(f"PreferenceVector concept {concept.id} ({concept.name!r}): "
            f"{len(row_pairs)} pair-examples, {total_steps} steps, {time.time() - t0:.1f}s")

    return torch.stack(rows), torch.tensor(biases, dtype=torch.float32)


class PreferenceVectorModel(AxbenchRowModel):
    """RePS rank-1 vectors (Wu et al. 2025). No released artifacts exist;
    rows come from training dumps (config.dump_dir) or from native fit()."""

    weight_name = "PreferenceVector"
    config_cls = PreferenceVectorConfig

    def fit(self, concepts, polar_samples):
        from ..generator import GeneratorLM

        model, tokenizer = self._require_bound()
        generator = GeneratorLM(self.config.subject, model, tokenizer)
        rows, biases = _fit_preference_rows(generator, concepts, polar_samples, self.config)
        self.rows = rows
        self.bias = biases
        self.concept_index = {c.id: i for i, c in enumerate(concepts)}
        self.raw_norms = None
        return self
