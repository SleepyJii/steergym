from dataclasses import dataclass

from .steer import GeneratorRef


@dataclass
class GenParams:
    max_new_tokens: int = 128
    temperature: float = 1.0
    do_sample: bool = True
    batch_size: int = 8
    seed: int | None = None
    # None = not passed to model.generate at all, so the model's own
    # generation-config defaults apply and pre-existing call sites are
    # bit-unchanged.
    top_p: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None


class GeneratorLM:
    """The loaded generator LLM (the steered subject) + tokenizer, and the
    chat-templating seam
    that pins the axbench protocol: NO system prompt, BOS
    stripped from the stored templated string then re-added once at
    tokenization (add_special_tokens default)."""

    def __init__(self, ref: GeneratorRef, model, tokenizer):
        self.ref = ref
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, ref: GeneratorRef, dtype: str = "bfloat16", device_map: str = "auto") -> "GeneratorLM":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            ref.model_id, revision=ref.revision,
            torch_dtype=getattr(torch, dtype), device_map=device_map,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(ref.model_id, revision=ref.revision)
        if tokenizer.pad_token is None:
            # base checkpoints ship no pad token and generate() left-pads: eos-as-pad
            # (safe — decode slices the prompt off; config line inert, parity-of-record)
            if tokenizer.eos_token is None:
                raise ValueError(f"{ref.model_id}: tokenizer has neither pad "
                                 "nor eos token — cannot left-pad batches")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        return cls(ref, model, tokenizer)

    def chat_prompt(self, text: str) -> str:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        bos = self.tokenizer.bos_token
        if bos and rendered.startswith(bos):
            rendered = rendered[len(bos):]
        return rendered

    def generate(self, prompts: list[str], gen: GenParams) -> list[str]:
        import torch

        self.tokenizer.padding_side = "left"
        extra = {k: v for k, v in [("top_p", gen.top_p),
                                   ("repetition_penalty", gen.repetition_penalty),
                                   ("no_repeat_ngram_size", gen.no_repeat_ngram_size)]
                 if v is not None}
        responses = []
        with torch.no_grad():
            for i in range(0, len(prompts), gen.batch_size):
                chunk = prompts[i:i + gen.batch_size]
                enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                out = self.model.generate(
                    **enc, max_new_tokens=gen.max_new_tokens,
                    do_sample=gen.do_sample, temperature=gen.temperature, **extra)
                prompt_len = enc["input_ids"].shape[1]
                decoded = self.tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
                responses.extend(decoded)
        return responses
