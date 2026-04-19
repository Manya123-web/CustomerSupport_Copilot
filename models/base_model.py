"""
models/base_model.py
--------------------
Loads Flan-T5-large through the central `models/quantization.py` helper
so precision is consistent with the bi-encoder and cross-encoder. Also
attaches saved DoRA adapters if present.
"""
from __future__ import annotations
import os
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models.quantization import quantized_kwargs, cuda_available, vram_gb


def load_base_llm(model_id: str, dtype: str = "float16") -> Tuple:
    """
    Load the generator LLM with the requested precision.

    Supported `dtype` values (strings, case-insensitive):
        "float32"   — full precision
        "float16"   — half precision (GPU only; default on GPU runs)
        "bfloat16"  — bf16 (GPU only)
        "int8"      — bitsandbytes 8-bit (GPU + bitsandbytes installed)

    If the requested precision is not available on the current hardware
    (e.g. fp16 without a GPU), the helper downgrades with a warning.
    """
    torch.cuda.empty_cache() if cuda_available() else None
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = quantized_kwargs(dtype, for_="seq2seq")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if cuda_available():
        print(f"[base_model] {model_id}  precision={dtype}  VRAM={vram_gb():.2f}GB")
    return model, tokenizer


def attach_dora_adapters(model, tokenizer, dora_path: str, strict: bool = True):
    """
    Attach saved DoRA adapters if present. If `strict=True` and the
    directory is missing, raise — no silent fall-back to base model
    (which invalidated every downstream claim in earlier versions).
    """
    from peft import PeftModel
    adapter_cfg = os.path.join(dora_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        msg = (f"DoRA adapters not found at {dora_path!r}. "
               "Run `python -m training.train --stage dora` first.")
        if strict:
            raise FileNotFoundError(msg)
        print(f"[base_model] WARNING: {msg}  -- continuing with base model")
        return model, tokenizer, False

    model = PeftModel.from_pretrained(model, dora_path, is_trainable=False)
    model.eval()
    tok_cfg = os.path.join(dora_path, "tokenizer_config.json")
    if os.path.exists(tok_cfg):
        tokenizer = AutoTokenizer.from_pretrained(dora_path)
    return model, tokenizer, True


def make_llm_fn(model, tokenizer, max_new_tokens: int = 250, num_beams: int = 2):
    """Return a closure that runs the LLM (used by retrieval/agent code)."""
    def _llm(prompt: str, max_new: int | None = None):
        mnt = max_new or max_new_tokens
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=mnt,
                                  do_sample=False, num_beams=num_beams)
        return [{"generated_text": tokenizer.decode(out[0], skip_special_tokens=True)}]
    return _llm
