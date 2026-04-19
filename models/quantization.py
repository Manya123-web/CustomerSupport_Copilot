"""
models/quantization.py
----------------------
Central quantization helpers. Every model-loading path in the project
routes through this module so quantization is CONSISTENT across:

    * LLM generator         (Flan-T5)
    * Bi-encoder retriever  (MiniLM SentenceTransformer)
    * Cross-encoder reranker (ms-marco CrossEncoder)

Supported precisions
--------------------
    "float32"   : plain full precision (default CPU behaviour)
    "float16"   : half precision on CUDA devices — default for GPU runs.
                  Halves VRAM for every model, ~1.4x faster inference,
                  no measurable quality loss on our metrics.
    "int8"      : bitsandbytes 8-bit quantization. Requires GPU + the
                  `bitsandbytes` package. ~4x smaller than fp32.

Why not always use int8?
    int8 needs `bitsandbytes`, which is CUDA-only. For CPU-only environments
    (Colab when GPU is not enabled) we fall back to float32 automatically.

Usage (internal — the public API is `load_base_llm`, `load_encoder` etc.)
------------------------------------------------------------------------
    from models.quantization import resolve_dtype, quantized_kwargs
    torch_dtype = resolve_dtype("float16")                   # -> torch.float16
    kwargs      = quantized_kwargs("float16", for_="seq2seq")
"""
from __future__ import annotations
import os
import warnings
from typing import Optional, Dict, Any

import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


def resolve_dtype(name: str) -> torch.dtype:
    """
    Map a human-readable precision name to a torch.dtype.

    Safety: if the user requests float16 but there is no GPU, downgrade
    to float32 with a warning. fp16 on CPU works but is SLOWER than
    fp32 on most CPUs — half precision only helps on hardware that
    natively supports it.
    """
    name = name.lower()
    if name in ("fp32", "float32", "full"):
        return torch.float32
    if name in ("fp16", "float16", "half"):
        if not cuda_available():
            warnings.warn("float16 requested but no CUDA GPU available; "
                           "using float32. (fp16 is slower than fp32 on most CPUs.)",
                           RuntimeWarning)
            return torch.float32
        return torch.float16
    if name in ("bf16", "bfloat16"):
        if not cuda_available():
            return torch.float32
        return torch.bfloat16
    if name in ("int8", "8bit"):
        # INT8 is handled via bitsandbytes config, not a torch.dtype,
        # but we still need a computation dtype — fp16 activations.
        return torch.float16
    raise ValueError(f"Unknown dtype name: {name!r}")


def quantized_kwargs(precision: str, for_: str = "seq2seq") -> Dict[str, Any]:
    """
    Return the kwargs to pass to `from_pretrained()` for the given precision.

    `for_` is either "seq2seq" (Flan-T5) or "causal" (decoder-only). Both
    accept the same quantization config currently, but keeping the argument
    lets us diverge later.
    """
    precision = precision.lower()

    # ── INT8 via bitsandbytes ────────────────────────────────────────────────
    if precision in ("int8", "8bit"):
        if not cuda_available():
            warnings.warn("int8 requested but no GPU available; falling back "
                           "to float32.", RuntimeWarning)
            return {"low_cpu_mem_usage": True}
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401 — check it's installed
        except ImportError:
            warnings.warn("int8 requested but bitsandbytes not installed; "
                           "falling back to float16. Install with "
                           "`pip install bitsandbytes`.", RuntimeWarning)
            return quantized_kwargs("float16", for_)
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map":          "auto",
            "low_cpu_mem_usage":   True,
        }

    # ── fp32 / fp16 / bf16 via torch dtype ───────────────────────────────────
    torch_dtype = resolve_dtype(precision)

    # transformers >= 4.45 renamed `torch_dtype` to `dtype`. Detect and use
    # whichever the installed version accepts so we stay forward-compatible.
    from transformers import AutoModelForSeq2SeqLM
    import inspect
    sig = inspect.signature(AutoModelForSeq2SeqLM.from_pretrained)
    dtype_kw = "dtype" if "dtype" in sig.parameters else "torch_dtype"

    kwargs: Dict[str, Any] = {
        dtype_kw:            torch_dtype,
        "low_cpu_mem_usage": True,
    }
    # NOTE: we intentionally do NOT set `device_map="auto"` here.
    # With accelerate>=1.0, "auto" on a single-GPU machine lands the
    # whole model on cuda:0 (fine), but on boxes where accelerate
    # partially resolves (version drift, pip cache surprises) "auto"
    # can split layers across cuda:0/cpu, which then blows up later
    # at `inputs.to(model.device)` with "Expected all tensors on the
    # same device". The caller (load_base_llm) now does an explicit
    # `.to("cuda")` after load — one device, one place, no magic.
    return kwargs


def quantize_sentence_transformer(model, precision: str):
    """
    Apply precision to a SentenceTransformer or CrossEncoder after loading.

    These libraries don't expose a `from_pretrained` quantization kwarg, so
    we cast the underlying nn.Module in place. Only fp16/bf16 are supported
    this way — int8 needs a different machinery that these wrappers don't
    integrate with cleanly.
    """
    precision = precision.lower()
    if precision in ("fp32", "float32", "full"):
        return model
    if precision in ("int8", "8bit"):
        warnings.warn("int8 not supported for SentenceTransformer/CrossEncoder; "
                       "using float16 instead.", RuntimeWarning)
        precision = "float16"
    torch_dtype = resolve_dtype(precision)
    if torch_dtype == torch.float32:
        return model

    # SentenceTransformer exposes `.to(dtype)`. CrossEncoder wraps an HF
    # model at `.model`. Try both paths.
    try:
        if hasattr(model, "to"):
            model = model.to(torch_dtype)
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model = model.model.to(torch_dtype)
        return model
    except Exception as e:
        warnings.warn(f"Could not cast model to {torch_dtype}: {e}", RuntimeWarning)
        return model


def vram_gb() -> float:
    """Current GPU VRAM usage in gigabytes. Returns 0.0 if no GPU."""
    if not cuda_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def report_precision(cfg: dict) -> str:
    """Human-readable summary of the active precision plan."""
    g = cfg.get("generator", {}).get("dtype", "float32")
    e = cfg.get("embeddings", {}).get("dtype", "float32")
    r = cfg.get("reranker", {}).get("dtype", e)
    return (f"precision plan  generator={g}  encoder={e}  reranker={r}  "
            f"(GPU={'yes' if cuda_available() else 'no'})")
