"""
retrieval/reranker.py
---------------------
Cross-encoder reranker (Component B). The bi-encoder returns top-K candidates
fast; the cross-encoder looks at (query, chunk) jointly and produces a more
accurate ordering. We emit `ce_score` back onto each chunk dict so the CGRA
gate can read post-rerank confidence.
"""
from __future__ import annotations
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from models.quantization import quantize_sentence_transformer, cuda_available


class CEReranker:
    def __init__(self, model_name: str, max_length: int = 512,
                 dtype: str = "float32"):
        """
        Parameters
        ----------
        dtype : str
            Precision for the cross-encoder weights. "float16" on GPU
            halves VRAM and is ~1.4x faster; "float32" on CPU (no fp16
            hardware support).
        """
        # CrossEncoder accepts a `device` kwarg in sentence-transformers
        # >=2.6; passing it explicitly avoids the silent CPU fallback that
        # happens when the wrapped HF model isn't moved to the active GPU.
        target_device = "cuda" if cuda_available() else "cpu"
        try:
            self.model = CrossEncoder(model_name, max_length=max_length,
                                       device=target_device)
        except TypeError:
            # Older sentence-transformers signatures don't take `device`
            self.model = CrossEncoder(model_name, max_length=max_length)
        # Apply precision after construction (cast happens in place)
        self.model = quantize_sentence_transformer(self.model, dtype)
        # Make sure the underlying HF model lives on the target device
        # AFTER the dtype cast — quantize_sentence_transformer only casts
        # dtype, never device, so on GPU machines we'd otherwise silently
        # run inference on CPU.
        try:
            if hasattr(self.model, "model") and hasattr(self.model.model, "to"):
                self.model.model = self.model.model.to(target_device)
        except Exception:
            pass

    def rerank(self, query: str, chunks: List[Dict[str, Any]],
               top_n: int = 5) -> List[Dict[str, Any]]:
        if not chunks:
            return chunks
        scores = self.model.predict([(query, c["text"]) for c in chunks])
        for c, s in zip(chunks, scores):
            c["ce_score"] = float(s)
        return sorted(chunks, key=lambda x: x["ce_score"], reverse=True)[:top_n]
