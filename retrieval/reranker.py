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

import numpy as np
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
        target_device = "cuda" if cuda_available() else "cpu"
        try:
            self.model = CrossEncoder(model_name, max_length=max_length,
                                       device=target_device)
        except TypeError:
            self.model = CrossEncoder(model_name, max_length=max_length)
        self.model = quantize_sentence_transformer(self.model, dtype)
        try:
            if hasattr(self.model, "model") and hasattr(self.model.model, "to"):
                self.model.model = self.model.model.to(target_device)
        except Exception:
            pass

    def rerank(self, query: str, chunks: List[Dict[str, Any]],
               top_n: int = 5) -> List[Dict[str, Any]]:
        if not chunks:
            return chunks

        # FIX: newer sentence-transformers expects list[list[str]] (pairs)
        # *not* list[tuple]. Passing tuples makes predict() wrap the input
        # into a BatchEncoding whose __getitem__ then refuses the tensor
        # 2-D slice `[:, [-1, 0]]` inside BERT's padding-check, raising
        #   TypeError: list indices must be integers or slices, not tuple
        pairs: List[List[str]] = [[query, c["text"]] for c in chunks]

        # Request numpy output explicitly so the return type is stable
        # across sentence-transformers versions. `predict()` on a single
        # pair returns a 0-d array; `.reshape(-1)` normalises it to 1-d.
        scores = self.model.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        # Defensive: the number of returned scores MUST equal len(chunks).
        # If it doesn't, we have a silent tokenizer truncation bug upstream
        # and we want to know about it NOW instead of mis-attributing scores.
        assert scores.shape[0] == len(chunks), (
            f"reranker returned {scores.shape[0]} scores "
            f"for {len(chunks)} chunks"
        )

        for c, s in zip(chunks, scores):
            c["ce_score"] = float(s)
        return sorted(chunks, key=lambda x: x["ce_score"], reverse=True)[:top_n]