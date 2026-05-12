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
import torch
from sentence_transformers import CrossEncoder

from models.quantization import quantize_sentence_transformer, cuda_available
from utils.device import get_device


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
        target_device = get_device()       # runtime detection
        target_device_str = str(target_device)
        self.max_length = max_length      # stored for rerank() to read
        try:
            self.model = CrossEncoder(model_name, max_length=max_length,
                                       device=target_device_str)
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

        # We bypass CrossEncoder.predict() and call the underlying transformers
        # model directly. Reason: sentence-transformers >= 3.0 rewrote predict()
        # to pass a BatchEncoding straight into the HF model, then
        # transformers' warn_if_padding_and_no_attention_mask does
        # `input_ids[:, [-1, 0]]` which is a 2-D tensor slice. BatchEncoding's
        # __getitem__ refuses tuple keys and raises:
        #     TypeError: list indices must be integers or slices, not tuple
        # Tokenising ourselves and feeding tensors to the model sidesteps that
        # entirely AND keeps behaviour consistent across ST 2.x / 3.x / 4.x.
        pairs: List[List[str]] = [[query, c["text"]] for c in chunks]

        # CrossEncoder exposes its tokenizer at `.tokenizer` and the underlying
        # HF model at `.model`. We tokenise the (query, passage) pairs as
        # text-pair inputs (single forward pass) and read the scalar logit.
        tokenizer = self.model.tokenizer
        hf_model  = self.model.model
        device    = next(hf_model.parameters()).device

        # Tokenize in batches to keep peak memory bounded on CPU
        BATCH = 16
        all_scores: List[float] = []
        hf_model.eval()
        with torch.no_grad():
            for i in range(0, len(pairs), BATCH):
                batch = pairs[i:i + BATCH]
                enc = tokenizer(
                    [p[0] for p in batch],
                    [p[1] for p in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = hf_model(**enc).logits  # shape (B, 1) or (B,)
                logits = logits.detach().float().cpu().numpy().reshape(-1)
                all_scores.extend(logits.tolist())

        scores = np.asarray(all_scores, dtype=np.float32)

        # Defensive: number of scores must equal number of chunks. If not,
        # something silently dropped a pair (rare, but worth catching).
        assert scores.shape[0] == len(chunks), (
            f"reranker returned {scores.shape[0]} scores "
            f"for {len(chunks)} chunks"
        )

        for c, s in zip(chunks, scores):
            c["ce_score"] = float(s)
        return sorted(chunks, key=lambda x: x["ce_score"], reverse=True)[:top_n]