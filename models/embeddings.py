"""
models/embeddings.py
--------------------
Bi-encoder loader (frozen baseline + fine-tuned variant) and a thin
fine-tuning helper that uses MultipleNegativesRankingLoss.

Component (A) of the "train at least 3 of A-E" rubric lives here.
"""
from __future__ import annotations
import os
import random
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from models.quantization import quantize_sentence_transformer, cuda_available
from utils.device import get_device


def load_embedding_model(base_model: str, fine_tuned_path: str | None = None,
                         use_fine_tuned: bool = False,
                         dtype: str = "float32") -> SentenceTransformer:
    """
    Return a sentence-transformer, either frozen base or fine-tuned.

    Parameters
    ----------
    dtype : str
        "float32" (default on CPU), "float16" (GPU), "bfloat16" (GPU).
        int8 is not supported for SentenceTransformer — falls back to fp16.
    """
    # Construct on the best available device. Universal selection covers
    # CUDA (NVIDIA/AMD), MPS (Apple Silicon) and CPU. Called at runtime so
    # the selection reflects the real hardware state — sentence-transformers'
    # .encode() path infers device from the underlying nn.Module's location,
    # so the model must actually live on the right device or the GPU shows
    # zero load while encoding takes minutes.
    target_device = get_device()
    target_device_str = str(target_device)
    if use_fine_tuned and fine_tuned_path and os.path.isdir(fine_tuned_path):
        model = SentenceTransformer(fine_tuned_path, device=target_device_str)
    else:
        model = SentenceTransformer(base_model, device=target_device_str)
    # Apply precision AFTER loading — sentence-transformers doesn't accept
    # a dtype kwarg at construction time.
    model = quantize_sentence_transformer(model, dtype)
    # Belt-and-braces: explicitly move to the target device after dtype
    # casting (some quantize paths reset device on certain torch builds).
    try:
        model = model.to(target_device)
    except Exception:
        pass
    return model


def fine_tune_bi_encoder(base_model: str, output_path: str,
                         train_chunks: List[Dict[str, Any]],
                         doc_to_idx: Dict[str, List[int]],
                         n_examples: int = 250,
                         epochs: int = 3,
                         batch_size: int = 16,
                         lr: float = 2e-5,
                         seed: int = 42) -> SentenceTransformer:
    """
    Fine-tune MiniLM with MultipleNegativesRankingLoss.

    MNRL treats every other example in a batch as an in-batch negative, so
    we only need (query, positive) pairs. Queries are the first 12 words of
    the positive chunk (a common weak-supervision trick for RAG training).
    """
    rnd = random.Random(seed)
    doc_ids = list(doc_to_idx.keys())
    # BUG FIX (auditor #3): previously we used doc_ids[:n_examples], which
    # always took the FIRST n docs in dict-insertion order. If the dataset
    # is ordered in any meaningful way (alphabetical, source, domain),
    # this biases training. Now we sample uniformly at random with a fixed
    # seed — reproducible and unbiased.
    if n_examples < len(doc_ids):
        sampled_ids = rnd.sample(doc_ids, n_examples)
    else:
        sampled_ids = list(doc_ids)
    examples: List[InputExample] = []
    for doc_id in sampled_ids:
        pos_idx = rnd.choice(doc_to_idx[doc_id])
        query   = " ".join(train_chunks[pos_idx]["text"].split()[:12])
        pos_txt = train_chunks[pos_idx]["text"]
        examples.append(InputExample(texts=[query, pos_txt]))

    model  = SentenceTransformer(base_model)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss   = losses.MultipleNegativesRankingLoss(model)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=10,
        optimizer_params={"lr": lr},
        show_progress_bar=False,
        output_path=output_path,
    )
    return model
