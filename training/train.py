"""
training/train.py
-----------------
Single training entry-point. Trains (A) bi-encoder, and (C/D) DoRA+DPO
for the generator. Writes artefacts into models/checkpoints/... so the
evaluation pipeline can pick them up.

Usage
-----
    python -m training.train --stage biencoder
    python -m training.train --stage dora
    python -m training.train --stage all
"""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from peft import LoraConfig, get_peft_model, TaskType

from utils.config import load_config
from models.embeddings import fine_tune_bi_encoder
from models.base_model import load_base_llm


# ── Shared data loader ───────────────────────────────────────────────────────
def _load_chunks(cfg):
    with open(cfg["data"]["train_split"]) as f:
        train = json.load(f)
    return train


def _doc_to_idx(chunks):
    from collections import defaultdict
    d = defaultdict(list)
    for i, c in enumerate(chunks):
        d[c["doc_id"]].append(i)
    return d


# ── Stage A : bi-encoder fine-tune (MNRL) ────────────────────────────────────
def train_biencoder(cfg):
    train = _load_chunks(cfg)
    d2i = _doc_to_idx(train)
    out = cfg["embeddings"]["fine_tuned_path"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[train] fine-tuning bi-encoder → {out}")
    fine_tune_bi_encoder(
        base_model=cfg["embeddings"]["base_model"],
        output_path=out,
        train_chunks=train,
        doc_to_idx=d2i,
        n_examples=250, epochs=3, batch_size=16, lr=2e-5,
        seed=cfg.get("seed", 42),
    )
    print(f"[train] bi-encoder saved.")


# ── Stage C+D : DoRA + DPO on the generator ──────────────────────────────────
def _build_preference_pairs(train, llm_fn, tokenizer, max_pairs: int = 40):
    """
    For each of `max_pairs` random chunks, generate one answer WITH a
    citation-priming prompt (chosen) and one WITHOUT (rejected).
    This gives us the (chosen, rejected) pairs that DPO needs.
    """
    rnd = random.Random(0)
    pairs = []
    picks = rnd.sample(range(len(train)), min(max_pairs, len(train)))
    for idx in picks:
        ctx = train[idx]["text"][:800]
        did = train[idx]["doc_id"]
        query = " ".join(train[idx]["text"].split()[:10])
        cite_prompt = (f"Answer using the context. Start with [doc_id: {did}].\n"
                       f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:")
        free_prompt = (f"Answer using the context.\n"
                       f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:")
        try:
            chosen   = llm_fn(cite_prompt)[0]["generated_text"].strip()
            rejected = llm_fn(free_prompt)[0]["generated_text"].strip()
        except Exception:
            continue
        if not chosen or not rejected or chosen == rejected:
            continue
        pairs.append({"cite_prompt": cite_prompt, "free_prompt": free_prompt,
                      "chosen": chosen, "rejected": rejected})
    return pairs


def train_dora(cfg):
    """Train DoRA adapters (single global set, not domain-specific)."""
    train = _load_chunks(cfg)
    model, tokenizer = load_base_llm(cfg["generator"]["base_model"],
                                     cfg["generator"]["dtype"])
    # Attach DoRA adapters
    dora_cfg = LoraConfig(
        task_type      = TaskType.SEQ_2_SEQ_LM,
        r              = cfg["dora"]["r"],
        lora_alpha     = cfg["dora"]["alpha"],
        lora_dropout   = cfg["dora"]["dropout"],
        use_dora       = True,
        target_modules = cfg["dora"]["target_modules"],
    )
    model = get_peft_model(model, dora_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing (trades compute for VRAM)
    if cfg["dpo"]["grad_ckpt"]:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "get_base_model"):
                model.get_base_model().gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        except Exception as e:
            print(f"[train] gradient_checkpointing_enable failed: {e}")

    from models.base_model import make_llm_fn
    llm_fn = make_llm_fn(model, tokenizer,
                         max_new_tokens=cfg["generator"]["max_new_tokens"],
                         num_beams=cfg["generator"]["num_beams"])

    print("[train] building preference pairs...")
    pairs = _build_preference_pairs(train, llm_fn, tokenizer)
    if not pairs:
        print("[train] no pairs — skipping DPO")
        return
    print(f"[train] {len(pairs)} preference pairs")

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad],
                     lr=cfg["dpo"]["lr"])
    beta = cfg["dpo"]["beta"]
    n_epochs = cfg["dpo"]["epochs"]

    def log_prob(prompt, response):
        inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512).to(model.device)
        tgt = tokenizer(response, return_tensors="pt", truncation=True,
                        max_length=128).to(model.device)
        # NOTE: no autocast. fp16 underflow on Flan-T5's DPO log-probs
        # drives the loss to NaN. Running without autocast keeps the model
        # in its loaded dtype (fp16 forward, fp32 loss) which is stable.
        out = model(**inp, labels=tgt["input_ids"])
        lp = -out.loss * tgt["input_ids"].shape[1]
        return lp

    # Hold out 20% of pairs as a val set so we can early-stop if it overfits
    random.Random(0).shuffle(pairs)
    n_val = max(2, len(pairs) // 5) if len(pairs) >= 10 else 0
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:] if n_val else pairs
    if n_val:
        print(f"[train] split: {len(train_pairs)} train / {len(val_pairs)} val pairs")
    else:
        print(f"[train] {len(train_pairs)} train pairs (too few for a val split)")

    def _loss_over(ps):
        """Compute mean DPO loss on a list of pairs without updating weights."""
        if not ps:
            return float("inf")
        tot = 0.0
        with torch.no_grad():
            for p in ps:
                lp_w = log_prob(p["cite_prompt"], p["chosen"])
                lp_l = log_prob(p["free_prompt"], p["rejected"])
                with model.disable_adapter():
                    lp_w_ref = log_prob(p["cite_prompt"], p["chosen"])
                    lp_l_ref = log_prob(p["free_prompt"], p["rejected"])
                rw = beta * (lp_w - lp_w_ref)
                rl = beta * (lp_l - lp_l_ref)
                tot += (-F.logsigmoid(rw - rl)).item()
        return tot / len(ps)

    model.train()
    best_val   = float("inf")
    patience   = 2
    bad_epochs = 0
    for epoch in range(n_epochs):
        total = 0.0
        random.shuffle(train_pairs)
        for p in train_pairs:
            lp_w = log_prob(p["cite_prompt"], p["chosen"])
            lp_l = log_prob(p["free_prompt"], p["rejected"])
            with model.disable_adapter():
                lp_w_ref = log_prob(p["cite_prompt"], p["chosen"]).detach()
                lp_l_ref = log_prob(p["free_prompt"], p["rejected"]).detach()
            rw = beta * (lp_w - lp_w_ref)
            rl = beta * (lp_l - lp_l_ref)
            loss = -F.logsigmoid(rw - rl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        train_loss = total / len(train_pairs)

        if val_pairs:
            model.eval()
            val_loss = _loss_over(val_pairs)
            model.train()
            print(f"[train] epoch {epoch+1}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            # Early stopping: if val stops improving for `patience` epochs, stop
            if val_loss < best_val - 1e-3:
                best_val = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[train] early stop: val_loss hasn't improved for "
                          f"{patience} epochs (best={best_val:.4f})")
                    break
        else:
            print(f"[train] epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}")

    model.config.use_cache = True
    model.eval()

    out = cfg["generator"]["dora_path"]
    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[train] DoRA + tokenizer saved → {out}")


def train_router(cfg):
    """Train the learned logistic-regression router on the canonical label set."""
    from training.router import default_router
    out = cfg.get("agent", {}).get("router_path",
                                     "models/checkpoints/router.npz")
    print(f"[train] training learned router → {out}")
    default_router(save_to=out)
    print(f"[train] learned router saved.")


def train_fusion(cfg):
    """Train and cache the learnable α/β fusion weights. Extracted from
    build_system() so evaluation can be strictly read-only (auditor #4)."""
    import json
    import os
    from models.embeddings import load_embedding_model
    from training.evaluation import _train_fusion_and_cache

    with open(cfg["data"]["train_split"]) as f:
        train_chunks = json.load(f)

    use_ft = cfg["embeddings"].get("use_fine_tuned", True)
    encoder = load_embedding_model(
        base_model=cfg["embeddings"]["base_model"],
        fine_tuned_path=cfg["embeddings"].get("fine_tuned_path"),
        use_fine_tuned=use_ft,
        dtype=cfg["embeddings"].get("dtype", "float32"))

    cache_path = os.path.join(
        os.path.dirname(cfg["embeddings"]["index_path_ft"] if use_ft
                        else cfg["embeddings"]["index_path_base"]),
        f"fusion_weights_{'ft' if use_ft else 'base'}.json")
    _train_fusion_and_cache(cfg, encoder, train_chunks, cache_path)


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/full.yaml")
    ap.add_argument("--stage",
                     choices=["biencoder", "fusion", "dora", "router", "all"],
                     default="all")
    args = ap.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    if args.stage in ("biencoder", "all"):
        train_biencoder(cfg)
    if args.stage in ("fusion", "all"):
        train_fusion(cfg)
    if args.stage in ("dora", "all"):
        train_dora(cfg)
    if args.stage in ("router", "all"):
        train_router(cfg)


if __name__ == "__main__":
    main()
