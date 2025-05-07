#!/usr/bin/env python3
"""
Evaluate s-nlp/roberta_toxicity_classifier on the Measuring-Hate-Speech corpus.

Model labels
    0 → neutral / non-toxic
    1 → toxic

Ground-truth mapping (identical to earlier script):
    hate_speech_score > 0.5  → 1  (hate ⇒ toxic)
    hate_speech_score < -1   → 0  (counter/supportive ⇒ non-toxic)
"""

from __future__ import annotations
import argparse, os, sys
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm


# ────────────────────────── helpers ──────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def map_score_to_label(score: float) -> int:
    """Binary label: 1 = hate (toxic), 0 = counter/supportive (non-toxic)."""
    return int(score > 0)          # dataset already filtered to |score| large


# ──────────────────────────── main ───────────────────────────
def main(split: str, sample_size: int | None, batch_size: int) -> None:
    print(f"▶ loading '{split}' split …", flush=True)
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split=split)

    # Keep only hate (>0.5) and counter/supportive (<-1)
    ds = ds.filter(
        lambda ex: (ex["hate_speech_score"] > 0.5) or (ex["hate_speech_score"] < -1),
        num_proc=4,
    )
    if sample_size:
        ds = ds.shuffle(seed=42).select(range(sample_size))

    ds = ds.map(
        lambda ex: {"label": map_score_to_label(ex["hate_speech_score"])},
        num_proc=4,
    )

    device = get_device()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model_name = "s-nlp/roberta_toxicity_classifier"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        pad_tok = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.add_special_tokens({"pad_token": pad_tok})

    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    toxic_idx = 1                                       # idx 1 = toxic

    print(f"▶ using device: {device}; running inference …", flush=True)
    y_pred: list[int] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(ds), batch_size)):
            texts = ds["text"][i : i + batch_size]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)

            logits = model(**enc).logits               # [batch, 2]
            probs = logits.softmax(dim=-1)
            preds = (probs[:, toxic_idx] > 0.5).int().tolist()
            y_pred.extend(preds)

    y_true = ds["label"]

    # ─────────────── metrics ───────────────
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    print("\n📊  Results (neutral removed)")
    print(f"  samples evaluated : {len(y_true):,}")
    print(f"  accuracy          : {acc:.4f}")
    print(f"  precision         : {prec:.4f}")
    print(f"  recall            : {rec:.4f}")
    print(f"  F1-score          : {f1:.4f}")


# ────────────────────────── CLI glue ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "validation"],
        help="Which dataset split to evaluate",
    )
    parser.add_argument(
        "--sample", type=int, dest="sample_size", help="Random sample size instead of full split"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=32, help="Batch size for inference"
    )
    args = parser.parse_args()

    try:
        main(args.split, args.sample_size, args.batch_size)
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
