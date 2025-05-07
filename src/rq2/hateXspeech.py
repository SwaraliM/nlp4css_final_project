#!/usr/bin/env python3
"""
Evaluate **Hate-speech-CNERG/bert-base-uncased-hatexplain** on the *Measuring Hate Speech* corpus
(ucberkeley-dlab/measuring-hate-speech), dropping neutral rows.

The model predicts three labels:
    0 → hate speech
    1 → normal
    2 → offensive

We collapse this into binary ground-truth by counting **hate speech** as 1 and
everything else as 0.

Binary ground-truth mapping:
    score > 0.5  → 1 (hate)
    score < -1   → 0 (counter/supportive)

Usage examples:
    python evaluate_hatexplain.py --split test
    python evaluate_hatexplain.py --sample 5000 --batch_size 16
"""

import argparse
import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm


def get_device() -> torch.device:
    """Prefer MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def map_score_to_label(score: float) -> int:
    """Binary label after filtering neutral rows."""
    return int(score > 0)


def main(split: str, sample_size: int | None, batch_size: int) -> None:
    print(f"▶ loading '{split}' split …", flush=True)
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split=split)

    # keep only hate (>0.5) and counter/supportive (<-1)
    ds = ds.filter(
        lambda ex: (ex["hate_speech_score"] > 0.5) or (ex["hate_speech_score"] < -1),
        num_proc=4
    )
    if sample_size:
        ds = ds.shuffle(seed=42).select(range(sample_size))

    ds = ds.map(
        lambda ex: {"label": map_score_to_label(ex["hate_speech_score"])},
        num_proc=4
    )

    device = get_device()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model_name = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        pad_tok = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.add_special_tokens({"pad_token": pad_tok})

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # pipeline automatically handles soft-max; we ask for all class scores
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type != "cpu" else -1,
        batch_size=batch_size,
        return_all_scores=True,
        truncation=True,
        padding=True,
    )

    print(f"▶ using device: {device}; running inference …", flush=True)
    y_pred = []

    for i in tqdm(range(0, len(ds), batch_size)):
        batch_texts = ds["text"][i: i + batch_size]

        outputs = pipe(
            batch_texts,
            truncation=True,
            padding=True
        )

        # `outputs` is List[List[Dict[label,score]]] because we asked for all scores
        for scores in outputs:
            hate_prob = next(item["score"] for item in scores
                             if item["label"] == "hate speech")
            y_pred.append(1 if hate_prob > 0.08 else 0)

    y_true = ds["label"]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    print("\n  Results (neutral removed)")
    print(f"  samples evaluated : {len(y_true):,}")
    print(f"  accuracy          : {acc:.4f}")
    print(f"  precision         : {prec:.4f}")
    print(f"  recall            : {rec:.4f}")
    print(f"  F1-score          : {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "test", "validation"],
        help="Which dataset split to evaluate"
    )
    parser.add_argument(
        "--sample", type=int, dest="sample_size",
        help="Random sample size instead of full split"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference"
    )
    args = parser.parse_args()

    try:
        main(args.split, args.sample_size, args.batch_size)
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
