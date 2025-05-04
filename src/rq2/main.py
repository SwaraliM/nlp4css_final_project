import os
import json
import argparse
from typing import Dict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from evaluator import Evaluator

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj

def train_and_evaluate_model(
    model_name: str,
    tokenizer_name: str,
    data_path: str,
    output_dir: str,
    num_labels: int,
    num_epochs: int,
    batch_size: int,
    subset_size: int = None
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Training {model_name} ===")
    
    dp = DataProcessor(tokenizer_name=tokenizer_name)
    (train_df, val_df, test_df,
     train_enc, train_lbls,
     val_enc,   val_lbls,
     test_enc,  test_lbls) = dp.prepare_dataset(
         csv_path=data_path,
         subset_size=subset_size
    )
    
    # Train
    trainer = ModelTrainer(model_name, num_labels)
    history = trainer.train(
        train_enc, train_lbls,
        val_enc,   val_lbls,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    trainer.save_model(output_dir)
    
    # Evaluate
    evaluator = Evaluator(trainer.model, trainer.tokenizer)
    test_df = test_df.reset_index(drop=True)
    texts  = test_df["processed_text"].tolist()
    y_true = test_df["label"].values
    print("▶ Running predictions on test set…")
    y_pred = evaluator.predict(texts)
    test_df["pred"] = y_pred

    # Overall metrics
    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    overall = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1
    }

    # Subgroup metrics
    subgroup_flags = dp.get_subgroup_flags()
    subgroup_metrics = {}
    for flag in subgroup_flags:
        sub_df = test_df[test_df[flag]]
        if len(sub_df) == 0:
            continue
        y_t = sub_df["label"].values
        y_p = sub_df["pred"].values  # align predictions by index

        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0,1]).ravel()
        accuracy = accuracy_score(y_t, y_p)
        prec_i, rec_i, f1_i, _ = precision_recall_fscore_support(
            y_t, y_p, average="binary", zero_division=0
        )
        subgroup_metrics[flag] = {
            "accuracy":  float(accuracy),
            "precision": float(prec_i),
            "recall":    float(rec_i),
            "f1":        float(f1_i),
            "fpr":       float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "fnr":       float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        }

    # Fairness disparities
    f1s  = [m["f1"]  for m in subgroup_metrics.values()]
    fprs = [m["fpr"] for m in subgroup_metrics.values()]
    fnrs = [m["fnr"] for m in subgroup_metrics.values()]
    fairness_metrics = {
        "f1_disparity":  float(np.std(f1s)),
        "fpr_disparity": float(np.std(fprs)),
        "fnr_disparity": float(np.std(fnrs)),
    }

    # Package results
    out = {
        "model_name":        model_name,
        "tokenizer_name":    tokenizer_name,
        "history":           history,
        "overall_metrics":   overall,
        "subgroup_metrics":  subgroup_metrics,
        "fairness_metrics":  fairness_metrics
    }

    # Save
    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(to_serializable(out), f, indent=2)
    print(f"▶ Saved results to {result_path}")

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',       nargs='+', required=True)
    parser.add_argument('--tokenizer',    type=str, default=None)
    parser.add_argument('--data_path',    type=str, required=True)
    parser.add_argument('--output_dir',   type=str, default='results/rq2')
    parser.add_argument('--num_labels',   type=int, default=2)
    parser.add_argument('--num_epochs',   type=int, default=3)
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--subset_size',  type=int, default=None)
    args = parser.parse_args()

    # Device info
    print(f"\nPyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for model_name in tqdm(args.models, desc="Models"):
        tok = args.tokenizer or model_name
        out_dir = os.path.join(args.output_dir, model_name.replace('/', '_'))
        print(f"\n→ Processing: {model_name}")
        res = train_and_evaluate_model(
            model_name, tok,
            args.data_path,
            out_dir,
            args.num_labels,
            args.num_epochs,
            args.batch_size,
            args.subset_size
        )
        all_results[model_name] = res

    # summary comparison
    cmp_path = os.path.join(args.output_dir, 'model_comparison.json')
    with open(cmp_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\nSaved comparison summary to {cmp_path}")

if __name__ == "__main__":
    main()
