# src/rq2/main.py

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
    print(f"\n=== Training {model_name} ===\nUsing tokenizer: {tokenizer_name}\nOutput dir: {output_dir}\n")
    
    dp = DataProcessor(tokenizer_name=tokenizer_name)
    (train_df, val_df, test_df,
     train_enc, train_lbls,
     val_enc,   val_lbls,
     test_enc,  test_lbls) = dp.prepare_dataset(
    csv_path=data_path,
    subset_size=subset_size
)
    # Print overall stats on the *full* DataFrame (we have to reload or capture it)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print("\nDataset Statistics (full):")
    print(f"  Total samples: {len(full_df)}")
    print(f"  Hate samples : {full_df['label'].sum()} "
        f"({full_df['label'].mean()*100:.2f}%)")
    print("\nTarget-group distribution:")
    print(full_df["target_group"].value_counts().to_string())
    
    print(f"Split sizes → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    
    trainer = ModelTrainer(model_name, num_labels)
    history = trainer.train(
        train_enc, train_lbls,
        val_enc,   val_lbls,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    trainer.save_model(output_dir)
    
    evaluator = Evaluator(trainer.model, trainer.tokenizer)
    print("\n▶ Generating evaluation report…")
    report = evaluator.generate_report(test_df)
    
    out = {
        'model_name': model_name,
        'tokenizer_name': tokenizer_name,
        'history': history,
        'report': report
    }
    
    serializable = to_serializable(out)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved results to {output_dir}/results.json")
    
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',         nargs='+', required=True,
                        help="List of model names (HuggingFace identifiers)")
    parser.add_argument('--tokenizer_name', type=str, default=None,
                        help="HuggingFace tokenizer to use (defaults to each model name)")
    parser.add_argument('--data_path',      type=str, default=None)
    parser.add_argument('--output_dir',     type=str, default='results/rq2')
    parser.add_argument('--num_labels',     type=int, default=2)
    parser.add_argument('--num_epochs',     type=int, default=3)
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--subset_size',    type=int, default=None,
                        help="If set, sample up to this many examples per split for quick tests.")
    args = parser.parse_args()
    
    # Basic device check
    print(f"\nPyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    
    for model_name in tqdm(args.models, desc="Models"):
        try:
            tok = args.tokenizer_name or model_name
            out_dir = os.path.join(args.output_dir, model_name.replace('/', '_'))
            print(f"\nProcessing model: {model_name}")
            
            res = train_and_evaluate_model(
                model_name=model_name,
                tokenizer_name=tok,
                data_path=args.data_path,
                output_dir=out_dir,
                num_labels=args.num_labels,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                subset_size=args.subset_size
            )
            all_results[model_name] = res
        except Exception as e:
            print(f"\nError with model {model_name}:")
            print(f"Error: {str(e)}")
            continue
    
    print("\n=== Model Comparisons ===")
    for name, res in all_results.items():
        fm = res['report']['fairness_metrics']
        print(f"{name}: F1_disp={fm['f1_disparity']:.4f}, FPR_disp={fm['fpr_disparity']:.4f}")
    
    cmp_path = os.path.join(args.output_dir, 'model_comparison.json')
    with open(cmp_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\nSaved comparison to {cmp_path}")

if __name__ == "__main__":
    main()
