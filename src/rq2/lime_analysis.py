# lime_analysis_from_csv.py

import os
import json
import pandas as pd

from model_trainer import ModelTrainer
from evaluator import Evaluator

# Configuration
MODEL_CKPTS = {
    "bert-base-uncased":   "results/rq2/bert-base-uncased",
    "HateBERT":            "results/rq2/GroNLP_hateBERT",
    "HateXplain":          "results/rq2/Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two"
}
SAMPLE_CSV    = "data/lime_sample_set.csv"
OUTPUT_PATH   = "results/rq2/lime_attributions_from_csv.json"
NUM_LIME_FEATURES = 10
NUM_LIME_SAMPLES  = 300  # adjust for speed / fidelity

def main():
    # 1. Load the fixed sample set
    df = pd.read_csv(SAMPLE_CSV)
    # each row has: original_index, subgroup_flag, bucket, label, text

    all_attributions = {}

    # 2. Loop over each model
    for model_name, ckpt in MODEL_CKPTS.items():
        print(f"\n=== Generating LIME attributions for {model_name} ===")
        # load model & tokenizer
        trainer   = ModelTrainer.load_model(ckpt, num_labels=2)
        evaluator = Evaluator(trainer.model, trainer.tokenizer)

        model_results = []
        # 3. For each example in the CSV, run LIME
        for _, row in df.iterrows():
            text = row["text"]
            exp  = evaluator.generate_lime_explanations(
                text,
                num_features=NUM_LIME_FEATURES,
                num_samples=NUM_LIME_SAMPLES
            )
            model_results.append({
                "original_index":  int(row["original_index"]),
                "subgroup_flag":   row["subgroup_flag"],
                "true_label":           int(row["true_label"]),
                "text":            text,
                "pred":            exp["pred"],
                "lime_explanation": exp["explanation"]
            })

        all_attributions[model_name] = model_results

    # 4. Save aggregated results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_attributions, f, indent=2)
    print(f"\nSaved LIME attributions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
