# export_lime_samples_fixed.py

import os
import pandas as pd
from data_processor import DataProcessor

# Config
DATA_PATH            = "data/measuring-hate-speech.csv"
OUTPUT_CSV           = "data/lime_sample_set.csv"
NUM_HATE_SAMPLES     = 2   # per subgroup
NUM_NONHATE_SAMPLES  = 2   # per subgroup

def get_flag_columns(df):
    return [c for c in df.columns 
            if c.startswith("target_") and c != "target_group"]

def sample_fixed_by_label(df, flag_cols):
    records = []
    for flag in flag_cols:
        sub = df[df[flag]]
        if sub.empty: 
            continue

        # sample hateful
        hate = sub[sub.label == 1]
        hate_sample = hate.sample(min(NUM_HATE_SAMPLES, len(hate)), random_state=42)
        # sample non‑hateful
        non = sub[sub.label == 0]
        non_sample = non.sample(min(NUM_NONHATE_SAMPLES, len(non)), random_state=42)

        for _, row in hate_sample.iterrows():
            records.append({
                "original_index": row.name,
                "subgroup_flag":  flag,
                "true_label":     int(row.label),
                "text":           row.processed_text
            })
        for _, row in non_sample.iterrows():
            records.append({
                "original_index": row.name,
                "subgroup_flag":  flag,
                "true_label":     int(row.label),
                "text":           row.processed_text
            })
    return pd.DataFrame(records)

def main():
    # 1. Load & preprocess test set
    dp = DataProcessor(tokenizer_name="bert-base-uncased")
    _, _, test_df, *_ = dp.prepare_dataset(DATA_PATH)

    # 2. Identify subgroup flags
    flag_cols = get_flag_columns(test_df)

    # 3. Sample fixed examples by true label
    sample_df = sample_fixed_by_label(test_df, flag_cols)

    # 4. Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    sample_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported {len(sample_df)} fixed examples to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
