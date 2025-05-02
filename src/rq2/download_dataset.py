# download_hate_speech_csv.py

import os
import pandas as pd
from datasets import load_dataset

def main(
    hf_name: str = "ucberkeley-dlab/measuring-hate-speech",
    split: str = "train",
    output_path: str = "data/measuring_hate_speech.csv"
):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the single train split
    print(f"Loading '{split}' split from {hf_name}…")
    ds = load_dataset(hf_name, split=split)
    
    # Convert to pandas DataFrame
    df = ds.to_pandas()
    print(f"Loaded {len(df)} rows.")

    # Save to local CSV
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
