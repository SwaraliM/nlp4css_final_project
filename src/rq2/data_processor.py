# src/rq2/data_processor.py

import os
import re
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class DataProcessor:
    """
    Loads the local Measuring Hate Speech CSV (single 'train' split from HF),
    filters & bins the hate_speech_score, assigns target_group from the
    built-in boolean columns, cleans text, optionally balances classes,
    does a stratified train/val/test split, and tokenizes for model input.
    """

    # Map high-level groups to their boolean column prefixes in the CSV
    GROUP_FIELDS = {
        "race_ethnicity":     "target_race",
        "religion":           "target_religion",
        "nationality":        "target_origin",
        "gender":             "target_gender",
        "sexual_orientation": "target_sexuality",
        "age":                "target_age",
        "disability":         "target_disability",
    }

    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _assign_group(self, row: pd.Series) -> str:
        # pick the first matching high-level group
        for group, prefix in self.GROUP_FIELDS.items():
            if row.get(prefix, False):
                return group
        return "other"

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _filter_and_label(self, df: pd.DataFrame) -> pd.DataFrame:
        # keep only clearly hateful (>0.5) or counter-speech/support (< -1)
        df = df[(df["hate_speech_score"] > 0.5) | (df["hate_speech_score"] < -1)].copy()
        # binarize: 1 = hate, 0 = non-hate
        df["label"] = (df["hate_speech_score"] > 0.5).astype(int)
        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._filter_and_label(df)
        # derive high-level group
        df["target_group"]    = df.apply(self._assign_group, axis=1)
        # clean the text
        df["processed_text"]  = df["text"].map(self._clean_text)
        return df.reset_index(drop=True)

    def load_local(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare the full dataset from a local CSV."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        return self._prepare_dataframe(df)

    def augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Duplicate minority-class examples to balance the labels."""
        counts = df["label"].value_counts()
        minority, majority = counts.idxmin(), counts.idxmax()
        diff = counts[majority] - counts[minority]
        if diff <= 0:
            return df
        dup = df[df["label"] == minority].sample(diff, replace=True, random_state=42)
        return pd.concat([df, dup], ignore_index=True)

    def tokenize_texts(self, texts: pd.Series, max_length: int = 128) -> Dict:
        return self.tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def prepare_dataset(
        self,
        csv_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        subset_size: Optional[int] = None
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        Dict, np.ndarray,
        Dict, np.ndarray,
        Dict, np.ndarray
    ]:
        """
        Loads the single local CSV, prepares it, and then:
          - stratified split into train / test (test_size)
          - stratified split train into train / val (val_size of original)
          - optional down-sampling to subset_size per split
          - optional augmentation of the train split
          - tokenization of each split
        """

        # 1. Load & prepare the full DataFrame
        full_df = self.load_local(csv_path)

        # 2. First split: train vs test
        train_df, test_df = train_test_split(
            full_df,
            test_size=test_size,
            random_state=42,
            stratify=full_df["target_group"]
        )

        # 3. Split train into train vs validation
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=train_df["target_group"]
        )

        # 4. Balance only the train split
        train_df = self.augment_data(train_df)

        # 5. Optional subset sampling
        if subset_size:
            train_df = train_df.sample(min(subset_size, len(train_df)), random_state=42).reset_index(drop=True)
            val_df   = val_df.sample(min(subset_size,   len(val_df)),   random_state=42).reset_index(drop=True)
            test_df  = test_df.sample(min(subset_size,  len(test_df)),  random_state=42).reset_index(drop=True)

        # 6. Tokenize text for each split
        train_enc = self.tokenize_texts(train_df["processed_text"])
        val_enc   = self.tokenize_texts(val_df["processed_text"])
        test_enc  = self.tokenize_texts(test_df["processed_text"])

        # 7. Extract labels arrays
        train_lbls = train_df["label"].values
        val_lbls   = val_df["label"].values
        test_lbls  = test_df["label"].values

        return (
            train_df, val_df, test_df,
            train_enc, train_lbls,
            val_enc,   val_lbls,
            test_enc,  test_lbls
        )
