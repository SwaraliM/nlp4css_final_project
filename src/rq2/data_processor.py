# src/rq2/data_processor.py

import os
import re
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# -------------------------------------------------------------------
# Explicit list of all 45 fine‑grained subgroup boolean columns:
SUBJECT_COLUMNS = [
    'target_race_asian', 'target_race_black', 'target_race_latinx',
    'target_race_middle_eastern', 'target_race_native_american',
    'target_race_pacific_islander', 'target_race_white', 'target_race_other',
    'target_religion_atheist', 'target_religion_buddhist',
    'target_religion_christian', 'target_religion_hindu',
    'target_religion_jewish', 'target_religion_mormon',
    'target_religion_muslim', 'target_religion_other',
    'target_origin_immigrant', 'target_origin_migrant_worker',
    'target_origin_specific_country', 'target_origin_undocumented',
    'target_origin_other',
    'target_gender_men', 'target_gender_non_binary',
    'target_gender_transgender_men', 'target_gender_transgender_unspecified',
    'target_gender_transgender_women', 'target_gender_women',
    'target_gender_other',
    'target_sexuality_bisexual', 'target_sexuality_gay',
    'target_sexuality_lesbian', 'target_sexuality_straight',
    'target_sexuality_other',
    'target_age_children', 'target_age_teenagers',
    'target_age_young_adults', 'target_age_middle_aged',
    'target_age_seniors', 'target_age_other',
    'target_disability_physical', 'target_disability_cognitive',
    'target_disability_neurological', 'target_disability_visually_impaired',
    'target_disability_hearing_impaired', 'target_disability_unspecific',
    'target_disability_other'
]
# -------------------------------------------------------------------

class DataProcessor:
    """
    Loads and preprocesses the Measuring Hate Speech CSV,
    including filtering, labeling, cleaning text, stratified splitting,
    and tokenization. Provides explicit access to the 45 fine‑grained
    subgroup flags defined in SUBJECT_COLUMNS.
    """

    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _filter_and_label(self, df: pd.DataFrame) -> pd.DataFrame:
        # Keep only explicit hate (>0.5) or counter (<-1), then binarize
        df = df[(df["hate_speech_score"] > 0.5) | (df["hate_speech_score"] < -1)].copy()
        df["label"] = (df["hate_speech_score"] > 0.5).astype(int)
        return df

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def load_local(self, csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Filter, label, and clean
        df = self._filter_and_label(df)
        df["processed_text"] = df["text"].map(self._clean_text)
        # Ensure all SUBJECT_COLUMNS exist (fill missing with False)
        for col in SUBJECT_COLUMNS:
            if col not in df.columns:
                df[col] = False
        return df.reset_index(drop=True)

    def augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = df["label"].value_counts()
        if len(counts) < 2:
            return df
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

    def get_subgroup_flags(self) -> List[str]:
        """
        Return the full, explicit list of subgroup flags to evaluate:
        exactly SUBJECT_COLUMNS.
        """
        return SUBJECT_COLUMNS.copy()

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
        # 1. Load & preprocess full DataFrame
        df = self.load_local(csv_path)

        # 2. Stratified split by a high-level group to get train/val/test
        #    (we keep the subgroup flags intact for later slicing)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df["label"]  # or you can stratify by 'label' to preserve class balance
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=train_df["label"]
        )

        # 3. augment training set for class balance
        train_df = self.augment_data(train_df)

        # 4. subset for quick tests
        if subset_size:
            train_df = train_df.sample(min(subset_size, len(train_df)), random_state=42).reset_index(drop=True)
            val_df   = val_df.sample(min(subset_size,   len(val_df)),   random_state=42).reset_index(drop=True)
            test_df  = test_df.sample(min(subset_size,  len(test_df)),  random_state=42).reset_index(drop=True)

        # 5. Tokenize
        train_enc = self.tokenize_texts(train_df["processed_text"])
        val_enc   = self.tokenize_texts(val_df["processed_text"])
        test_enc  = self.tokenize_texts(test_df["processed_text"])

        return (
            train_df, val_df, test_df,
            train_enc, train_df["label"].values,
            val_enc,   val_df["label"].values,
            test_enc,  test_df["label"].values
        )
