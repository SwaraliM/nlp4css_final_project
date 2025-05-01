# src/rq2/data_processor.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import re
import torch
import os

class DataProcessor:
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load and preprocess the Measuring Hate Speech dataset."""
        if data_path is None:
            data_path = "data/measuring-hate-speech.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset file not found at {data_path}. "
                "Please provide a valid path to the dataset file."
            )
        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)
        df['label'] = (df['hate_speech_score'] > 0.5).astype(int)
        # identity regex patterns...
        target_groups = {
            'race_ethnicity': r'(black|white|asian|latino|hispanic|african|caucasian|indian)',
            'religion':         r'(muslim|jew|christian|hindu|buddhist|atheist)',
            'nationality':      r'(immigrant|foreigner|american|chinese|mexican|russian)',
            'gender':           r'(woman|man|female|male|trans|nonbinary)',
            'sexual_orientation': r'(gay|lesbian|bisexual|queer|lgbt)',
            'age':              r'(old|young|elderly|teen|child)',
            'disability':       r'(disabled|handicapped|retarded|autistic)',
            'political':        r'(liberal|conservative|democrat|republican|leftist|rightist)'
        }
        df['target_group'] = 'other'
        for group, pattern in target_groups.items():
            mask = df['text'].str.contains(pattern, case=False, regex=True)
            df.loc[mask, 'target_group'] = group
        print("\nDataset Statistics:")
        print(f" Total samples: {len(df)}")
        print(f" Hate samples: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
        print("\nTarget group distribution:")
        print(df['target_group'].value_counts())
        return df
    
    def preprocess_text(self, text: str) -> str:
        """Basic text cleaning while preserving identity terms."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def tokenize_data(self, texts: List[str], max_length: int = 128) -> Dict:
        return self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df['target_group']
        )
        train_df, val_df = train_test_split(
            train_df, test_size=val_size/(1-test_size),
            random_state=random_state,
            stratify=train_df['target_group']
        )
        return train_df, val_df, test_df
    
    def augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Duplicate minority class to balance."""
        counts = df['label'].value_counts()
        minority, majority = counts.idxmin(), counts.idxmax()
        n = counts[majority] - counts[minority]
        samp = df[df['label']==minority].sample(n=n, replace=True, random_state=42)
        return pd.concat([df, samp], ignore_index=True)
    
    def prepare_dataset(
        self,
        data_path: str = None,
        subset_size: int = None
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        Dict, np.ndarray, Dict, np.ndarray, Dict, np.ndarray
    ]:
        """
        Returns:
          train_df, val_df, test_df,
          train_enc, train_labels,
          val_enc, val_labels,
          test_enc, test_labels
        """
        df = self.load_data(data_path)
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        df = self.augment_data(df)
        train_df, val_df, test_df = self.split_data(df)
        
        if subset_size:
            train_df = train_df.sample(n=min(subset_size,len(train_df)), random_state=42).reset_index(drop=True)
            val_df   = val_df.sample(n=min(subset_size,len(val_df)),   random_state=42).reset_index(drop=True)
            test_df  = test_df.sample(n=min(subset_size,len(test_df)), random_state=42).reset_index(drop=True)
        
        train_enc = self.tokenize_data(train_df['processed_text'].tolist())
        val_enc   = self.tokenize_data(val_df['processed_text'].tolist())
        test_enc  = self.tokenize_data(test_df['processed_text'].tolist())
        
        return (
            train_df, val_df, test_df,
            train_enc, train_df['label'].values,
            val_enc,   val_df['label'].values,
            test_enc,  test_df['label'].values
        )
