# src/rq2/traditional_baseline.py

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score
)

from data_processor import DataProcessor


class TraditionalBaseline:
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        tfidf_ngram_range=(1, 2),
        tfidf_max_features: int = 5000
    ):
        # reuse DataProcessor for loading & preprocessing
        self.dp = DataProcessor(tokenizer_name)
        # set up TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=tfidf_ngram_range,
            max_features=tfidf_max_features
        )
        # balanced logistic regression classifier
        self.clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )

    def train(self, texts: pd.Series, labels: pd.Series):
        X = self.vectorizer.fit_transform(texts)
        y = labels.values
        self.clf.fit(X, y)

    def predict(self, texts: pd.Series) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.clf.predict(X)

    def evaluate_subgroup(
        self,
        texts: pd.Series,
        labels: pd.Series
    ) -> Dict[str, float]:
        y_true = labels.values
        y_pred = self.predict(texts)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1":       f1_score(y_true, y_pred, zero_division=0),
            "fpr":      fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "fnr":      fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        }

    def evaluate_all_subgroups(self, df: pd.DataFrame) -> Dict[str, Dict]:
        results = {}
        for grp in df["target_group"].unique():
            sub = df[df["target_group"] == grp]
            results[grp] = self.evaluate_subgroup(
                sub["processed_text"],
                sub["label"]
            )
        return results

    def calculate_fairness_metrics(self, subgroup_results: Dict[str, Dict]) -> Dict:
        f1s  = [r["f1"]  for r in subgroup_results.values()]
        fprs = [r["fpr"] for r in subgroup_results.values()]
        fnrs = [r["fnr"] for r in subgroup_results.values()]
        return {
            "f1_disparity":  float(np.std(f1s)),
            "fpr_disparity": float(np.std(fprs)),
            "fnr_disparity": float(np.std(fnrs)),
        }

    def run(self, data_path: str, subset_size: int = None) -> Dict:
        # load & preprocess the data
        (
            train_df, val_df, test_df,
            *_,
        ) = self.dp.prepare_dataset(data_path, subset_size)

        # train on the training split
        self.train(train_df["processed_text"], train_df["label"])

        # overall metrics on test set
        y_true = test_df["label"].values
        y_pred = self.predict(test_df["processed_text"])
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # subgroup‐level & fairness metrics
        subgroup_res = self.evaluate_all_subgroups(test_df)
        fairness    = self.calculate_fairness_metrics(subgroup_res)

        return {
            "overall": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            },
            "subgroup_results": subgroup_res,
            "fairness_metrics": fairness
        }


def feature_weight_analysis(baseline: TraditionalBaseline, output_dir: str):
    """
    Extract and save top positive and negative TF-IDF feature weights
    from the trained logistic regression model.
    """
    # Get feature names and coefficients
    features = baseline.vectorizer.get_feature_names_out()
    coefs    = baseline.clf.coef_[0]

    # Pair and sort
    feat_coef    = list(zip(features, coefs))
    top_positive = sorted(feat_coef, key=lambda x: x[1], reverse=True)[:20]
    top_negative = sorted(feat_coef, key=lambda x: x[1])[:20]

    # Create DataFrames
    df_pos = pd.DataFrame(top_positive, columns=["feature", "weight"])
    df_neg = pd.DataFrame(top_negative, columns=["feature", "weight"])

    # Save as CSV
    os.makedirs(output_dir, exist_ok=True)
    pos_path = os.path.join(output_dir, "top_positive_features.csv")
    neg_path = os.path.join(output_dir, "top_negative_features.csv")
    df_pos.to_csv(pos_path, index=False)
    df_neg.to_csv(neg_path, index=False)

    # Print to console
    print("\nTop 20 features most indicative of HATE:")
    print(df_pos.to_string(index=False))
    print("\nTop 20 features most indicative of NON-HATE:")
    print(df_neg.to_string(index=False))
    print(f"\nFeature weights saved to:\n  {pos_path}\n  {neg_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run traditional TF-IDF + LogisticRegression baseline"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/measuring-hate-speech.csv",
        help="Path to the Measuring Hate Speech CSV file"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="If set, sample this many examples per split"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/rq2/baseline",
        help="Directory to save the results JSON and feature weights"
    )
    args = parser.parse_args()

    # Run the baseline
    baseline = TraditionalBaseline()
    results = baseline.run(args.data_path, args.subset_size)

    # Save overall results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "traditional_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Traditional baseline results saved to {out_path}")

    # Perform feature weight analysis
    feature_weight_analysis(baseline, args.output_dir)
