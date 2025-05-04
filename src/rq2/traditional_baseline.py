# src/rq2/traditional_baseline.py

import os
import json
import argparse
import numpy as np
import pandas as pd

from data_processor import DataProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model        import LogisticRegression
from sklearn.metrics             import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from lime.lime_text import LimeTextExplainer


# --------------------------------------------------------------------------------------------------------------------
# Your 45 fine‑grained subgroup flags
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
# --------------------------------------------------------------------------------------------------------------------


def run_evaluation(train_df, test_df, output_dir):
    """
    Trains TF‑IDF + LR on train_df; evaluates on test_df.
    Saves overall + subgroup + fairness metrics.
    Returns the fitted vectorizer & classifier for downstream LIME.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Train
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=50_000)
    X_train = vect.fit_transform(train_df["processed_text"])
    y_train = train_df["label"]
    clf     = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    # 2) Predict on full test split
    test_df = test_df.reset_index(drop=True)
    X_test  = vect.transform(test_df["processed_text"])
    y_test  = test_df["label"].values
    y_pred  = clf.predict(X_test)

    # 3) Overall metrics
    overall = {}
    overall["accuracy"] = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    overall.update({"precision": prec, "recall": rec, "f1": f1})

    # 4) Subgroup metrics
    subgroup_metrics = {}
    for flag in SUBJECT_COLUMNS:
        df_flag = test_df[test_df[flag]]
        if df_flag.empty:
            continue
        idx = df_flag.index
        y_t = df_flag["label"].values
        y_p = y_pred[idx]

        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        acc_i = accuracy_score(y_t, y_p)
        prec_i, rec_i, f1_i, _ = precision_recall_fscore_support(
            y_t, y_p, average="binary", zero_division=0
        )
        subgroup_metrics[flag] = {
            "accuracy":  float(acc_i),
            "precision": float(prec_i),
            "recall":    float(rec_i),
            "f1":        float(f1_i),
            "fpr":       float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "fnr":       float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        }

    # 5) Fairness disparities
    f1s  = [m["f1"]  for m in subgroup_metrics.values()]
    fprs = [m["fpr"] for m in subgroup_metrics.values()]
    fnrs = [m["fnr"] for m in subgroup_metrics.values()]
    fairness = {
        "f1_disparity":  float(np.std(f1s)),
        "fpr_disparity": float(np.std(fprs)),
        "fnr_disparity": float(np.std(fnrs)),
    }

    # 6) Save metrics
    results = {
        "overall":   overall,
        "subgroups": subgroup_metrics,
        "fairness":  fairness
    }
    with open(os.path.join(output_dir, "traditional_eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("▶ Traditional baseline evaluation complete.")
    return vect, clf


def run_lime_analysis(
    vect,
    clf,
    sample_csv:  str,
    output_json: str,
    n_feats:     int,
    n_samples:   int
):
    """
    Given a fitted vect & clf, load the fixed sample CSV,
    run LIME on each text, and save explanations.
    """
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    df = pd.read_csv(sample_csv)
    # expects columns: original_index, subgroup_flag, true_label, text

    explainer = LimeTextExplainer(class_names=["non_hate", "hate"])

    def predict_proba(texts):
        X = vect.transform(texts)
        return clf.predict_proba(X)

    all_expls = []
    for _, row in df.iterrows():
        txt = row["text"]
        exp = explainer.explain_instance(
            txt,
            predict_proba,
            num_features=n_feats,
            num_samples=n_samples
        )
        pred = int(clf.predict(vect.transform([txt]))[0])
        all_expls.append({
            "original_index":  int(row["original_index"]),
            "subgroup_flag":   row["subgroup_flag"],
            "true_label":      int(row["true_label"]),
            "text":            txt,
            "pred":            pred,
            "lime_explanation": exp.as_list()
        })

    with open(output_json, "w") as f:
        json.dump(all_expls, f, indent=2)
    print(f"▶ LIME explanations saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Train & eval traditional baseline, then (optionally) run LIME"
    )
    parser.add_argument(
        "--data_csv", type=str,
        default="data/measuring-hate-speech.csv"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="results/rq2/traditional_eval"
    )
    parser.add_argument(
        "--sample_csv", type=str,
        default="results/rq2/lime_sample_set_fixed.csv"
    )
    parser.add_argument(
        "--do_lime", action="store_true",
        help="Run LIME explanations (requires --sample_csv)"
    )
    parser.add_argument(
        "--lime_output", type=str,
        default="results/rq2/lime_baseline_explanations.json"
    )
    parser.add_argument(
        "--num_features", type=int, default=5,
        help="Top tokens per LIME explanation"
    )
    parser.add_argument(
        "--num_samples", type=int, default=200,
        help="Number of LIME perturbations"
    )
    args = parser.parse_args()

    # 1) Prepare data
    dp = DataProcessor(tokenizer_name="bert-base-uncased")
    train_df, val_df, test_df, *_ = dp.prepare_dataset(args.data_csv)

    # 2) Eval & save metrics
    vect, clf = run_evaluation(train_df, test_df, args.output_dir)

    # 3) Optionally run LIME
    if args.do_lime:
        run_lime_analysis(
            vect, clf,
            sample_csv  = args.sample_csv,
            output_json = args.lime_output,
            n_feats     = args.num_features,
            n_samples   = args.num_samples
        )


if __name__ == "__main__":
    main()
