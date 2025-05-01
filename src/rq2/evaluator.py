# src/rq2/evaluator.py

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix
)
from lime.lime_text import LimeTextExplainer
import shap
import torch
from torch.utils.data import DataLoader, TensorDataset

class Evaluator:
    def __init__(self, model, tokenizer, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                batch_preds = torch.argmax(outputs.logits, dim=1)
                preds.extend(batch_preds.cpu().numpy())
        return np.array(preds)

    def evaluate_subgroup(
        self,
        texts: List[str],
        labels: np.ndarray,
        subgroup: str
    ) -> Dict:
        preds = self.predict(texts)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1":       float(f1_score(labels, preds, average="weighted")),
            "fpr":      float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "fnr":      float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        }

    def evaluate_all_subgroups(self, test_data: pd.DataFrame) -> Dict:
        print(" ▶ Evaluating subgroups…")
        results = {}
        for grp in test_data["target_group"].unique():
            df = test_data[test_data["target_group"] == grp]
            results[grp] = self.evaluate_subgroup(
                df["processed_text"].tolist(),
                df["label"].values,
                grp
            )
        return results

    def calculate_fairness_metrics(self, subgroup_results: Dict) -> Dict:
        print(" ▶ Calculating fairness metrics…")
        f1s  = [r["f1"] for r in subgroup_results.values()]
        fprs = [r["fpr"] for r in subgroup_results.values()]
        fnrs = [r["fnr"] for r in subgroup_results.values()]
        return {
            "f1_scores":     dict(zip(subgroup_results.keys(), f1s)),
            "f1_disparity":  float(np.std(f1s)),
            "fpr_disparity": float(np.std(fprs)),
            "fnr_disparity": float(np.std(fnrs)),
        }

    def generate_lime_explanations(
        self,
        text: str,
        num_features: int = 5,
        num_samples: int = 20
    ) -> Dict:
        print(f" ▶ LIME on sample…")
        explainer = LimeTextExplainer(class_names=["not_hate", "hate"])
        def predict_proba(texts):
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"]
                ).logits
            return torch.softmax(logits, dim=1).cpu().numpy()

        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        return {
            "explanation": exp.as_list(),
            "pred":        int(exp.predict_proba.argmax())
        }

    def _to_list(self, obj):
        """
        Convert numpy arrays, tuples, lists, scalars into pure Python lists/scalars.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return list(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        return obj
    
    def generate_shap_explanations(
        self,
        texts: List[str],
        max_samples: int = 10
    ) -> Dict:
        print(" ▶ Generating SHAP explanations…")
        def f(x_texts):
            enc = self.tokenizer(
                x_texts.tolist(),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                return self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"]
                ).logits.cpu().numpy()

        explainer = shap.Explainer(f, self.tokenizer, algorithm="partition")
        shap_out = explainer(texts[:max_samples])

        # shap_out might be a tuple (Explanation, meta) or an Explanation directly
        explanation = shap_out[0] if isinstance(shap_out, tuple) else shap_out

        # Use our helper to convert each field
        values      = self._to_list(explanation.values)      if hasattr(explanation, "values")      else None
        base_values = self._to_list(explanation.base_values) if hasattr(explanation, "base_values") else None
        data        = self._to_list(explanation.data)        if hasattr(explanation, "data")        else None

        return {
            "values":      values,
            "base_values": base_values,
            "data":        data
        }
    

    def generate_report(self, test_data: pd.DataFrame) -> Dict:
        sg      = self.evaluate_all_subgroups(test_data)
        fm      = self.calculate_fairness_metrics(sg)
        samples = test_data.sample(min(5, len(test_data)))["processed_text"].tolist()

        lime_ex = {t: self.generate_lime_explanations(t) for t in samples}
        shap_ex = self.generate_shap_explanations(samples)

        print(" ▶ Report complete")
        return {
            "subgroup_results": sg,
            "fairness_metrics": fm,
            "lime":             lime_ex,
            "shap":             shap_ex
        }
