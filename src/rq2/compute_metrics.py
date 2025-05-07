from data_processor import DataProcessor
from model_trainer import ModelTrainer
from evaluator import Evaluator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    # 1. Load your fine‑tuned model
    ckpt_dir = "results/rq2/GroNLP_hateBERT"   
    print(f"Loading model from {ckpt_dir}...")
    trainer   = ModelTrainer.load_model(ckpt_dir, num_labels=2)
    
    # 2. Wrap it in the Evaluator
    evaluator = Evaluator(trainer.model, trainer.tokenizer)

    # 3. Load & preprocess the test split
    dp = DataProcessor(tokenizer_name="bert-base-uncased")
    # We only need the test_df here
    _, _, test_df, *_ = dp.prepare_dataset(csv_path="data/measuring-hate-speech.csv")

    # 4. Make predictions
    texts  = test_df["processed_text"].tolist()
    y_true = test_df["label"].values
    print("Running predictions on test set...")
    y_pred = evaluator.predict(texts)

    # 5. Compute overall metrics
    acc    = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # 6. Print results
    print(f"\nOverall Test Metrics for {ckpt_dir}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

if __name__ == "__main__":
    main()