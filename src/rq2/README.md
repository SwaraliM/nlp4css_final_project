# RQ2: Supervised Hate Speech Detection Across Identity Subgroups

This directory contains the implementation for Research Question 2 (RQ2), which focuses on evaluating supervised hate speech detection models across different identity subgroups.

## Project Structure

```
src/rq2/
├── data_processor.py    # Handles data loading, preprocessing, and augmentation
├── model_trainer.py     # Manages model training and evaluation
├── evaluator.py         # Implements subgroup analysis and fairness metrics
├── main.py             # Main script to run the complete pipeline
└── README.md           # This file
```

## Implementation Details

### Data Processing (`data_processor.py`)
- Handles data loading and preprocessing while preserving identity terms
- Implements data splitting with balanced subgroup representation
- Provides data augmentation capabilities (to be implemented with ToxiGen)

### Model Training (`model_trainer.py`)
- Supports multiple BERT-based models:
  - Baseline BERT
  - HateBERT
  - HateXplain-BERT
  - (More models can be added)
- Implements training with early stopping and learning rate scheduling
- Provides model saving and loading functionality

### Evaluation (`evaluator.py`)
- Implements subgroup-specific evaluation metrics
- Calculates fairness metrics (F1 disparity, FPR disparity, FNR disparity)
- Provides interpretability analysis using LIME and SHAP
- Generates visualizations of model performance across subgroups

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
- Ensure your dataset includes:
  - Text content
  - Hate speech labels
  - Identity subgroup labels
- Format: CSV with columns 'text', 'label', 'target_group'

3. Update configuration in `main.py`:
- Set the path to your dataset
- Configure model parameters
- Adjust training hyperparameters

4. Run the pipeline:
```bash
python src/rq2/main.py
```

## Output

The pipeline generates:
1. Trained models for each configuration
2. Evaluation results per subgroup
3. Fairness metrics comparison
4. Interpretability analysis
5. Visualizations of model performance

Results are saved in the `results/rq2` directory:
- `results.json` for each model
- `model_comparison.json` for overall comparison
- Model checkpoints
- Visualization plots

## Notes

- The implementation assumes binary classification (hate/not-hate)
- Data augmentation using ToxiGen needs to be implemented
- Multilingual support can be added by including appropriate models
- The pipeline can be extended to include more models or evaluation metrics

## References

1. Measuring Hate Speech Corpus
2. HateXplain Dataset
3. ToxiGen Dataset
4. LIME and SHAP for interpretability
5. Fairness metrics from Jigsaw's unintended bias challenge 