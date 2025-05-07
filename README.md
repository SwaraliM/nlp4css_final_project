# Hate Speech Detection and Analysis in Social Media

## Project Overview
This project aims to analyze hate speech in social media through two main research questions:

1. **RQ1**: How do the semantic patterns and topic clusters of hate speech differ across identity groups (e.g., race, religion, gender) when analyzed using neural topic modeling?
2. **RQ2**: How do various hate speech detection models perform, particularly on subgroup-targeted content, and what does this reveal about fairness and generalization?

## Setup Instructions
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Implementation

The project is organized into two pipelines corresponding to RQ1 and RQ2:

### RQ1: Identity-Specific Topic Modeling
- We extract posts with high hate scores and group them by 45 identity subgroup flags.
- For each group, we apply three topic modeling methods:
  - **BERTopic**: Transformer-based topic modeling using sentence embeddings.
  - **Top2Vec**: Learns joint topic and document embeddings.
  - **LDA**: Traditional probabilistic topic modeling with coherence evaluation.
- Each model outputs top keywords per topic and coherence scores using the Cv metric.
- Results are aggregated and visualized using `analyze.py`.

**Key scripts:**
- `topic.py`: Loads data, filters by subgroup, runs topic models, and computes coherence.
- `analyze.py`: Extracts top words and exports coherence summaries per model.

### RQ2: Hate Speech Classification and Fairness Evaluation
- Compares four classifiers: TF–IDF + LR, BERT-base, HateBERT, and HateXplain.
- Preprocessing includes hate score filtering, stratified splitting, and tokenization.
- Models are trained using PyTorch and Hugging Face Transformers.
- Fairness is analyzed via subgroup-level metrics and disparity statistics (F1, FPR, FNR).
- LIME is used to explain model predictions on edge-case examples.
- Code Modules:
  - `data_processor.py`: Cleans text, filters labels, and processes subgroups
  - `model_trainer.py`: Trains and fine-tunes models
  - `evaluator.py`: Computes metrics and fairness disparities
  - `lime_analysis.py`: Performs LIME-based interpretability
  - `main.py`: End-to-end orchestration of training, evaluation, and analysis

Both pipelines are modular and support reproducibility and diagnostic evaluation across identity groups.

## Dataset
We use the [Measuring Hate Speech dataset](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) by Kennedy et al. (2020), containing over 39,000 annotated social media comments with subgroup flags and hate speech intensity scores. We binarize the data into hate and non-hate using score thresholds, and focus our analysis on 45 identity subgroups.
```
@article{kennedy2020constructing,
  title={Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application},
  author={Kennedy, Chris J and Bacon, Geoff and Sahn, Alexander and von Vacano, Claudia},
  journal={arXiv preprint arXiv:2009.10277},
  year={2020}
}
```
