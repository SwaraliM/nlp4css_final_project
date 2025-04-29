# Hate Speech Detection and Analysis in Social Media

## Project Overview
This project aims to analyze hate speech in social media through two main research questions:
1. What identity-specific terms are most commonly associated with hate speech in social media?
2. How do different hate speech detection models perform in terms of their reliance on identity-specific terms, and what does this suggest about their generalizability?

## Project Structure
```
nlp4css_final_project/
├── src/
│   ├── rq1/                            # Team A: Topic modeling
│   │   ├── data_processing/            # loading & prep of Measuring Hate Speech
│   │   │   └── loader.py
│   │   ├── models/                     # BERTopic, Top2Vec, LDA wrappers
│   │   │   └── topic_models.py
│   │   ├── analysis/                   # coherence scoring, qualitative checks
│   │   │   └── evaluate_topics.py
│   │   └── utils/                      # shared/helper functions
│   │       └── text_utils.py
│   │
│   └── rq2/                            # Team B: Supervised BERT classification
│       ├── data_processing/            # loading & preprocessing
│       │   └── loader.py
│       ├── models/                     # fine-tuning & prediction
│       │   └── finetune.py
│       ├── evaluation/                 # subgroup metrics & robustness tests
│       │   └── subgroup_eval.py
│       ├── interpretability/           # LIME, SHAP wrappers
│       │   └── explainers.py
│       └── utils/                      # config, logger, common funcs
│           └── config.py
│
├── notebooks/
│   ├── rq1/                            # exploratory & demo notebooks for RQ1
│   └── rq2/                            # experiments & visualizations for RQ2
│
├── requirements.txt
└── README.md
```

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

## Research Questions Implementation

### Research Question 1: Identity-Specific Terms Analysis
Team A will focus on identifying and analyzing identity-specific terms in hate speech using topic modeling approaches.

#### Implementation Steps:
1. **Data Processing** (`src/rq1/data_processing/`)
   - Load and preprocess the Measuring Hate Speech dataset
   - Implement text cleaning and normalization
   - Extract identity-specific terms

2. **Topic Modeling** (`src/rq1/models/`)
   - Implement BERTopic, Top2Vec, and LDA models
   - Fine-tune models for optimal performance
   - Extract and analyze topics

3. **Analysis** (`src/rq1/analysis/`)
   - Evaluate topic coherence
   - Perform qualitative analysis of topics
   - Identify identity-specific patterns

### Research Question 2: Model Interpretability Analysis
Team B will focus on analyzing model performance and interpretability using pre-trained hate speech detection models.

#### Implementation Steps:
1. **Data Processing** (`src/rq2/data_processing/`)
   - Load and preprocess data for model training
   - Create train/validation/test splits
   - Implement data augmentation if needed

2. **Model Implementation** (`src/rq2/models/`)
   - Implement and fine-tune pre-trained models:
     - Toxigen Hate BERT
     - Cardiffnlp/twitter-roberta-base-hate-latest
     - Hate-speech-CNERG/bert-base-uncased-hatexplain
   - Set up model evaluation pipeline

3. **Evaluation** (`src/rq2/evaluation/`)
   - Implement subgroup analysis
   - Calculate performance metrics
   - Analyze model robustness

4. **Interpretability** (`src/rq2/interpretability/`)
   - Implement LIME and SHAP analysis
   - Analyze feature importance
   - Visualize model decisions

## Team Roles and Responsibilities

### Team A (RQ1): Topic Modeling Team
- **Member 1**: Data Processing
  - Implement data loading and preprocessing
  - Create data documentation
  - Set up data pipelines

- **Member 2**: Model Implementation
  - Implement topic modeling approaches
  - Fine-tune models
  - Document model configurations

### Team B (RQ2): Model Analysis Team
- **Member 3**: Model Implementation
  - Implement and fine-tune hate speech models
  - Set up evaluation pipeline
  - Document model performance

- **Member 4**: Interpretability Analysis
  - Implement LIME/SHAP analysis
  - Analyze model weights
  - Create interpretability visualizations

## Dependencies
- Python 3.8+
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- lime
- shap
- bertopic
- top2vec
- gensim (for LDA)
- spacy
- nltk

## Notebooks
- `notebooks/rq1/`: Contains exploratory analysis and topic modeling experiments
- `notebooks/rq2/`: Contains model training, evaluation, and interpretability experiments

## Contributing
1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Commit your changes: `git commit -m 'Add some feature'`
3. Push to the branch: `git push origin feature/your-feature-name`
4. Submit a pull request 