# Hate Speech Detection and Analysis in Social Media

## Project Overview
This project aims to analyze hate speech in social media through two main research questions:
1. What identity-specific terms are most commonly associated with hate speech in social media?
2. How do different hate speech detection models perform in terms of their reliance on identity-specific terms, and what does this suggest about their generalizability?

## Project Structure
```
nlp4css_final_project/
├── data/                    # Data directory
│   ├── raw/                # Raw datasets
│   ├── processed/          # Processed datasets
│   └── results/            # Analysis results
├── src/                    # Source code
│   ├── data_processing/    # Data preprocessing scripts
│   ├── analysis/          # Analysis scripts
│   ├── models/            # Model implementation
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
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

## Team Roles and Responsibilities

### Team Member 1: Data Collection and Preprocessing
- Collect and curate social media datasets
- Implement data cleaning and preprocessing pipelines
- Create data documentation

### Team Member 2: Identity Term Analysis
- Implement term extraction and analysis
- Develop methods to identify identity-specific terms
- Create visualizations for term analysis

### Team Member 3: Model Implementation
- Implement and fine-tune hate speech detection models
- Set up model evaluation pipeline
- Document model performance metrics

### Team Member 4: Interpretability Analysis
- Implement LIME/SHAP analysis
- Analyze model weights and feature importance
- Create interpretability visualizations

## Research Pipeline

1. **Data Collection and Preprocessing**
   - Collect social media datasets with hate speech annotations
   - Clean and preprocess text data
   - Split data into train/validation/test sets

2. **Identity Term Analysis (RQ1)**
   - Extract and analyze identity-specific terms
   - Create frequency distributions and visualizations
   - Document patterns in identity-specific hate speech

3. **Model Implementation**
   - Implement pre-trained hate speech detection models:
     - Toxigen Hate BERT
     - Cardiffnlp/twitter-roberta-base-hate-latest
     - Hate-speech-CNERG/bert-base-uncased-hatexplain
   - Fine-tune models on the dataset
   - Evaluate model performance

4. **Interpretability Analysis (RQ2)**
   - Apply LIME/SHAP to analyze model decisions
   - Analyze model weights for identity-specific terms
   - Compare model reliance on identity terms
   - Draw conclusions about model generalizability

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