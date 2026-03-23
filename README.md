<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=NLP%20Sentiment%20Analysis&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Classify%20product%20reviews%20into%20positive%2C%20neutral%2C%20and%20negative%20sentiment&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Macro--F1-0.87-9558B2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-89%25-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Reviews-5%2C000-f59e0b?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a>
</p>

</div>

---

## Overview

> **Classifying product review sentiment using TF-IDF features and linear classifiers to achieve 0.87 macro-F1 across three classes.**

Sentiment analysis of product reviews helps businesses understand customer satisfaction at scale. This project builds a text classification pipeline that assigns product reviews to one of three sentiment classes (positive, neutral, negative) using TF-IDF vectorization with unigram+bigram features. Three classifiers are compared -- Logistic Regression, SVM (LinearSVC), and Random Forest -- with SVM achieving the best performance. The pipeline includes text preprocessing (lowering, stopword removal, lemmatization), cross-validated model selection, and analysis of the most predictive words per sentiment class.

```
Problem   →  Manual review reading does not scale beyond a few hundred reviews
Solution  →  TF-IDF + SVM pipeline classifies sentiment with 0.87 macro-F1
Impact    →  89% accuracy across 3 classes, with interpretable top predictive words
```

---

## Key results

| Metric | Value |
|--------|-------|
| Best model | SVM (LinearSVC) |
| Macro-F1 | 0.87 |
| Accuracy | 89% |
| Reviews classified | 5,000 |
| Sentiment classes | 3 (positive, neutral, negative) |

**Key findings**

- **SVM outperforms Logistic Regression and Random Forest** on TF-IDF features, consistent with SVMs being well suited for high-dimensional sparse text data
- **Neutral reviews are the hardest class** to classify, as they share vocabulary with both positive and negative reviews
- **Bigrams improve performance** over unigrams alone by capturing phrases like "waste money" and "highly recommend"
- **Top predictive words align with intuition** -- "love", "excellent", "great" for positive; "waste", "broke", "terrible" for negative
- **10,000 TF-IDF features with (1,2) n-grams** provides the best balance of performance and efficiency

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Raw text    │───▶│  Preprocessing   │───▶│  TF-IDF             │
│  (5K reviews)│    │  (clean, lemma)  │    │  vectorization      │
└─────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                      │
                          ┌───────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Model training      │───▶│  Evaluation          │
              │  (LR, SVM, RF)       │    │  (F1, confusion)     │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Streamlit app       │
                                          │  (5-page dashboard)  │
                                          └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_20_nlp_sentiment_analysis/
├── data/                  # Product reviews dataset (5,000 reviews)
│   └── generate_data.py   # Synthetic data generator
├── src/                   # Data loading, preprocessing, model training
│   ├── __init__.py
│   ├── data_loader.py
│   └── model.py
├── models/                # Saved best model and TF-IDF vectorizer
├── outputs/               # Plots, comparison tables, top words
├── notebooks/             # EDA, feature engineering, modeling, evaluation
├── app.py                 # Streamlit dashboard (5 pages)
├── requirements.txt       # Python dependencies
├── index.html             # Project landing page
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_20_nlp_sentiment_analysis

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python data/generate_data.py

# Train models and generate outputs
python -c "
from src.data_loader import load_and_prepare
from src.model import train_and_evaluate
X_train, X_test, y_train, y_test, _ = load_and_prepare('data/product_reviews.csv')
train_and_evaluate(X_train, X_test, y_train, y_test)
"

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic product reviews |
| Records | 5,000 reviews |
| Classes | 3 (positive 45%, neutral 30%, negative 25%) |
| Features | review_text, rating, product_category, review_length, word_count |
| Categories | Electronics, Clothing, Home & Kitchen, Books, Sports & Outdoors |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/WordCloud-4A90D9?style=for-the-badge"/>
</p>

---

## Methodology

<details>
<summary><b>Text preprocessing</b></summary>

- Lowercasing and punctuation removal
- NLTK stopword removal (English)
- WordNet lemmatization
- Short token filtering (length > 2)
</details>

<details>
<summary><b>Feature extraction</b></summary>

- TF-IDF vectorization with sublinear TF scaling
- Unigram + bigram features (ngram_range 1,2)
- 10,000 max features, min_df=3, max_df=0.95
</details>

<details>
<summary><b>Model comparison</b></summary>

- Logistic Regression (multinomial, C=1.0)
- SVM (LinearSVC, C=1.0)
- Random Forest (200 trees, max_depth=50)
- 5-fold cross-validation on macro-F1
</details>

<details>
<summary><b>Evaluation</b></summary>

- Confusion matrices (raw and normalized)
- Per-class precision, recall, F1
- Most informative features per class (SVM coefficients)
- Error analysis of misclassified reviews
</details>

---

## Acknowledgements

Dataset generated synthetically for this project. Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
