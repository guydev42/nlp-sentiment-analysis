"""
Data loading, text preprocessing, and train/test splitting
for the product reviews sentiment analysis dataset.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Try nltk imports; download if missing
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _stops = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _stops = stopwords.words("english")

STOP_WORDS = set(_stops)
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def load_and_prepare(filepath="data/product_reviews.csv", test_size=0.2, random_state=42):
    """
    Load the product reviews CSV, clean text, and return train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, df (full dataframe with cleaned text)
    """
    df = pd.read_csv(filepath)

    # Clean text
    df["clean_text"] = df["review_text"].apply(clean_text)

    # Encode target
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["sentiment"].map(label_map)

    # Features and target
    X = df["clean_text"].values
    y = df["label"].values

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Classes:      {sorted(label_map.keys())}")
    print(f"Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = load_and_prepare()
    print(f"\nSample cleaned text:\n{df['clean_text'].iloc[0]}")
