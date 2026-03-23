"""
Train, evaluate, and compare sentiment classification models.
Includes: TF-IDF + Logistic Regression, TF-IDF + SVM, TF-IDF + Random Forest.
Generates confusion matrices, classification reports, and top predictive words.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
CLASS_NAMES = ["negative", "neutral", "positive"]


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _get_models():
    """Return model instances for comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, C=1.0, multi_class="multinomial"
        ),
        "SVM (LinearSVC)": LinearSVC(
            random_state=RANDOM_STATE, max_iter=2000, C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=200, max_depth=50, n_jobs=-1
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Vectorize text with TF-IDF (unigrams + bigrams), train three classifiers,
    cross-validate, compare metrics, and generate all outputs.
    """
    _ensure_dirs()

    # TF-IDF vectorization with unigrams and bigrams
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    feature_names = tfidf.get_feature_names_out()

    print(f"TF-IDF vocabulary size: {len(feature_names)}")
    print(f"Train matrix shape: {X_train_tfidf.shape}")
    print(f"Test matrix shape:  {X_test_tfidf.shape}")

    models_config = _get_models()
    results = {}
    trained_models = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)

    for name, model in models_config.items():
        print(f"\n--- {name} ---")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring="f1_macro")
        print(f"  CV macro-F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Train on full training set
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model

        # Predict
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        macro_prec = precision_score(y_test, y_pred, average="macro")
        macro_rec = recall_score(y_test, y_pred, average="macro")
        per_class_prec = precision_score(y_test, y_pred, average=None)
        per_class_rec = recall_score(y_test, y_pred, average=None)
        per_class_f1 = f1_score(y_test, y_pred, average=None)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "per_class_precision": per_class_prec,
            "per_class_recall": per_class_rec,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }

        print(f"  Accuracy:        {acc:.4f}")
        print(f"  Macro-F1:        {macro_f1:.4f}")
        print(f"  Macro-precision: {macro_prec:.4f}")
        print(f"  Macro-recall:    {macro_rec:.4f}")
        print(f"\n  Classification report:")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # --- Model comparison table ---
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame({
        name: {
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "macro_precision": r["macro_precision"],
            "macro_recall": r["macro_recall"],
            "cv_f1_mean": r["cv_f1_mean"],
        }
        for name, r in results.items()
    }).T.round(4)
    print(comparison_df.to_string())
    comparison_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))

    # --- Find best model ---
    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best_f1 = results[best_name]["macro_f1"]
    print(f"\nBest model: {best_name} (macro-F1 = {best_f1:.4f})")

    # Save best model and vectorizer
    joblib.dump(trained_models[best_name], os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    print(f"Saved best model and TF-IDF vectorizer to {MODELS_DIR}/")

    # --- Generate plots ---
    _plot_confusion_matrices(results)
    _plot_model_comparison(comparison_df)
    _plot_top_predictive_words(trained_models, feature_names, best_name)
    _save_per_class_metrics(results)

    return results


def _plot_confusion_matrices(results):
    """Plot confusion matrices for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(
            r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        )
        ax.set_title(f"{name}\n(macro-F1: {r['macro_f1']:.3f})")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved confusion matrices plot.")


def _plot_model_comparison(comparison_df):
    """Bar chart comparing model metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df[["accuracy", "macro_f1", "macro_precision", "macro_recall"]].plot(
        kind="bar", ax=ax, edgecolor="black", width=0.7
    )
    ax.set_title("Model comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved model comparison plot.")


def _plot_top_predictive_words(trained_models, feature_names, best_name):
    """Extract and plot top predictive words per class from the best model."""
    model = trained_models[best_name]

    # Get coefficients (works for LogReg and LinearSVC)
    if hasattr(model, "coef_"):
        coef = model.coef_
    else:
        # Random Forest does not have coef_; use feature importances instead
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-30:]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_idx)), importances[top_idx], color="steelblue")
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([feature_names[i] for i in top_idx])
        ax.set_title(f"Top 30 features - {best_name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUTS_DIR, "top_features.png"), dpi=150)
        plt.close(fig)
        print("Saved top features plot.")
        return

    # For models with coef_ (one row per class in multiclass)
    n_top = 15
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for idx, (class_name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        if coef.shape[0] > idx:
            class_coef = coef[idx]
        else:
            class_coef = coef[0]

        top_positive = np.argsort(class_coef)[-n_top:]
        top_words = [feature_names[i] for i in top_positive]
        top_scores = class_coef[top_positive]

        colors = ["#E8C230" if s > 0 else "#3B6FD4" for s in top_scores]
        ax.barh(range(len(top_words)), top_scores, color=colors, edgecolor="black", alpha=0.85)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words)
        ax.set_title(f"Top words: {class_name}")
        ax.set_xlabel("TF-IDF coefficient")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Most predictive words per class ({best_name})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "top_predictive_words.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved top predictive words plot.")

    # Also save as CSV
    records = []
    for idx, class_name in enumerate(CLASS_NAMES):
        if coef.shape[0] > idx:
            class_coef = coef[idx]
        else:
            class_coef = coef[0]
        top_idx = np.argsort(class_coef)[-n_top:][::-1]
        for rank, fi in enumerate(top_idx, 1):
            records.append({
                "class": class_name,
                "rank": rank,
                "word": feature_names[fi],
                "coefficient": round(class_coef[fi], 4),
            })

    pd.DataFrame(records).to_csv(os.path.join(OUTPUTS_DIR, "top_words_per_class.csv"), index=False)
    print("Saved top words per class CSV.")


def _save_per_class_metrics(results):
    """Save per-class precision, recall, and F1 for each model."""
    records = []
    for name, r in results.items():
        for idx, class_name in enumerate(CLASS_NAMES):
            records.append({
                "model": name,
                "class": class_name,
                "precision": round(r["per_class_precision"][idx], 4),
                "recall": round(r["per_class_recall"][idx], 4),
                "f1": round(r["per_class_f1"][idx], 4),
            })
    pd.DataFrame(records).to_csv(os.path.join(OUTPUTS_DIR, "per_class_metrics.csv"), index=False)
    print("Saved per-class metrics CSV.")


if __name__ == "__main__":
    from data_loader import load_and_prepare
    X_train, X_test, y_train, y_test, df = load_and_prepare()
    train_and_evaluate(X_train, X_test, y_train, y_test)
