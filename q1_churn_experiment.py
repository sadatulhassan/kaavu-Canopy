"""
Q1 starter experiment:
Does prediction churn shrink as fast as error when training size increases?

Dataset: reproducible synthetic binary classification data (12,000 rows).
Test set and feature columns are fixed across all conditions.
Models: Logistic Regression, Decision Tree, Histogram Gradient Boosting.

Run:
    python q1_churn_experiment.py

Outputs:
    q1_results.csv
    q1_summary.csv

Definitions:
    accuracy = fraction of correct test predictions
    error_rate = 1 - accuracy
    churn = fraction of test points where two models' predictions differ

Two controls:
    data-driven churn: train on different samples, with algorithm seed held fixed
    seed-driven churn: train on the same sample, with algorithm seed varied

This is a starter experiment. After validating the pipeline, repeat it on your
chosen real dataset and document the dataset and preprocessing decisions.
"""

from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 2026
TRAIN_SIZES = [250, 500, 1000, 2000, 4000, 8000]
N_MODELS = 8
N_SEED_MODELS = 8
TEST_SIZE = 0.25
OUTPUT_DIR = Path(__file__).resolve().parent

def make_dataset():
    """Create one fixed dataset and one fixed holdout test set."""
    X, y = make_classification(
        n_samples=12000,
        n_features=20,
        n_informative=8,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        weights=[0.55, 0.45],
        class_sep=0.8,
        flip_y=0.03,
        random_state=RANDOM_STATE,
    )
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train_pool, X_test, y_train_pool, y_test

def make_model(family, seed):
    """Return a fresh estimator. Scaling is applied only to logistic regression."""
    if family == "Logistic Regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
        )
    if family == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=None, min_samples_leaf=2, random_state=seed
        )
    if family == "Histogram Gradient Boosting":
        return HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=15,
            l2_regularization=0.0, random_state=seed
        )
    raise ValueError(f"Unknown model family: {family}")

def pairwise_churn(predictions):
    """Mean fraction of test predictions that disagree across all model pairs."""
    values = []
    for i, j in combinations(range(len(predictions)), 2):
        values.append(np.mean(predictions[i] != predictions[j]))
    return float(np.mean(values)) if values else np.nan

def pairwise_error(predictions, y_true):
    """Mean error rate across independently trained models."""
    return float(np.mean([np.mean(p != y_true) for p in predictions]))

def safe_ratio(churn, error):
    """Avoid interpreting an unstable/undefined ratio when error is near zero."""
    return churn / error if error > 1e-8 else np.nan

def main():
    X_pool, X_test, y_pool, y_test = make_dataset()
    max_n = max(TRAIN_SIZES)
    if max_n > len(X_pool):
        raise ValueError(
            f"Largest requested n={max_n}, but training pool has {len(X_pool)} rows."
        )

    families = [
        "Logistic Regression",
        "Decision Tree",
        "Histogram Gradient Boosting",
    ]
    rows = []

    # DATA-DRIVEN CHURN:
    # Different training subsets; fixed algorithm seed for each family.
    for family in families:
        for n in TRAIN_SIZES:
            predictions, accuracies = [], []
            for replicate in range(N_MODELS):
                rng = np.random.default_rng(RANDOM_STATE + 1000 * replicate + n)
                indices = rng.choice(len(X_pool), size=n, replace=False)
                model = make_model(family, seed=RANDOM_STATE)
                model.fit(X_pool[indices], y_pool[indices])
                pred = model.predict(X_test)
                predictions.append(pred)
                accuracies.append(np.mean(pred == y_test))

            churn = pairwise_churn(predictions)
            error = pairwise_error(predictions, y_test)
            rows.append({
                "family": family,
                "n": n,
                "control": "data-driven",
                "mean_accuracy": float(np.mean(accuracies)),
                "mean_error_rate": error,
                "churn": churn,
                "churn_over_error": safe_ratio(churn, error),
                "n_models": N_MODELS,
            })

    # SEED-DRIVEN CHURN CONTROL:
    # Same training subset; vary algorithm seed. This isolates randomness in the
    # estimator as far as supported by each model family.
    rng = np.random.default_rng(RANDOM_STATE)
    fixed_indices = rng.choice(len(X_pool), size=max_n, replace=False)
    for family in families:
        for n in TRAIN_SIZES:
            # Nested fixed subsets: each smaller sample is contained in the larger
            # sample, keeping the control deterministic across n.
            indices = fixed_indices[:n]
            predictions, accuracies = [], []
            for replicate in range(N_SEED_MODELS):
                seed = RANDOM_STATE + 500 + replicate
                model = make_model(family, seed=seed)
                model.fit(X_pool[indices], y_pool[indices])
                pred = model.predict(X_test)
                predictions.append(pred)
                accuracies.append(np.mean(pred == y_test))

            churn = pairwise_churn(predictions)
            error = pairwise_error(predictions, y_test)
            rows.append({
                "family": family,
                "n": n,
                "control": "seed-driven",
                "mean_accuracy": float(np.mean(accuracies)),
                "mean_error_rate": error,
                "churn": churn,
                "churn_over_error": safe_ratio(churn, error),
                "n_models": N_SEED_MODELS,
            })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "q1_results.csv", index=False)

    summary = (
        results.groupby(["family", "control", "n"], as_index=False)
        .agg(
            mean_accuracy=("mean_accuracy", "mean"),
            mean_error_rate=("mean_error_rate", "mean"),
            churn=("churn", "mean"),
            churn_over_error=("churn_over_error", "mean"),
        )
    )
    summary.to_csv(OUTPUT_DIR / "q1_summary.csv", index=False)

    print("\nQ1 experiment complete.")
    print(f"Detailed results: {OUTPUT_DIR / 'q1_results.csv'}")
    print(f"Summary results:  {OUTPUT_DIR / 'q1_summary.csv'}")
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        "\nInterpretation reminder: churn/error is undefined when error is zero "
        "and can be unstable when error is very small. Compare raw churn and "
        "error rates too."
    )

if __name__ == "__main__":
    main()
