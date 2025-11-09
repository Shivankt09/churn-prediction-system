# src/churn/models/train_baseline.py
# ------------------------------------------------------------
# Baseline churn training script that:
# - Loads a CSV
# - Cleans & encodes features via sklearn Pipeline
# - Trains Logistic Regression
# - Computes metrics (ROC-AUC, PR-AUC, classification report)
# - Saves model + metrics + plots to reports/
#
# Run from project root, e.g.:
#   python src/churn/models/train_baseline.py \
#       --data "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv" \
#       --target Churn \
#       --id-cols customerID \
#       --threshold 0.5
# ------------------------------------------------------------

import argparse
from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# CLI ARGUMENTS
# -------------------------------
def parse_args():
    """
    Parse command-line arguments so you can experiment without changing code.
    """
    p = argparse.ArgumentParser(description="Train baseline churn model")

    # Required: where is your CSV?
    p.add_argument("--data", required=True, help="Path to raw CSV")

    # Your target column name (case-sensitive)
    p.add_argument("--target", default="Churn", help="Target column name")

    # Optional list of ID-like columns to drop (comma-separated)
    # Example: --id-cols customerID,account_id
    p.add_argument("--id-cols", default="", help="Comma-separated ID columns to drop")

    # For string targets, specify which string is the POSITIVE class (e.g., 'Yes')
    p.add_argument("--positive-label", default="yes", help="Positive class label if target is string")

    # Train/test split + randomness
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction (0-1)")
    p.add_argument("--random-state", type=int, default=42, help="Reproducibility")

    # Decision threshold for turning probabilities into 0/1 predictions
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (0-1)")

    return p.parse_args()


# -------------------------------
# FILESYSTEM HELPERS
# -------------------------------
def ensure_dirs():
    """Create output folders if they don't exist."""
    Path("models").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)


# -------------------------------
# PLOT HELPERS
# -------------------------------
def plot_and_save_roc(y_true, prob, out_path):
    """Save a ROC curve image."""
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_and_save_pr(y_true, prob, out_path):
    """Save a Precision–Recall curve image."""
    precision, recall, _ = precision_recall_curve(y_true, prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_and_save_confusion(y_true, pred, out_path):
    """Save a simple confusion matrix image with counts annotated."""
    cm = confusion_matrix(y_true, pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # annotate counts
    for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                         [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]]):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------
# MAIN
# -------------------------------
def main():
    args = parse_args()
    ensure_dirs()

    # 1) Load data ------------------------------------------------------------
    # Using pandas read_csv; if file has encoding issues, add encoding="utf-8" or "latin-1".
    df = pd.read_csv(args.data)

    # Optional: drop ID-like columns that provide no predictive value and could leak uniqueness.
    if args.id_cols.strip():
        to_drop = [c.strip() for c in args.id_cols.split(",") if c.strip()]
        df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    # 2) Normalize/prepare target to binary 0/1 --------------------------------
    # If the target is object/string, we map positives/negatives safely.
    # Example: "Yes"/"No", "True"/"False", "1"/"0", mixed case, extra spaces.
    target = args.target
    if df[target].dtype == "O":
        # Lowercase + strip for consistent matching
        t = df[target].astype(str).str.strip().str.lower()

        # Figure out the positive label (also lowercased)
        pos = str(args.positive-label if hasattr(args, "positive-label") else args.positive_label).lower()

        # Map common patterns; anything not matched will be left as-is
        mapping = {
            pos: 1,
            "yes": 1, "true": 1, "1": 1,
            "no": 0, "false": 0, "0": 0
        }
        mapped = t.map(mapping)

        # If some values couldn't be mapped, warn and try to coerce numerically
        if mapped.isna().any():
            # Attempt numeric coercion (e.g., "0"/"1")
            coerced = pd.to_numeric(t, errors="coerce")
            mapped = mapped.fillna(coerced)

        # Final safety: if still NA anywhere, raise a clear error so you can fix labels
        if mapped.isna().any():
            bad = sorted(df[target].astype(str).str.strip().str.lower().unique())
            raise ValueError(
                f"Target '{target}' contains unrecognized values: {bad}. "
                f"Use --positive-label to specify the positive class, "
                f"or pre-clean the target."
            )

        df[target] = mapped

    # Now we require target to be numeric 0/1
    y = df[target].astype(int)

    # 3) Features table --------------------------------------------------------
    X = df.drop(columns=[target])

    # Try converting "numeric-looking" object columns into numbers (e.g., 'tenure', 'MonthlyCharges' sometimes come as text)
    for col in X.columns:
        if X[col].dtype == "O":
            # If most values look numeric, coerce
            numeric_try = pd.to_numeric(X[col].astype(str).str.strip(), errors="coerce")
            # Heuristic: if >80% become numbers, treat as numeric
            if numeric_try.notna().mean() > 0.8:
                X[col] = numeric_try

    # Identify column types AFTER coercion
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    # 4) Build preprocessing + model pipeline ---------------------------------
    # - Numeric: median impute + standardize
    # - Categorical: most-frequent impute + one-hot encode
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # 5) Split, fit, predict ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y  # keep class balance roughly same across splits
    )

    model.fit(X_train, y_train)

    # Predicted probabilities for the positive class (churn = 1)
    prob = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to hard predictions via threshold
    threshold = float(args.threshold)
    pred = (prob >= threshold).astype(int)

    # 6) Metrics ---------------------------------------------------------------
    roc_auc = roc_auc_score(y_test, prob)
    pr_auc = average_precision_score(y_test, prob)
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print(f"Threshold used: {threshold}")
    print(classification_report(y_test, pred, digits=4))

    # 7) Save artifacts --------------------------------------------------------
    joblib.dump(model, Path("models") / "baseline_logreg.joblib")

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": threshold,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_label": args.positive_label
    }
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save plots
    plot_and_save_roc(y_test, prob, "reports/figures/roc_curve.png")
    plot_and_save_pr(y_test, prob, "reports/figures/pr_curve.png")
    plot_and_save_confusion(y_test, pred, "reports/figures/confusion_matrix.png")

    print("\nArtifacts saved:")
    print("  models/baseline_logreg.joblib")
    print("  reports/metrics.json")
    print("  reports/figures/roc_curve.png")
    print("  reports/figures/pr_curve.png")
    print("  reports/figures/confusion_matrix.png")


if __name__ == "__main__":
    main()
