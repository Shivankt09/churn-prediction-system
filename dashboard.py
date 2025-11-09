# src/churn/app/dashboard.py
import json
import streamlit as st
import pandas as pd
from pathlib import Path

METRICS_PATH = Path("reports/metrics.json")
ROC_PATH = Path("reports/figures/roc_curve.png")
PR_PATH = Path("reports/figures/pr_curve.png")
CM_PATH = Path("reports/figures/confusion_matrix.png")

def main():
    st.title("📈 Churn Model Evaluation Dashboard")

    # ---- Metrics ----
    st.subheader("Model Metrics")
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r") as f:
            metrics_dict = json.load(f)

        # Show raw JSON for transparency
        st.json(metrics_dict)

        # Also show as a one-row table
        metrics_df = pd.DataFrame([metrics_dict])
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("No metrics found yet. Train the model first to create 'reports/metrics.json'.")

    # ---- Figures ----
    st.subheader("ROC Curve")
    if ROC_PATH.exists():
        st.image(str(ROC_PATH))
    else:
        st.info("ROC curve not found. It will appear after training.")

    st.subheader("Precision–Recall Curve")
    if PR_PATH.exists():
        st.image(str(PR_PATH))
    else:
        st.info("PR curve not found. It will appear after training.")

    st.subheader("Confusion Matrix")
    if CM_PATH.exists():
        st.image(str(CM_PATH))
    else:
        st.info("Confusion matrix not found. It will appear after training.")

if __name__ == "__main__":
    main()
