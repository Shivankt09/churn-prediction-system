import streamlit as st
import pandas as pd
import joblib
from churn.data.loader import load_csv

MODEL_PATH = "models/baseline_logreg.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def main():
    st.title("📊 Customer Churn Prediction App")

    st.write("Upload customer data and get churn probabilities.")

    # Load model
    model = load_model()

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())

        # Predict churn
        probs = model.predict_proba(df)[:, 1]
        df["Churn_Probability"] = probs

        st.write("### Churn Predictions:")
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions",
            csv,
            "churn_predictions.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == "__main__":
    main()
