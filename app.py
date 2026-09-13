import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔍 AI Anomaly Detection for Financial Transactions")

uploaded_file = st.file_uploader("Upload your transactions CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.write(df.head())

    features = [
        "TransactionAmount",
        "TransactionHour",
        "Account_Frequency",
        "Device_Frequency",
        "IP_Frequency",
        "Merchant_Frequency",
        "LocationDistanceKM"
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    iso.fit(X_scaled)

    df["is_anomaly"] = iso.predict(X_scaled)
    df["anomaly_score"] = iso.decision_function(X_scaled)

    st.subheader("🚨 Detected Anomalies")
    anomalies = df[df["is_anomaly"] == -1]
    st.write(anomalies)

    st.subheader("📊 Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="TransactionAmount",
        y="Account_Frequency",
        hue="is_anomaly",
        palette={1: "blue", -1: "red"},
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("⬇ Download Results")
    st.download_button(
        label="Download anomaly results as CSV",
        data=df.to_csv(index=False),
        file_name="anomaly_results.csv",
        mime="text/csv"
    )
