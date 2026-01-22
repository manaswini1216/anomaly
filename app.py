import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Anomaly Detection Model Comparison",
    layout="wide"
)

st.title("Anomaly Detection Model Comparison Dashboard")
st.markdown(
    """
    Comparative analysis of **four anomaly detection models**  
    evaluated on industrial time-series datasets.
    """
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_metrics():
    lstm_unsup = pd.read_csv(
        os.path.join(BASE_DIR, "result_lstm_us", "fault_metrics.csv")
    )
    lstm_sup = pd.read_csv(
        os.path.join(BASE_DIR, "result_lstm_s", "supervised_test_metrics.csv")
    )
    mset = pd.read_csv(
        os.path.join(BASE_DIR, "result_mset", "mset_metrics.csv")
    )
    iso = pd.read_csv(
        os.path.join(BASE_DIR, "result_isolation_forest", "isolation_forest_metrics.csv")
    )

    lstm_unsup["Model"] = "LSTM Unsupervised"
    lstm_sup["Model"] = "LSTM Supervised"
    mset["Model"] = "MSET"
    iso["Model"] = "Isolation Forest"

    return lstm_unsup, lstm_sup, mset, iso


lstm_unsup, lstm_sup, mset, iso = load_metrics()


def normalize(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df

lstm_unsup = normalize(lstm_unsup)
lstm_sup   = normalize(lstm_sup)
mset       = normalize(mset)
iso        = normalize(iso)


anomaly_df = pd.concat(
    [
        lstm_unsup[["folder", "file", "anomaly_ratio", "model"]],
        mset[["folder", "file", "anomaly_ratio", "model"]],
        iso[["folder", "file", "anomaly_ratio", "model"]],
    ],
    ignore_index=True
)

st.sidebar.header("Filters")

model_filter = st.sidebar.multiselect(
    "Select Models",
    anomaly_df["model"].unique(),
    default=anomaly_df["model"].unique()
)

filtered_df = anomaly_df[anomaly_df["model"].isin(model_filter)]


st.header("Dataset Coverage")

coverage = (
    filtered_df
    .groupby("model")
    .size()
    .reset_index(name="Number of Files")
)

st.dataframe(coverage, use_container_width=True)

st.header("Average Anomaly Ratio")

avg_ratio = (
    filtered_df
    .groupby("model")["anomaly_ratio"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=avg_ratio, x="model", y="anomaly_ratio", ax=ax)
ax.set_ylabel("Average Anomaly Ratio")
ax.set_xlabel("Model")
st.pyplot(fig)

st.header("Anomaly Ratio Distribution")

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(
    data=filtered_df,
    x="model",
    y="anomaly_ratio",
    ax=ax
)
ax.set_ylabel("Anomaly Ratio")
ax.set_xlabel("Model")
st.pyplot(fig)

st.header("Model Stability")

stability = (
    filtered_df
    .groupby("model")["anomaly_ratio"]
    .std()
    .reset_index(name="Standard Deviation")
)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=stability, x="model", y="Standard Deviation", ax=ax)
ax.set_ylabel("Std Dev of Anomaly Ratio")
ax.set_xlabel("Model")
st.pyplot(fig)

st.markdown(
    """
    **Lower standard deviation → more stable detection**  
    **Higher deviation → sensitive to operating conditions**
    """
)

st.header("LSTM Supervised Classification Metrics")

st.dataframe(
    lstm_sup[["folder", "file", "accuracy", "precision", "recall", "f1"]],
    use_container_width=True
)

st.markdown("---")
st.markdown(
    "End-to-end anomaly detection benchmarking using classical, deep learning, and hybrid models."
)
