"""Streamlit dashboard for the Employee Sentiment Analysis project.

Run locally with: streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis_pipeline import main

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
VIZ = ROOT / "visualization"

st.set_page_config(page_title="Employee Sentiment Analysis", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(OUTPUT / "sentiment_labeled.csv"),
        pd.read_csv(OUTPUT / "employee_monthly_scores.csv"),
        pd.read_csv(OUTPUT / "employee_rankings.csv"),
        pd.read_csv(OUTPUT / "flight_risk_employees.csv"),
    )


def show_chart(filename: str) -> None:
    chart = VIZ / filename
    if chart.exists():
        st.image(str(chart), use_container_width=True)


st.title("Employee Sentiment Analysis")
st.caption("Offline analysis dashboard — results are descriptive, not HR decisions.")

with st.sidebar:
    st.header("Controls")
    if st.button("Regenerate analysis", type="primary", use_container_width=True):
        with st.spinner("Running analysis pipeline..."):
            main()
        load_results.clear()
        st.success("Outputs regenerated.")
    st.divider()
    st.markdown("Run locally with:")
    st.code("streamlit run app.py", language="bash")

try:
    messages, monthly, rankings, flight_risk = load_results()
except FileNotFoundError:
    st.warning("No generated output files found. Use **Regenerate analysis** in the sidebar.")
    st.stop()

counts = messages["sentiment"].value_counts()
metrics = st.columns(4)
metrics[0].metric("Messages", f"{len(messages):,}")
metrics[1].metric("Positive", f"{int(counts.get('Positive', 0)):,}")
metrics[2].metric("Negative", f"{int(counts.get('Negative', 0)):,}")
metrics[3].metric("Flight-risk flags", len(flight_risk))

overview, monthly_tab, risk_tab, model_tab, data_tab = st.tabs(
    ["Overview", "Monthly Scores & Rankings", "Flight Risk", "Model", "Data"]
)

with overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment distribution")
        show_chart("sentiment_distribution.png")
    with right:
        st.subheader("Monthly sentiment trend")
        show_chart("monthly_sentiment_trend.png")
    st.subheader("Message volume over time")
    show_chart("monthly_message_volume.png")

with monthly_tab:
    months = sorted(monthly["month"].dropna().unique())
    selected_month = st.selectbox("Select month", months)
    scores = monthly[monthly["month"] == selected_month].sort_values(
        ["sentiment_score", "employee"], ascending=[False, True]
    )
    st.subheader(f"Employee monthly scores — {selected_month}")
    st.dataframe(scores, use_container_width=True, hide_index=True)
    positive, negative = st.columns(2)
    month_rankings = rankings[rankings["month"] == selected_month]
    with positive:
        st.caption("Top positive")
        st.dataframe(month_rankings[month_rankings["ranking_type"] == "Top Positive"], use_container_width=True, hide_index=True)
    with negative:
        st.caption("Top negative")
        st.dataframe(month_rankings[month_rankings["ranking_type"] == "Top Negative"], use_container_width=True, hide_index=True)
    show_chart("employee_rankings.png")

with risk_tab:
    st.subheader("Rolling 30-day flight-risk flags")
    st.write("A flag requires at least four negative messages in any rolling 30-day window.")
    if flight_risk.empty:
        st.success("No employee met the stated threshold in this dataset.")
    else:
        st.dataframe(flight_risk, use_container_width=True, hide_index=True)
    show_chart("flight_risk.png")

with model_tab:
    st.subheader("Leakage-safe linear regression")
    st.write("The model uses message volume, average message and word length, active days, and prior-month sentiment. It excludes sentiment class counts and percentages because they would reconstruct the target score.")
    left, right = st.columns(2)
    with left:
        show_chart("actual_vs_predicted.png")
    with right:
        show_chart("residual_plot.png")
    st.info("Latest pipeline metrics: MAE 1.067 · MSE 1.930 · RMSE 1.389 · R² 0.388")

with data_tab:
    st.subheader("Labelled messages")
    labels = st.multiselect("Filter sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
    filtered = messages[messages["sentiment"].isin(labels)]
    st.dataframe(filtered[["employee", "date", "message", "sentiment", "sentiment_score"]], use_container_width=True, hide_index=True, height=450)
    st.download_button("Download filtered messages", filtered.to_csv(index=False).encode("utf-8"), "filtered_sentiment_messages.csv", "text/csv")
