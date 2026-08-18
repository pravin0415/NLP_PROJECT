"""Reproducible employee-message sentiment analysis pipeline.

Run from this project's directory: python analysis_pipeline.py
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "test.csv"
DATA_DIR, VIZ_DIR, OUT_DIR = ROOT / "data", ROOT / "visualization", ROOT / "output"
SENTIMENT_VALUES = {"Positive": 1, "Neutral": 0, "Negative": -1}

# A transparent workplace-oriented lexicon.  Phrase matching precedes words;
# negation reverses the next sentiment-bearing term. It is reproducible offline.
POSITIVE = set("thank thanks appreciate appreciated helpful pleased please good great excellent happy glad congratulations congrats success successful welcome agree approved benefit beneficial support supported exciting enjoy fantastic wonderful perfect resolved solution progress opportunity".split())
NEGATIVE = set("unfortunately unable problem problems issue issues concern concerns delay delayed fail failed failing error errors risk urgent disappointed disappointing frustrated frustration complaint complain difficult difficulty worst wrong bad poor decline declined loss losses conflict regret unhappy unacceptable cancel cancelled denied rejection".split())
POS_PHRASES = ("thank you", "well done", "looking forward", "good news", "great job", "please let me know")
NEG_PHRASES = ("does not work", "not going to work", "cannot", "can't", "unable to", "not available", "no longer", "failed to", "do not", "doesn't")
NEGATORS = {"not", "no", "never", "without", "hardly", "cannot", "cant", "can't", "isnt", "isn't", "wasnt", "wasn't", "dont", "don't", "doesnt", "doesn't"}


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map common source-column variants to analytical fields."""
    candidates = {
        "employee": ("from", "employee", "employee_id", "sender", "author", "email"),
        "message": ("body", "message", "text", "content", "email_body"),
        "date": ("date", "datetime", "timestamp", "sent_at", "time"),
    }
    normalized = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    result = {}
    for field, choices in candidates.items():
        found = next((normalized.get(re.sub(r"[^a-z0-9]", "", x)) for x in choices if re.sub(r"[^a-z0-9]", "", x) in normalized), None)
        if not found:
            raise ValueError(f"Could not identify required {field} column from {list(df.columns)}")
        result[field] = found
    return result


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("=20", " ").replace("=01", " ")).strip()


def classify_sentiment(text: str) -> str:
    """Rule-based, phrase-aware three-class classifier for unlabeled messages."""
    lower = text.lower()
    score = sum(lower.count(p) for p in POS_PHRASES) - sum(lower.count(p) for p in NEG_PHRASES)
    tokens = re.findall(r"[a-z']+", lower)
    for i, token in enumerate(tokens):
        value = 1 if token in POSITIVE else -1 if token in NEGATIVE else 0
        if value and any(t in NEGATORS for t in tokens[max(0, i - 3):i]):
            value *= -1
        score += value
    # A small threshold avoids calling ordinary scheduling/transactional mail positive.
    return "Positive" if score >= 2 else "Negative" if score <= -2 else "Neutral"


def load_and_prepare() -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    raw = pd.read_csv(INPUT)
    columns = detect_columns(raw)
    df = raw.copy()
    quality = {
        "rows": len(df), "missing_employee_ids": int(df[columns["employee"]].isna().sum()),
        "missing_messages": int(df[columns["message"]].isna().sum()), "duplicate_rows": int(df.duplicated().sum()),
    }
    df["employee"] = df[columns["employee"]].fillna("Unknown").astype(str).str.strip()
    df["message"] = df[columns["message"]].fillna("").map(clean_text)
    df["date"] = pd.to_datetime(df[columns["date"]], errors="coerce")
    quality["invalid_dates"] = int(df["date"].isna().sum())
    quality["empty_messages"] = int(df["message"].eq("").sum())
    # Invalid dates cannot contribute to time-dependent analysis, but are retained in labelled output.
    df["sentiment"] = df["message"].map(classify_sentiment)
    df["sentiment_score"] = df["sentiment"].map(SENTIMENT_VALUES).astype("int8")
    df["message_length"] = df["message"].str.len()
    df["word_count"] = df["message"].str.findall(r"\b\w+\b").str.len()
    df["month"] = df["date"].dt.to_period("M").astype("string")
    df["year"] = df["date"].dt.year.astype("Int64")
    df["year_month"] = df["month"]
    df["day"] = df["date"].dt.day.astype("Int64")
    df["weekday"] = df["date"].dt.day_name()
    return df, columns, quality


def save_plot(name: str) -> None:
    plt.tight_layout(); plt.savefig(VIZ_DIR / name, dpi=160, bbox_inches="tight"); plt.close()


def make_visualizations(df: pd.DataFrame, monthly: pd.DataFrame, rankings: pd.DataFrame, risk: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment"].value_counts().reindex(order, fill_value=0)
    ax = counts.plot.bar(color=["#2a9d8f", "#8d99ae", "#e76f51"]); ax.set(title="Sentiment Distribution", xlabel="Sentiment", ylabel="Messages"); save_plot("sentiment_distribution.png")
    ax = (counts / len(df) * 100).plot.bar(color=["#2a9d8f", "#8d99ae", "#e76f51"]); ax.set(title="Sentiment Percentage Distribution", xlabel="Sentiment", ylabel="Percentage of messages"); save_plot("sentiment_percentage.png")
    dated = df.dropna(subset=["date"]).copy()
    volume = dated.groupby("month", observed=True).size()
    ax = volume.plot(marker="o"); ax.set(title="Monthly Message Volume", xlabel="Month", ylabel="Messages"); plt.xticks(rotation=45); save_plot("monthly_message_volume.png")
    trend = pd.crosstab(dated["month"], dated["sentiment"]).reindex(columns=order, fill_value=0)
    ax = trend.plot(marker="o"); ax.set(title="Monthly Sentiment Trend", xlabel="Month", ylabel="Messages"); plt.xticks(rotation=45); save_plot("monthly_sentiment_trend.png")
    ax = sns.histplot(df["message_length"], bins=40, color="#457b9d"); ax.set(title="Message Length Distribution", xlabel="Characters", ylabel="Messages"); save_plot("message_length_distribution.png")
    employee_counts = df["employee"].value_counts().head(20).sort_values()
    ax = employee_counts.plot.barh(color="#457b9d"); ax.set(title="Top 20 Employees by Message Volume", xlabel="Messages", ylabel="Employee"); save_plot("employee_message_frequency.png")
    rank_top = rankings[rankings["ranking_type"] == "Top Positive"].head(15)
    ax = sns.barplot(rank_top, x="sentiment_score", y="employee", hue="month", dodge=False); ax.set(title="Top Positive Monthly Employee Rankings", xlabel="Monthly sentiment score", ylabel="Employee"); save_plot("employee_rankings.png")
    if risk.empty:
        fig, ax = plt.subplots(); ax.text(.5, .5, "No employees met the rolling 30-day threshold", ha="center", va="center"); ax.set_axis_off(); plt.title("Flight-Risk Employees")
    else:
        ax = sns.barplot(risk.sort_values("negative_message_count"), x="negative_message_count", y="employee", color="#e76f51"); ax.set(title="Flight-Risk Employees: Negative Messages in Qualifying Window", xlabel="Negative messages", ylabel="Employee")
    save_plot("flight_risk.png")


def monthly_scores(df: pd.DataFrame) -> pd.DataFrame:
    dated = df.dropna(subset=["date"]).copy()
    grouped = dated.groupby(["employee", "month"], as_index=False, observed=True).agg(
        total_messages=("message", "size"), sentiment_score=("sentiment_score", "sum"),
        average_message_length=("message_length", "mean"), average_word_count=("word_count", "mean"),
        active_days=("date", lambda x: x.dt.normalize().nunique()),
    )
    pivot = pd.crosstab([dated["employee"], dated["month"]], dated["sentiment"]).reindex(columns=["Positive", "Negative", "Neutral"], fill_value=0).reset_index()
    pivot.columns = ["employee", "month", "positive_count", "negative_count", "neutral_count"]
    result = grouped.merge(pivot, on=["employee", "month"], validate="one_to_one")
    for s in ("positive", "negative", "neutral"):
        result[f"{s}_percentage"] = result[f"{s}_count"] / result["total_messages"] * 100
    return result.sort_values(["month", "employee"]).reset_index(drop=True)


def rank_employees(monthly: pd.DataFrame) -> pd.DataFrame:
    records = []
    for month, group in monthly.groupby("month", sort=True, observed=True):
        for label, asc in [("Top Positive", False), ("Top Negative", True)]:
            selected = group.sort_values(["sentiment_score", "employee"], ascending=[asc, True]).head(3)
            for rank, row in enumerate(selected.itertuples(index=False), 1):
                records.append({"month": month, "ranking_type": label, "rank": rank, "employee": row.employee, "sentiment_score": row.sentiment_score})
    return pd.DataFrame(records)


def find_flight_risk(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    neg = df[(df["sentiment"] == "Negative") & df["date"].notna()].sort_values(["employee", "date"])
    for employee, group in neg.groupby("employee", sort=True):
        dates = group["date"].reset_index(drop=True)
        left = 0
        for right, end in enumerate(dates):
            while end - dates.iloc[left] > pd.Timedelta(days=30): left += 1
            n = right - left + 1
            if n >= 4:
                rows.append({"employee": employee, "first_risk_date": end.date().isoformat(), "window_start": dates.iloc[left].date().isoformat(), "window_end": end.date().isoformat(), "negative_message_count": n})
                break
    return pd.DataFrame(rows, columns=["employee", "first_risk_date", "window_start", "window_end", "negative_message_count"])


def model_monthly_sentiment(monthly: pd.DataFrame) -> dict:
    model_df = monthly.sort_values(["employee", "month"]).copy()
    model_df["previous_sentiment_score"] = model_df.groupby("employee")["sentiment_score"].shift(1)
    # These features do not algebraically reconstruct the target; the lag is historical only.
    features = ["total_messages", "average_message_length", "average_word_count", "active_days", "previous_sentiment_score"]
    model_df = model_df.dropna(subset=features + ["sentiment_score"]).sort_values("month")
    if len(model_df) < 8 or model_df["month"].nunique() < 2:
        return {"available": False, "reason": "Insufficient chronological employee-month records after creating lag features."}
    cutoff = max(1, int(len(model_df) * .8))
    train, test = model_df.iloc[:cutoff], model_df.iloc[cutoff:]
    if test.empty: return {"available": False, "reason": "No held-out chronological test observations."}
    model = LinearRegression().fit(train[features], train["sentiment_score"])
    pred = model.predict(test[features]); actual = test["sentiment_score"].to_numpy()
    metrics = {"MAE": mean_absolute_error(actual, pred), "MSE": mean_squared_error(actual, pred), "RMSE": float(np.sqrt(mean_squared_error(actual, pred))), "R2": r2_score(actual, pred)}
    fig, ax = plt.subplots(); ax.scatter(actual, pred, alpha=.7); lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max()); ax.plot([lo, hi], [lo, hi], "r--"); ax.set(title="Actual vs Predicted Monthly Sentiment", xlabel="Actual score", ylabel="Predicted score"); save_plot("actual_vs_predicted.png")
    fig, ax = plt.subplots(); ax.scatter(pred, actual - pred, alpha=.7); ax.axhline(0, color="r", ls="--"); ax.set(title="Regression Residual Plot", xlabel="Predicted score", ylabel="Residual (actual − predicted)"); save_plot("residual_plot.png")
    return {"available": True, "features": features, "metrics": metrics, "coefficients": dict(zip(features, model.coef_)), "train_rows": len(train), "test_rows": len(test)}


def validate(df: pd.DataFrame, monthly: pd.DataFrame, rankings: pd.DataFrame, risk: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "valid_sentiment_labels": set(df.sentiment.unique()) <= set(SENTIMENT_VALUES),
        "valid_sentiment_scores": set(df.sentiment_score.unique()) <= {-1, 0, 1},
        "monthly_score_reconciles": bool((monthly.sentiment_score == monthly.positive_count - monthly.negative_count).all()),
        "unique_employee_month": not monthly.duplicated(["employee", "month"]).any(),
        "ranking_matches_scores": all((g.sort_values("rank").sentiment_score.is_monotonic_decreasing if name == "Top Positive" else g.sort_values("rank").sentiment_score.is_monotonic_increasing) for (_, name), g in rankings.groupby(["month", "ranking_type"])),
        "flight_risk_windows_valid": all(r.negative_message_count >= 4 for r in risk.itertuples()),
    }
    return checks


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without adding a tabulate dependency."""
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def write_readme(columns: dict[str, str], quality: dict[str, int], df: pd.DataFrame, monthly: pd.DataFrame, rankings: pd.DataFrame, risk: pd.DataFrame, model: dict, checks: dict) -> None:
    top_pos = markdown_table(rankings[rankings.ranking_type.eq("Top Positive")].head(3))
    top_neg = markdown_table(rankings[rankings.ranking_type.eq("Top Negative")].head(3))
    risks = "No employee met the stated threshold." if risk.empty else markdown_table(risk)
    model_text = model.get("reason", "; ".join(f"{k}={v:.3f}" for k,v in model["metrics"].items()))
    (ROOT / "README.md").write_text(f"""# Employee Sentiment Analysis

## Project Overview
An offline, reproducible analysis of {len(df):,} employee messages. Source fields detected: employee=`{columns['employee']}`, message=`{columns['message']}`, date=`{columns['date']}`. The source contains {quality['duplicate_rows']} duplicate rows, {quality['invalid_dates']} invalid dates, and {quality['empty_messages']} empty messages.

## Methodology
Messages are cleaned and assigned Positive, Neutral, or Negative by a transparent, phrase-aware workplace lexicon with local negation handling. This is appropriate for an unlabeled data set and runs entirely offline; it should be manually validated before being used for HR decisions. Scores are +1, 0, and -1. Monthly totals reset each calendar month.

## EDA and Results
Sentiment counts: {df.sentiment.value_counts().to_dict()}. The processed data, plots, and all output tables are generated by `analysis_pipeline.py`.

### First available monthly rankings
Top positive:

{top_pos}

Top negative:

{top_neg}

## Flight Risk Detection
An employee qualifies only when four or more negative messages occur in a rolling inclusive 30-day window. The first qualifying window per employee is retained.

{risks}

## Predictive Modeling
Chronological 80/20 employee-month split, LinearRegression. Features are message volume, average message/word length, active days, and prior-month sentiment. Direct sentiment class counts and percentages are excluded to prevent target leakage. {model_text}

## Validation
 {markdown_table(pd.DataFrame({"check": list(checks), "passed": list(checks.values())}))}

## Installation and Run
```bash
pip install -r requirements.txt
python analysis_pipeline.py
```

## Project Structure
`test.csv` is the source input; `data/` stores processed data, `output/` stores tables, and `visualization/` stores PNG charts. `Employee_Sentiment_Analysis.ipynb` is a documented notebook front end.

## Limitations
Rule-based sentiment can miss context, sarcasm, and organizational jargon. Sentiment does not establish employee performance, intent, or likelihood of departure; flight-risk output is a project-rule flag, not an HR determination. Regression coefficients are associational and the small, time-specific dataset limits generalization.
""", encoding="utf-8")


def write_notebook() -> None:
    import nbformat as nbf
    nb = nbf.v4.new_notebook(); cells = []
    cells.append(nbf.v4.new_markdown_cell("# Employee Sentiment Analysis\n\nThis reproducible notebook runs the accompanying local pipeline and documents its methodology."))
    cells.append(nbf.v4.new_markdown_cell("## 1. Project Overview\nThe source is unlabeled, so sentiment uses the transparent offline phrase-aware workplace lexicon defined in `analysis_pipeline.py`. Labels are Positive (+1), Neutral (0), and Negative (-1)."))
    cells.append(nbf.v4.new_code_cell("from pathlib import Path\nimport sys\nROOT = Path.cwd()\nif not (ROOT / 'analysis_pipeline.py').exists(): ROOT = ROOT / 'employee-sentiment-analysis'\nsys.path.insert(0, str(ROOT))\nfrom analysis_pipeline import main\nresults = main()\nresults"))
    for title, body in [("2. Data Quality Assessment", "The pipeline detects source columns, validates missing IDs/messages/dates and duplicates, and creates a compact data dictionary in the console output."), ("3. Sentiment Labeling", "Phrase signals, workplace vocabulary, and local negation are evaluated without external data or row-by-row model downloads. The method is reproducible but must not be used as an HR decision system without review."), ("4. Exploratory Data Analysis", "Charts are saved under `visualization/`. Inspect the generated PNGs for sentiment, temporal volume, lengths, and employee message frequency."), ("5. Monthly Employee Sentiment Score", "Aggregation is employee + calendar month, validated as positive_count − negative_count. Scores never carry across months."), ("6. Rankings and Flight Risk", "Monthly rankings use sentiment score then alphabetical employee tie-breaker. Flight risk uses a two-pointer rolling 30-day negative-message window, never monthly totals."), ("7. Predictive Modeling", "A chronological split predicts monthly score without positive/negative counts or class percentages, preventing direct target reconstruction."), ("8. Limitations", "This is a rule-based text analysis of historical messages, not evidence of employee intent or a basis for personnel decisions.")]:
        cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{body}\n\n### Observation\n\nRun the pipeline to regenerate observations from the current local data."))
    nb["cells"] = cells; nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nbf.write(nb, ROOT / "Employee_Sentiment_Analysis.ipynb")


def main() -> dict:
    for directory in (DATA_DIR, VIZ_DIR, OUT_DIR): directory.mkdir(parents=True, exist_ok=True)
    df, columns, quality = load_and_prepare(); monthly = monthly_scores(df); rankings = rank_employees(monthly); risk = find_flight_risk(df)
    make_visualizations(df, monthly, rankings, risk); model = model_monthly_sentiment(monthly); checks = validate(df, monthly, rankings, risk)
    df.to_csv(DATA_DIR / "processed_data.csv", index=False); df.to_csv(OUT_DIR / "sentiment_labeled.csv", index=False); monthly.to_csv(OUT_DIR / "employee_monthly_scores.csv", index=False); rankings.to_csv(OUT_DIR / "employee_rankings.csv", index=False); risk.to_csv(OUT_DIR / "flight_risk_employees.csv", index=False)
    write_readme(columns, quality, df, monthly, rankings, risk, model, checks); write_notebook()
    result = {"quality": quality, "sentiment_counts": df.sentiment.value_counts().to_dict(), "employee_months": len(monthly), "flight_risk_employees": len(risk), "model": model, "validation": checks}
    print(result); return result


if __name__ == "__main__": main()
