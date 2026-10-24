"""
Streamlit dashboard for the Financial Crime Detection &
Risk Scoring System.

Run locally:
    streamlit run streamlit_app.py

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (  # noqa: E402
    FEATURE_IMPORTANCE_REPORT,
    GLOBAL_SHAP_REPORT,
    HIGH_RISK_MAX,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
    METRICS_REPORT,
    MODEL_BUNDLE,
    PR_CURVE_REPORT,
    ROC_CURVE_REPORT,
    SAMPLE_TRANSACTIONS,
    TARGET,
)
from config.constants import APP_TITLE, SIDEBAR_TITLE, TRANSACTION_ID  # noqa: E402
from src.predict import FraudRiskPredictor  # noqa: E402
from src.report_generator import (  # noqa: E402
    build_batch_csv,
    build_transaction_report,
)
from src.risk_scoring import (  # noqa: E402
    CATEGORY_COLORS,
    RECOMMENDATION_DETAILS,
    RISK_CATEGORIES,
)

# Chart colors (validated reference palette). Risk categories use the
# status palette in CATEGORY_COLORS and are always labelled by name.
SERIES_BLUE = "#2A78D6"
INCREASES_RISK = "#E34948"
DECREASES_RISK = "#2A78D6"
REFERENCE_GRAY = "#898781"

MAX_BATCH_ROWS = 50_000

CATEGORY_ICONS = {
    "Low": "✅",
    "Medium": "👀",
    "High": "⚠️",
    "Critical": "🚨",
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------------- loading


@st.cache_resource(show_spinner="Loading model...")
def load_predictor() -> FraudRiskPredictor:
    return FraudRiskPredictor.load(MODEL_BUNDLE)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_REPORT.read_text())


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def style_chart(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(namelength=-1),
    )
    return fig


def category_badge(category: str) -> str:
    return f"{CATEGORY_ICONS[category]} {category}"


# ---------------------------------------------------------------- pages


def page_overview(metrics: dict, predictor: FraudRiskPredictor) -> None:
    st.title(APP_TITLE)
    st.caption(
        "Stacked ensemble (LightGBM + XGBoost + CatBoost → Logistic "
        "Regression) trained on the IEEE-CIS Fraud Detection dataset. "
        "All figures below are measured on the most recent transactions, "
        "which the model never saw during training."
    )

    test = metrics["test_metrics"]

    cols = st.columns(5)
    cols[0].metric("PR-AUC", f"{test['pr_auc']:.3f}",
                   help="Area under the precision-recall curve. "
                        f"A random model would score {test['fraud_rate']:.3f}.")
    cols[1].metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    cols[2].metric("Precision", f"{test['precision']:.1%}",
                   help="Of the transactions flagged, the share that were fraud.")
    cols[3].metric("Recall", f"{test['recall']:.1%}",
                   help="Of all fraud, the share the model flagged.")
    cols[4].metric("Decision threshold", f"{metrics['threshold']:.3f}",
                   help="Probability above which a transaction is flagged "
                        "(maps to risk score 50). Tuned to maximise F1.")

    st.subheader("Risk categories on unseen transactions")
    st.caption(
        "If the scoring is useful, the observed fraud rate should rise "
        "steeply from Low to Critical - and most fraud should land in "
        "the top bands."
    )

    bands = pd.DataFrame(metrics["risk_bands"])

    left, right = st.columns([3, 2])

    with left:
        fig = go.Figure(
            go.Bar(
                x=bands["category"],
                y=bands["fraud_rate"],
                marker_color=[CATEGORY_COLORS[c] for c in bands["category"]],
                text=[f"{v:.1%}" for v in bands["fraud_rate"]],
                textposition="outside",
                customdata=bands[["transactions", "fraud_cases"]],
                hovertemplate=(
                    "<b>%{x}</b><br>Fraud rate: %{y:.1%}<br>"
                    "Transactions: %{customdata[0]:,}<br>"
                    "Fraud cases: %{customdata[1]:,}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title="Observed fraud rate by risk category",
            yaxis_tickformat=".0%",
            yaxis_title="Fraud rate",
            showlegend=False,
        )
        st.plotly_chart(style_chart(fig), width="stretch")

    with right:
        table = bands.assign(
            category=bands["category"].map(category_badge),
            share=bands["share"].map("{:.1%}".format),
            fraud_rate=bands["fraud_rate"].map("{:.1%}".format),
            fraud_captured=bands["fraud_captured"].map("{:.1%}".format),
        ).rename(
            columns={
                "category": "Category",
                "transactions": "Transactions",
                "share": "Share",
                "fraud_cases": "Fraud cases",
                "fraud_rate": "Fraud rate",
                "fraud_captured": "Share of all fraud",
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.markdown(
            f"**Score bands:** Low 0–{LOW_RISK_MAX} · Medium "
            f"{LOW_RISK_MAX + 1}–{MEDIUM_RISK_MAX} · High "
            f"{MEDIUM_RISK_MAX + 1}–{HIGH_RISK_MAX} · Critical "
            f"{HIGH_RISK_MAX + 1}–100"
        )

    st.subheader("Model performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        pr = load_csv(str(PR_CURVE_REPORT))
        fig = go.Figure(
            go.Scatter(
                x=pr["recall"], y=pr["precision"], mode="lines",
                line=dict(color=SERIES_BLUE, width=2),
                hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>",
            )
        )
        fig.add_hline(y=test["fraud_rate"], line_dash="dot",
                      line_color=REFERENCE_GRAY,
                      annotation_text="random model",
                      annotation_position="top right")
        fig.update_layout(title="Precision-recall curve",
                          xaxis_title="Recall", yaxis_title="Precision",
                          hovermode="x unified")
        st.plotly_chart(style_chart(fig), width="stretch")

    with c2:
        roc = load_csv(str(ROC_CURVE_REPORT))
        fig = go.Figure(
            go.Scatter(
                x=roc["fpr"], y=roc["tpr"], mode="lines",
                line=dict(color=SERIES_BLUE, width=2),
                hovertemplate="FPR %{x:.3f}<br>TPR %{y:.2f}<extra></extra>",
            )
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color=REFERENCE_GRAY, dash="dot"))
        fig.update_layout(title="ROC curve",
                          xaxis_title="False positive rate",
                          yaxis_title="True positive rate",
                          hovermode="x unified")
        st.plotly_chart(style_chart(fig), width="stretch")

    with c3:
        cm = test["confusion_matrix"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=["Predicted legitimate", "Predicted fraud"],
                y=["Actual legitimate", "Actual fraud"],
                colorscale=[[0, "#CDE2FB"], [1, "#184F95"]],
                showscale=False,
                text=[[f"{v:,}" for v in row] for row in z],
                texttemplate="%{text}",
                hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>",
                xgap=2, ygap=2,
            )
        )
        fig.update_layout(title="Confusion matrix (at threshold)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_chart(fig), width="stretch")

    with st.expander("All holdout metrics and base-model comparison"):
        detail = {
            "Precision": test["precision"],
            "Recall": test["recall"],
            "F1-score": test["f1"],
            "PR-AUC": test["pr_auc"],
            "ROC-AUC": test["roc_auc"],
            "Balanced accuracy": test["balanced_accuracy"],
            "Specificity": test["specificity"],
            "False positive rate": test["false_positive_rate"],
            "False negative rate": test["false_negative_rate"],
        }
        st.dataframe(
            pd.DataFrame({"Metric": detail.keys(),
                          "Value": [f"{v:.4f}" for v in detail.values()]}),
            hide_index=True,
        )
        comparison = pd.DataFrame(metrics["model_comparison"]).T
        comparison.index.name = "Model"
        st.dataframe(comparison.style.format("{:.4f}"))
        sizes = metrics["split_sizes"]
        meta = predictor.metadata
        st.caption(
            f"Chronological split - train: {sizes['train']:,} · threshold "
            f"tuning: {sizes['threshold']:,} · test: {sizes['test']:,} "
            f"transactions. Model trained {meta.get('trained_at', 'n/a')}."
        )


def risk_gauge(score: int, category: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix=" / 100"),
            title=dict(text=category_badge(category)),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="#0B0B0B", thickness=0.25),
                steps=[
                    dict(range=[0, LOW_RISK_MAX + 1], color=CATEGORY_COLORS["Low"]),
                    dict(range=[LOW_RISK_MAX + 1, MEDIUM_RISK_MAX + 1],
                         color=CATEGORY_COLORS["Medium"]),
                    dict(range=[MEDIUM_RISK_MAX + 1, HIGH_RISK_MAX + 1],
                         color=CATEGORY_COLORS["High"]),
                    dict(range=[HIGH_RISK_MAX + 1, 100],
                         color=CATEGORY_COLORS["Critical"]),
                ],
            ),
        )
    )
    style_chart(fig, height=280)
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=10))
    return fig


def explanation_chart(explanation: pd.DataFrame) -> go.Figure:
    ordered = explanation.iloc[::-1]
    labels = [
        f"{row.feature} = {row.value:.4g}" if isinstance(row.value, float)
        else f"{row.feature} = {row.value}"
        for row in ordered.itertuples()
    ]
    fig = go.Figure(
        go.Bar(
            x=ordered["contribution"],
            y=labels,
            orientation="h",
            marker_color=[
                INCREASES_RISK if v >= 0 else DECREASES_RISK
                for v in ordered["contribution"]
            ],
            customdata=ordered["direction"],
            hovertemplate="%{y}<br>%{customdata}: %{x:+.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=REFERENCE_GRAY, line_width=1)
    fig.update_layout(
        title="Why this score? Top drivers (red increases risk, blue decreases)",
        xaxis_title="Contribution to fraud log-odds (SHAP)",
        showlegend=False,
        bargap=0.25,
    )
    return style_chart(fig, height=420)


def page_investigate(predictor: FraudRiskPredictor, metrics: dict) -> None:
    st.title("Transaction investigation")
    st.caption(
        "Pick a transaction from the unseen test period, optionally adjust "
        "a few fields to run a what-if analysis, and generate a case report."
    )

    sample = load_csv(str(SAMPLE_TRANSACTIONS))

    filter_choice = st.radio(
        "Show", ["All", "Known fraud", "Known legitimate"], horizontal=True,
    )
    pool = sample
    if filter_choice == "Known fraud":
        pool = sample[sample[TARGET] == 1]
    elif filter_choice == "Known legitimate":
        pool = sample[sample[TARGET] == 0]

    def describe(i: int) -> str:
        row = pool.loc[i]
        return (f"{int(row[TRANSACTION_ID])} · ${row['TransactionAmt']:,.2f} · "
                f"{row.get('ProductCD', '')} · {row.get('card4', '')}")

    index = st.selectbox("Transaction", pool.index, format_func=describe)

    transaction = pool.loc[[index]].copy()

    with st.expander("What-if: adjust transaction fields"):
        c1, c2, c3, c4 = st.columns(4)
        amount = c1.number_input(
            "Amount (USD)", min_value=0.01,
            value=float(transaction["TransactionAmt"].iloc[0]), step=10.0,
        )
        domains = sorted(sample["P_emaildomain"].dropna().unique().tolist())
        current = transaction["P_emaildomain"].iloc[0]
        domain = c2.selectbox(
            "Purchaser e-mail domain", ["(missing)"] + domains,
            index=(domains.index(current) + 1) if current in domains else 0,
        )
        device_options = ["(missing)", "desktop", "mobile"]
        current_device = transaction["DeviceType"].iloc[0]
        device = c3.selectbox(
            "Device type", device_options,
            index=device_options.index(current_device)
            if current_device in device_options else 0,
        )
        seconds = float(transaction["TransactionDT"].iloc[0])
        hour = c4.slider("Hour of day", 0, 23, int(seconds // 3600 % 24))

        transaction["TransactionAmt"] = amount
        transaction["P_emaildomain"] = None if domain == "(missing)" else domain
        transaction["DeviceType"] = None if device == "(missing)" else device
        transaction["TransactionDT"] = (seconds // 86400) * 86400 + hour * 3600

    result = predictor.score(transaction).iloc[0].to_dict()
    explanation = predictor.explain(transaction)

    left, right = st.columns([2, 3])

    with left:
        st.plotly_chart(
            risk_gauge(result["risk_score"], result["risk_category"]),
            width="stretch",
        )
        st.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
        st.markdown(f"**Recommendation: {result['recommendation']}**")
        st.write(RECOMMENDATION_DETAILS[result["recommendation"]])
        actual = int(transaction[TARGET].iloc[0])
        st.caption(
            "Actual outcome in the dataset: "
            + ("**fraud**" if actual == 1 else "**legitimate**")
        )

    with right:
        st.plotly_chart(explanation_chart(explanation), width="stretch")

    pdf = build_transaction_report(
        transaction.iloc[0],
        result,
        explanation,
        {"threshold": metrics["threshold"], "test_metrics": metrics["test_metrics"]},
    )

    st.download_button(
        "Download investigation report (PDF)",
        data=pdf,
        file_name=f"investigation_{int(result[TRANSACTION_ID])}.pdf",
        mime="application/pdf",
        type="primary",
    )

    with st.expander("Raw transaction fields"):
        st.dataframe(
            transaction.T.dropna().rename(columns=lambda _: "value").astype(str),
            width="stretch",
        )


def page_batch(predictor: FraudRiskPredictor) -> None:
    st.title("Batch scoring")
    st.caption(
        "Upload transactions in the IEEE-CIS format (transaction and identity "
        "columns merged on TransactionID). Missing columns are treated as "
        f"missing values. Up to {MAX_BATCH_ROWS:,} rows per file."
    )

    source = st.radio(
        "Data source",
        ["Bundled sample (500 transactions)", "Upload my own CSV"],
        horizontal=True,
    )

    if source == "Upload my own CSV":
        upload = st.file_uploader("CSV file", type=["csv"])
        if upload is None:
            st.info("Upload a CSV file to score it.")
            return
        data = pd.read_csv(upload, nrows=MAX_BATCH_ROWS)
        st.caption(f"Loaded {len(data):,} rows from {upload.name}.")
    else:
        data = load_csv(str(SAMPLE_TRANSACTIONS))

    required = {"TransactionAmt", "TransactionDT"}
    missing = required - set(data.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        return

    with st.spinner(f"Scoring {len(data):,} transactions..."):
        results = predictor.score(data)

    if TARGET in data.columns:
        results["actual_isFraud"] = data[TARGET].to_numpy()

    counts = (
        results["risk_category"].value_counts()
        .reindex(RISK_CATEGORIES, fill_value=0)
    )

    cols = st.columns(4)
    for col, category in zip(cols, RISK_CATEGORIES):
        col.metric(category_badge(category), f"{counts[category]:,}")

    left, right = st.columns([2, 3])
    with left:
        fig = go.Figure(
            go.Bar(
                x=counts.index, y=counts.values,
                marker_color=[CATEGORY_COLORS[c] for c in counts.index],
                text=counts.values, textposition="outside",
                hovertemplate="<b>%{x}</b>: %{y:,} transactions<extra></extra>",
            )
        )
        fig.update_layout(title="Transactions by risk category",
                          yaxis_title="Transactions", showlegend=False)
        st.plotly_chart(style_chart(fig), width="stretch")
    with right:
        fig = go.Figure(
            go.Histogram(
                x=results["risk_score"], xbins=dict(start=0, end=101, size=5),
                marker_color=SERIES_BLUE,
                hovertemplate="Score %{x}: %{y:,} transactions<extra></extra>",
            )
        )
        for edge in [LOW_RISK_MAX + 0.5, MEDIUM_RISK_MAX + 0.5, HIGH_RISK_MAX + 0.5]:
            fig.add_vline(x=edge, line_dash="dot", line_color=REFERENCE_GRAY)
        fig.update_layout(title="Risk score distribution",
                          xaxis_title="Risk score", yaxis_title="Transactions",
                          bargap=0.08)
        st.plotly_chart(style_chart(fig), width="stretch")

    st.subheader("Highest-risk transactions")
    st.dataframe(
        results.sort_values("risk_score", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={
            "fraud_probability": st.column_config.NumberColumn(format="%.4f"),
            "risk_score": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%d"),
        },
    )

    st.download_button(
        "Download results (CSV)",
        data=build_batch_csv(results),
        file_name="risk_scores.csv",
        mime="text/csv",
        type="primary",
    )


def page_explainability() -> None:
    st.title("Model explainability")
    st.caption(
        "Global SHAP importance on 2,000 test-period transactions. SHAP "
        "values are computed exactly for each tree model and combined with "
        "the weights the meta-learner gives each model."
    )

    shap_table = load_csv(str(GLOBAL_SHAP_REPORT)).head(20).iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=shap_table["mean_abs_shap"], y=shap_table["feature"],
            orientation="h", marker_color=SERIES_BLUE,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title="Top 20 features by mean |SHAP|",
                      xaxis_title="Mean absolute contribution (log-odds)",
                      bargap=0.25)
    st.plotly_chart(style_chart(fig, height=600), width="stretch")

    with st.expander("Feature selection ranking (LightGBM gain)"):
        st.dataframe(load_csv(str(FEATURE_IMPORTANCE_REPORT)),
                     hide_index=True, width="stretch")

    st.markdown(
        """
**Reading the feature names**

- `*_frequency` - how often the value (card, address, e-mail domain, device,
  amount) appeared in training data; rare values are riskier.
- `*_amount_mean / _std / _median / _to_mean / _zscore` - how this amount
  compares with what is typical for the same card, address, e-mail domain or
  pseudo-customer (`client_key` = card1 + addr1 + purchaser e-mail domain).
- `transaction_amount_cents` - the decimal part of the amount; card-testing
  fraud often uses unusual amounts.
- `C*`, `D*`, `M*`, `V*`, `id_*` - anonymised Vesta engineered fields
  (counts, time deltas, match flags, and identity/device signals).
"""
    )


def page_about(predictor: FraudRiskPredictor) -> None:
    st.title("About this system")
    meta = predictor.metadata
    st.markdown(
        f"""
### Pipeline

1. **Data** - IEEE-CIS Fraud Detection (590k labelled card transactions),
   transaction and identity tables merged on `TransactionID`.
2. **Preprocessing** - schema alignment, numeric conversion, categorical
   gaps filled with `Unknown`; numeric gaps kept as missing because tree
   models use missingness as a signal.
3. **Feature engineering** - time of day/week, amount transforms,
   frequency encodings, per-card/address/e-mail/customer amount statistics.
   All lookup tables are learned from the training period only.
4. **Feature selection** - constant, duplicate and highly correlated
   (|r| > 0.95) features removed, then the top 100 by LightGBM gain.
5. **Model** - stacking ensemble of LightGBM, XGBoost and CatBoost with a
   Logistic Regression meta-learner (5-fold stacking), class-imbalance
   weighting.
6. **Evaluation** - chronological holdout: the most recent 20% of
   transactions, split into a threshold-tuning half and a final test half.
7. **Risk scoring** - probability → 0-100 score anchored so the tuned
   threshold sits at 50, then Low / Medium / High / Critical and a
   recommended action.
8. **Explainability** - ensemble-weighted TreeSHAP for every prediction.

### Limitations

- The dataset's features are anonymised and come from one e-commerce
  provider; the model is a demonstration of method, not a production
  control for other institutions.
- Scores are decision support - every flagged case needs human review.

### Model build

- Trained: {meta.get('trained_at', 'n/a')} on {meta.get('training_rows', 0):,}
  transactions ({meta.get('training_fraud_rate', 0):.2%} fraud)
- Features used: {meta.get('n_features', 'n/a')}
- Libraries: {', '.join(f'{k} {v}' for k, v in meta.get('packages', {}).items())}

**Author:** Muhammad Ibrahim Gimba ·
[GitHub](https://github.com/zameer-Gimba/financial-crime-risk-scoring-system)
"""
    )


# ---------------------------------------------------------------- main


def main() -> None:
    if not MODEL_BUNDLE.exists():
        st.error(
            "Model bundle not found. Train it first with "
            "`python -m src.pipeline`."
        )
        st.stop()

    predictor = load_predictor()
    metrics = load_metrics()

    pages = {
        "Overview": lambda: page_overview(metrics, predictor),
        "Investigate a transaction": lambda: page_investigate(predictor, metrics),
        "Batch scoring": lambda: page_batch(predictor),
        "Explainability": page_explainability,
        "About": lambda: page_about(predictor),
    }

    with st.sidebar:
        st.header(SIDEBAR_TITLE)
        choice = st.radio("Go to", list(pages), label_visibility="collapsed")
        st.divider()
        st.caption(
            f"Decision threshold: {predictor.threshold:.3f}\n\n"
            "Risk bands: Low 0-24 · Medium 25-49 · High 50-74 · Critical 75-100"
        )

    pages[choice]()


main()
