"""
Investigation report generator for the Financial Crime Detection &
Risk Scoring System.

Builds downloadable PDF case reports for individual transactions and
CSV exports for batch scoring.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config.constants import APP_TITLE, REPORT_TITLE
from src.risk_scoring import CATEGORY_COLORS, RECOMMENDATION_DETAILS

# Raw fields shown in the "transaction details" section when present.
DETAIL_FIELDS = {
    "TransactionID": "Transaction ID",
    "TransactionAmt": "Amount (USD)",
    "ProductCD": "Product code",
    "card4": "Card network",
    "card6": "Card type",
    "P_emaildomain": "Purchaser e-mail domain",
    "R_emaildomain": "Recipient e-mail domain",
    "DeviceType": "Device type",
    "DeviceInfo": "Device",
    "addr1": "Billing region",
}


def _format_value(value) -> str:
    """
    Render a value for display in the report.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"

    if isinstance(value, float):
        return f"{value:,.2f}"

    return str(value)


def _table(rows: list[list[str]], widths: list[float]) -> Table:
    """
    A consistently styled table.
    """

    table = Table(rows, colWidths=widths, hAlign="LEFT")

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2A44")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C8CDD8")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F4F6FA")],
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return table


def build_transaction_report(
    transaction: pd.Series,
    result: dict,
    explanation: pd.DataFrame,
    model_info: dict | None = None,
) -> bytes:
    """
    Create a PDF investigation report for one transaction.

    Parameters
    ----------
    transaction : pd.Series
        The raw transaction.
    result : dict
        One row of FraudRiskPredictor.score() as a dict.
    explanation : pd.DataFrame
        Output of FraudRiskPredictor.explain().
    model_info : dict, optional
        Model metadata (threshold, metrics) for the appendix.

    Returns
    -------
    bytes
        The PDF document.
    """

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=REPORT_TITLE,
    )

    styles = getSampleStyleSheet()

    body = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
    )

    category = result["risk_category"]

    category_color = colors.HexColor(CATEGORY_COLORS[category])

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story = [
        Paragraph(REPORT_TITLE, styles["Title"]),
        Paragraph(f"{APP_TITLE} &middot; generated {generated}", body),
        Spacer(1, 6 * mm),
    ]

    summary = Table(
        [
            ["Risk score", "Risk category", "Fraud probability", "Recommendation"],
            [
                f"{result['risk_score']} / 100",
                category,
                f"{result['fraud_probability']:.2%}",
                result["recommendation"],
            ],
        ],
        colWidths=[38 * mm, 38 * mm, 38 * mm, 60 * mm],
        hAlign="LEFT",
    )

    summary.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#5B6475")),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 1), 13),
                ("TEXTCOLOR", (1, 1), (1, 1), category_color),
                ("LINEBELOW", (0, 1), (-1, 1), 2, category_color),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
            ]
        )
    )

    story += [
        summary,
        Spacer(1, 4 * mm),
        Paragraph(
            f"<b>Recommended action:</b> "
            f"{RECOMMENDATION_DETAILS[result['recommendation']]}",
            body,
        ),
        Spacer(1, 6 * mm),
        Paragraph("Transaction details", styles["Heading2"]),
    ]

    detail_rows = [["Field", "Value"]] + [
        [label, _format_value(transaction.get(field))]
        for field, label in DETAIL_FIELDS.items()
        if field in transaction.index
    ]

    story += [
        _table(detail_rows, [60 * mm, 110 * mm]),
        Spacer(1, 6 * mm),
        Paragraph("Key risk drivers (SHAP)", styles["Heading2"]),
        Paragraph(
            "Contributions are on the log-odds scale and combine the "
            "base models of the ensemble. Positive values push the "
            "transaction towards fraud.",
            body,
        ),
        Spacer(1, 2 * mm),
    ]

    driver_rows = [["Feature", "Value", "Contribution", "Effect"]] + [
        [
            row.feature,
            _format_value(row.value),
            f"{row.contribution:+.3f}",
            row.direction,
        ]
        for row in explanation.itertuples()
    ]

    story.append(_table(driver_rows, [62 * mm, 38 * mm, 30 * mm, 40 * mm]))

    if model_info:
        metrics = model_info.get("test_metrics", {})

        story += [
            Spacer(1, 6 * mm),
            Paragraph("Model information", styles["Heading2"]),
            Paragraph(
                "Stacked ensemble (LightGBM, XGBoost, CatBoost with a "
                "Logistic Regression meta-learner) trained on the "
                "IEEE-CIS Fraud Detection dataset. "
                f"Decision threshold: {model_info.get('threshold', 0):.3f}. "
                f"Holdout precision {metrics.get('precision', 0):.1%}, "
                f"recall {metrics.get('recall', 0):.1%}, "
                f"PR-AUC {metrics.get('pr_auc', 0):.3f}, "
                f"ROC-AUC {metrics.get('roc_auc', 0):.3f}.",
                body,
            ),
        ]

    story += [
        Spacer(1, 8 * mm),
        Paragraph(
            "<i>This report is decision support generated by a machine "
            "learning model. Final determinations require review by a "
            "qualified analyst.</i>",
            body,
        ),
    ]

    document.build(story)

    return buffer.getvalue()


def build_batch_csv(
    results: pd.DataFrame,
) -> bytes:
    """
    Encode batch scoring results as CSV bytes for download.
    """

    return results.to_csv(index=False).encode("utf-8")
