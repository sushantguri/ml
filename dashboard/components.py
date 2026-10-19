"""
UI/UX Components and Plotly Visualizations for Cybersecurity Dashboard.
Adaptive Cyber-Physical Security System.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def inject_custom_css() -> None:
    """Inject SOC dark theme and glassmorphic styling into Streamlit."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');

        /* Root styling and dark theme */
        :root {
            --bg-primary: #0a0e17;
            --bg-card: rgba(17, 24, 39, 0.75);
            --bg-card-hover: rgba(30, 41, 59, 0.85);
            --border-glow: rgba(56, 189, 248, 0.25);
            --accent-cyan: #38bdf8;
            --accent-blue: #2563eb;
            --accent-crimson: #f43f5e;
            --accent-amber: #fbbf24;
            --accent-emerald: #10b981;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
        }

        .stApp {
            background: radial-gradient(circle at 15% 15%, #0f172a 0%, #0a0e17 50%, #030712 100%);
            color: var(--text-main);
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Metric Cards */
        .metric-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.25rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .metric-card:hover {
            border-color: var(--accent-cyan);
            box-shadow: 0 8px 28px rgba(56, 189, 248, 0.15);
            transform: translateY(-2px);
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
            opacity: 0.5;
        }

        .metric-title {
            color: var(--text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
            font-weight: 600;
        }

        .metric-value {
            color: var(--text-main);
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.2;
        }

        .metric-sub {
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 0.3rem;
        }

        /* Priority Badges */
        .badge-critical {
            background: rgba(244, 63, 94, 0.2);
            color: #fb7185;
            border: 1px solid #f43f5e;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
        }

        .badge-high {
            background: rgba(251, 146, 60, 0.2);
            color: #fdba74;
            border: 1px solid #fb923c;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
        }

        .badge-medium {
            background: rgba(251, 191, 36, 0.2);
            color: #fde047;
            border: 1px solid #fbbf24;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
        }

        .badge-low {
            background: rgba(16, 185, 129, 0.2);
            color: #6ee7b7;
            border: 1px solid #10b981;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
        }

        /* Disclaimer Banner */
        .soc-banner {
            background: rgba(30, 41, 59, 0.6);
            border-left: 4px solid var(--accent-cyan);
            padding: 0.9rem 1.2rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
            font-size: 0.9rem;
            color: #cbd5e1;
        }

        /* Code & Monospace */
        code {
            font-family: 'JetBrains Mono', monospace !important;
            background: rgba(15, 23, 42, 0.8) !important;
            color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.2) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: Any, subtitle: str = "", border_color: Optional[str] = None) -> None:
    """Render a modern glassmorphic SOC metric card."""
    border_style = f"border-color: {border_color};" if border_color else ""
    st.markdown(
        f"""
        <div class="metric-card" style="{border_style}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_score_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot interactive histogram/density of continuous anomaly scores."""
    fig = go.Figure()

    normal_scores = df[df["is_anomaly"] == 0]["anomaly_score"]
    anomaly_scores = df[df["is_anomaly"] == 1]["anomaly_score"]

    if len(normal_scores) > 0:
        fig.add_trace(
            go.Histogram(
                x=normal_scores,
                name="Normal Traffic",
                nbinsx=40,
                marker_color="#10b981",
                opacity=0.65,
            )
        )

    if len(anomaly_scores) > 0:
        fig.add_trace(
            go.Histogram(
                x=anomaly_scores,
                name="Flagged Anomaly",
                nbinsx=40,
                marker_color="#f43f5e",
                opacity=0.75,
            )
        )

    # Decision threshold line
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="#38bdf8",
        annotation_text="Anomaly Decision Boundary (0.50)",
        annotation_position="top right",
        annotation_font_color="#38bdf8",
    )

    fig.update_layout(
        title="<b>Anomaly Score Distribution [0.0 = Normal, 1.0 = Extreme Outlier]</b>",
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        xaxis_title="Calibrated Anomaly Score",
        yaxis_title="Record Count",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_pca_2d(df: pd.DataFrame, max_points: int = 1500) -> go.Figure:
    """Plot 2D PCA projection of analyzed records."""
    plot_df = df.head(max_points).copy()
    if "pca_component_1" not in plot_df.columns or "pca_component_2" not in plot_df.columns:
        fig = go.Figure()
        fig.update_layout(title="PCA coordinates not available")
        return fig

    color_col = "anomaly_prediction"
    color_map = {"Normal": "#10b981", "Anomaly": "#f43f5e"}

    fig = px.scatter(
        plot_df,
        x="pca_component_1",
        y="pca_component_2",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={
            "protocol_type": True,
            "service": True,
            "anomaly_score": True,
            "review_priority": True,
            "pca_component_1": False,
            "pca_component_2": False,
        },
        title=f"<b>2D PCA Latent Space Projection (Sample of {len(plot_df)} records)</b>",
        template="plotly_dark",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        xaxis_title="Principal Component 1 (PC1)",
        yaxis_title="Principal Component 2 (PC2)",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_confusion_matrix(cm_dict: Dict[str, int]) -> go.Figure:
    """Plot interactive confusion matrix heatmap."""
    tn = cm_dict["true_negatives"]
    fp = cm_dict["false_positives"]
    fn = cm_dict["false_negatives"]
    tp = cm_dict["true_positives"]

    z = [[tn, fp], [fn, tp]]
    x = ["Predicted Normal (0)", "Predicted Anomaly (1)"]
    y = ["Actual Normal (0)", "Actual Attack (1)"]

    annotations = [
        [f"<b>TN: {tn:,}</b><br>Correct Normal", f"<b>FP: {fp:,}</b><br>False Alarm"],
        [f"<b>FN: {fn:,}</b><br>Missed Attack", f"<b>TP: {tp:,}</b><br>Detected Attack"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "#0f172a"], [0.5, "#1e3a8a"], [1, "#0284c7"]],
            showscale=False,
            hoverongaps=False,
        )
    )

    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=x[j],
                y=y[i],
                text=annotations[i][j],
                showarrow=False,
                font=dict(color="#ffffff", size=14, family="JetBrains Mono"),
            )

    fig.update_layout(
        title="<b>Confusion Matrix Heatmap (KDDTest+ Benchmark)</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_pr_curve(pr_data: Dict[str, Any]) -> go.Figure:
    """Plot Precision-Recall curve."""
    recalls = pr_data.get("recalls", [])
    precisions = pr_data.get("precisions", [])
    ap = pr_data.get("average_precision", 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recalls,
            y=precisions,
            mode="lines",
            name=f"PR Curve (AP = {ap:.4f})",
            line=dict(color="#38bdf8", width=3),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.15)",
        )
    )

    fig.update_layout(
        title=f"<b>Precision-Recall Curve (Average Precision: {ap:.4f})</b>",
        xaxis_title="Recall (Sensitivity to Attacks)",
        yaxis_title="Precision (Accuracy of Raised Alerts)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        xaxis=dict(range=[0, 1.02]),
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_explained_variance(pca_summary: Dict[str, Any]) -> go.Figure:
    """Plot PCA cumulative explained variance."""
    cum_var = pca_summary.get("cumulative_variance_ratios", [])
    exp_var = pca_summary.get("explained_variance_ratios", [])
    n_90 = pca_summary.get("n_components_90_variance", 20)

    x_comps = list(range(1, len(cum_var) + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_comps,
            y=exp_var,
            name="Individual Variance",
            marker_color="rgba(56, 189, 248, 0.4)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_comps,
            y=cum_var,
            mode="lines+markers",
            name="Cumulative Variance",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=5),
        )
    )

    # 90% threshold reference
    fig.add_hline(
        y=0.90,
        line_dash="dash",
        line_color="#fbbf24",
        annotation_text="90% Variance Threshold",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=n_90,
        line_dash="dot",
        line_color="#f43f5e",
        annotation_text=f"k={n_90} components",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"<b>PCA Explained Variance (Top {len(cum_var)} Components)</b>",
        xaxis_title="Principal Component Index",
        yaxis_title="Explained Variance Ratio",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_priority_breakdown(df: pd.DataFrame) -> go.Figure:
    """Plot donut chart of SOC review priorities."""
    priority_counts = df["review_priority"].value_counts()
    colors = {
        "CRITICAL": "#f43f5e",
        "HIGH": "#fb923c",
        "MEDIUM": "#fbbf24",
        "LOW": "#10b981",
    }
    marker_colors = [colors.get(p, "#94a3b8") for p in priority_counts.index]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=priority_counts.index,
                values=priority_counts.values,
                hole=0.55,
                marker=dict(colors=marker_colors),
                textinfo="label+percent",
                insidetextorientation="radial",
            )
        ]
    )
    fig.update_layout(
        title="<b>Triage Review Priority Breakdown</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


def plot_category_recall(cat_metrics: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plot detection recall across attack categories."""
    categories = list(cat_metrics.keys())
    recalls = [cat_metrics[c]["recall"] * 100 for c in categories]
    totals = [cat_metrics[c]["total"] for c in categories]
    hover_texts = [f"Total: {t:,}<br>Recall: {r:.1f}%" for t, r in zip(totals, recalls)]

    fig = go.Figure(
        data=[
            go.Bar(
                x=[c.upper() for c in categories],
                y=recalls,
                text=[f"{r:.1f}%" for r in recalls],
                textposition="auto",
                hovertext=hover_texts,
                marker_color=["#ef4444", "#f97316", "#eab308", "#8b5cf6"][: len(categories)],
            )
        ]
    )
    fig.update_layout(
        title="<b>Detection Recall Across Attack Categories</b>",
        xaxis_title="Attack Class",
        yaxis_title="Detection Rate (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
