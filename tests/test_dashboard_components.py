"""
Unit Tests for Dashboard Components and Plotly Visualizations.
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard.components import (
    plot_score_distribution,
    plot_pca_2d,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_explained_variance,
    plot_priority_breakdown,
    plot_category_recall,
    plot_radar_risk_profile,
    plot_protocol_sunburst,
    plot_telemetry_flow_timeline,
)


@pytest.fixture
def dummy_results_df():
    """Create sample predictions DataFrame for visualization testing."""
    return pd.DataFrame({
        "protocol_type": ["tcp", "udp", "icmp", "tcp", "tcp"],
        "service": ["http", "dns", "eco_i", "smtp", "ftp"],
        "is_anomaly": [0, 0, 1, 1, 0],
        "anomaly_score": [0.12, 0.35, 0.88, 0.74, 0.22],
        "anomaly_prediction": ["Normal", "Normal", "Anomaly", "Anomaly", "Normal"],
        "review_priority": ["LOW", "LOW", "CRITICAL", "HIGH", "LOW"],
        "pca_component_1": [0.1, -0.2, 2.5, 1.8, -0.5],
        "pca_component_2": [-0.3, 0.4, 1.2, -1.1, 0.0],
        "src_bytes": [200, 50, 45000, 1000, 300],
        "dst_bytes": [400, 100, 0, 0, 150],
        "count": [2, 1, 450, 20, 1],
        "serror_rate": [0.0, 0.0, 1.0, 0.2, 0.0],
        "diff_srv_rate": [0.0, 0.0, 0.8, 0.1, 0.0],
        "dst_host_count": [10, 5, 255, 150, 20],
    })


def test_plot_score_distribution(dummy_results_df):
    """Test score distribution figure generation."""
    fig = plot_score_distribution(dummy_results_df)
    assert isinstance(fig, go.Figure)


def test_plot_pca_2d(dummy_results_df):
    """Test 2D PCA projection figure generation."""
    fig = plot_pca_2d(dummy_results_df)
    assert isinstance(fig, go.Figure)


def test_plot_confusion_matrix():
    """Test confusion matrix heatmap figure generation."""
    cm = {"true_negatives": 8000, "false_positives": 500, "false_negatives": 1000, "true_positives": 9000}
    fig = plot_confusion_matrix(cm)
    assert isinstance(fig, go.Figure)


def test_plot_pr_curve():
    """Test precision recall curve figure generation."""
    pr_data = {"recalls": [0.0, 0.5, 1.0], "precisions": [1.0, 0.9, 0.8], "average_precision": 0.92}
    fig = plot_pr_curve(pr_data)
    assert isinstance(fig, go.Figure)


def test_plot_explained_variance():
    """Test explained variance figure generation."""
    pca_summary = {
        "cumulative_variance_ratios": [0.2, 0.4, 0.6, 0.8, 0.92],
        "explained_variance_ratios": [0.2, 0.2, 0.2, 0.2, 0.12],
        "n_components_90_variance": 5,
    }
    fig = plot_explained_variance(pca_summary)
    assert isinstance(fig, go.Figure)


def test_plot_priority_breakdown(dummy_results_df):
    """Test review priority donut chart generation."""
    fig = plot_priority_breakdown(dummy_results_df)
    assert isinstance(fig, go.Figure)


def test_plot_category_recall():
    """Test category recall bar chart generation."""
    cat_metrics = {
        "dos": {"recall": 0.84, "total": 5000},
        "probe": {"recall": 0.88, "total": 2000},
    }
    fig = plot_category_recall(cat_metrics)
    assert isinstance(fig, go.Figure)


def test_plot_radar_risk_profile(dummy_results_df):
    """Test polar radar risk chart generation."""
    record = dummy_results_df.iloc[2]
    fig = plot_radar_risk_profile(record)
    assert isinstance(fig, go.Figure)


def test_plot_protocol_sunburst(dummy_results_df):
    """Test protocol hierarchy sunburst chart generation."""
    fig = plot_protocol_sunburst(dummy_results_df)
    assert isinstance(fig, go.Figure)


def test_plot_telemetry_flow_timeline(dummy_results_df):
    """Test telemetry sequence timeline generation."""
    fig = plot_telemetry_flow_timeline(dummy_results_df)
    assert isinstance(fig, go.Figure)
