"""
Unit Tests for Command-Line Interface (CLI).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.cli import main


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Create a temporary CSV file with test network sessions."""
    csv_file = tmp_path / "test_sessions.csv"
    df = pd.DataFrame([
        {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 215,
            "dst_bytes": 450,
            "count": 1,
            "srv_count": 1,
            "serror_rate": 0.0,
            "diff_srv_rate": 0.0,
            "dst_host_count": 1,
            "dst_host_srv_count": 1,
        },
        {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "private",
            "flag": "S0",
            "src_bytes": 0,
            "dst_bytes": 0,
            "count": 280,
            "srv_count": 280,
            "serror_rate": 1.0,
            "diff_srv_rate": 0.0,
            "dst_host_count": 255,
            "dst_host_srv_count": 5,
        },
    ])
    df.to_csv(csv_file, index=False)
    return csv_file


def test_cli_help(capsys: pytest.CaptureFixture):
    """Test CLI help output without arguments."""
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Adaptive Cyber-Physical Anomaly Detection" in captured.out


def test_cli_info(capsys: pytest.CaptureFixture):
    """Test model info subcommand."""
    exit_code = main(["info"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "MODEL ARTIFACT SPECIFICATIONS" in captured.out
    assert "Benchmark Performance" in captured.out


def test_cli_score_stdout(sample_csv_path: Path, capsys: pytest.CaptureFixture):
    """Test scoring CSV to stdout."""
    exit_code = main(["score", "-i", str(sample_csv_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "CYBER-PHYSICAL ANOMALY DETECTION BATCH SUMMARY" in captured.out
    assert "Total Sessions:     2" in captured.out


def test_cli_score_json(sample_csv_path: Path, capsys: pytest.CaptureFixture):
    """Test scoring CSV with JSON output format."""
    exit_code = main(["score", "-i", str(sample_csv_path), "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["total_records"] == 2
    assert "anomalies_detected" in data
    assert "priorities" in data


def test_cli_score_output_file(sample_csv_path: Path, tmp_path: Path):
    """Test scoring CSV and writing output enriched CSV."""
    out_csv = tmp_path / "scored_output.csv"
    exit_code = main(["score", "-i", str(sample_csv_path), "-o", str(out_csv)])
    assert exit_code == 0
    assert out_csv.exists()

    df_out = pd.read_csv(out_csv)
    assert "is_anomaly" in df_out.columns
    assert "anomaly_score" in df_out.columns
    assert "review_priority" in df_out.columns


def test_cli_score_missing_file(capsys: pytest.CaptureFixture):
    """Test error handling for non-existent input file."""
    exit_code = main(["score", "-i", "non_existent_file.csv"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_cli_report(sample_csv_path: Path, tmp_path: Path):
    """Test report generation subcommand."""
    out_report = tmp_path / "incident_report.md"
    exit_code = main(["report", "-i", str(sample_csv_path), "-o", str(out_report)])
    assert exit_code == 0
    assert out_report.exists()

    content = out_report.read_text(encoding="utf-8")
    assert "Security Incident Investigation Report" in content
