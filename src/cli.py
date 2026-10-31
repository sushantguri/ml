"""
Command-Line Interface (CLI) for Adaptive Cyber-Physical Anomaly Detection.

Provides command-line commands for:
- Batch scoring network telemetry sessions from CSV files
- Automated SOC incident dossier and summary report generation
- Model artifact inspection and benchmark statistics display
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import InferencePipeline
from src.reporting import generate_incident_dossier

logger = logging.getLogger("anomaly_cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ml-anomaly",
        description="Adaptive Cyber-Physical Anomaly Detection & Incident Response CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational subcommands")

    # Command: score
    score_parser = subparsers.add_parser(
        "score",
        help="Batch score network traffic CSV against baseline model",
    )
    score_parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to input CSV containing network session records",
    )
    score_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional path to export enriched CSV containing anomaly scores and labels",
    )
    score_parser.add_argument(
        "--json",
        action="store_true",
        help="Output summary statistics in JSON format",
    )
    score_parser.add_argument(
        "--fail-on-anomaly",
        action="store_true",
        help="Exit with code 2 if any CRITICAL or HIGH priority anomalies are detected",
    )

    # Command: report
    report_parser = subparsers.add_parser(
        "report",
        help="Generate automated Markdown SOC incident dossiers for anomalous sessions",
    )
    report_parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to input CSV containing network session records",
    )
    report_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to write the generated markdown incident report (defaults to stdout)",
    )
    report_parser.add_argument(
        "--max-dossiers",
        type=int,
        default=5,
        help="Maximum number of individual incident dossiers to compile (default: 5)",
    )

    # Command: info
    subparsers.add_parser(
        "info",
        help="Inspect active model parameters and benchmark validation metrics",
    )

    return parser


def handle_score(args: argparse.Namespace) -> int:
    """Execute batch telemetry scoring."""
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: Input file '{args.input}' does not exist.\n")
        return 1

    try:
        df = pd.read_csv(input_path)
    except (ValueError, pd.errors.ParserError, OSError) as e:
        sys.stderr.write(f"Error: Failed to parse CSV '{args.input}': {e}\n")
        return 1

    if df.empty:
        sys.stderr.write(f"Error: Input file '{args.input}' contains no records.\n")
        return 1

    pipeline = InferencePipeline()
    if not pipeline.is_loaded:
        sys.stderr.write("Error: Inference pipeline artifacts could not be loaded.\n")
        return 1

    enriched = pipeline.predict_records(df)

    total_records = len(enriched)
    anomalies_count = int((enriched["is_anomaly"] == 1).sum())
    anomaly_rate = (anomalies_count / total_records) * 100

    priority_counts = (
        enriched["review_priority"].value_counts().to_dict()
        if "review_priority" in enriched
        else {}
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(out_path, index=False)

    if args.json:
        summary = {
            "input_file": str(input_path),
            "output_file": str(args.output) if args.output else None,
            "total_records": total_records,
            "anomalies_detected": anomalies_count,
            "anomaly_rate_pct": round(anomaly_rate, 2),
            "priorities": priority_counts,
        }
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    else:
        sys.stdout.write("=====================================================\n")
        sys.stdout.write("   CYBER-PHYSICAL ANOMALY DETECTION BATCH SUMMARY    \n")
        sys.stdout.write("=====================================================\n")
        sys.stdout.write(f"Input File:         {input_path}\n")
        if args.output:
            sys.stdout.write(f"Output File:        {args.output}\n")
        sys.stdout.write(f"Total Sessions:     {total_records:,}\n")
        sys.stdout.write(f"Anomalies Flagged:  {anomalies_count:,} ({anomaly_rate:.2f}%)\n")
        sys.stdout.write("-----------------------------------------------------\n")
        sys.stdout.write("Triage Review Breakdown:\n")
        for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = priority_counts.get(priority, 0)
            sys.stdout.write(f"  - {priority:<10}: {count:,}\n")
        sys.stdout.write("=====================================================\n")

    if args.fail_on_anomaly and (priority_counts.get("CRITICAL", 0) > 0 or priority_counts.get("HIGH", 0) > 0):
        return 2

    return 0


def handle_report(args: argparse.Namespace) -> int:
    """Execute incident report generation."""
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: Input file '{args.input}' does not exist.\n")
        return 1

    try:
        df = pd.read_csv(input_path)
    except (ValueError, pd.errors.ParserError, OSError) as e:
        sys.stderr.write(f"Error: Failed to parse CSV '{args.input}': {e}\n")
        return 1

    pipeline = InferencePipeline()
    if not pipeline.is_loaded:
        sys.stderr.write("Error: Inference pipeline artifacts could not be loaded.\n")
        return 1

    enriched = pipeline.predict_records(df)
    anomalies = enriched[enriched["is_anomaly"] == 1]

    report_sections = [
        "# Security Incident Investigation Report",
        "## Automated Threat Dossier Compilation",
        f"- **Source File:** `{input_path.name}`",
        f"- **Evaluated Records:** {len(enriched):,}",
        f"- **Anomalies Detected:** {len(anomalies):,}",
        "",
        "---",
        "",
    ]

    if anomalies.empty:
        report_sections.append("No anomalous network telemetry detected in the provided batch.")
    else:
        top_anomalies = anomalies.sort_values(by="anomaly_score", ascending=False).head(args.max_dossiers)
        for idx, (_, row) in enumerate(top_anomalies.iterrows(), start=1):
            dossier = generate_incident_dossier(
                record=row,
                alert_id=f"BATCH-{idx:03d}",
                disposition="PENDING_SOC_REVIEW",
            )
            report_sections.append(dossier)
            report_sections.append("\n---\n")

    full_report = "\n".join(report_sections)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(full_report, encoding="utf-8")
        sys.stdout.write(f"Incident report generated successfully: {out_path}\n")
    else:
        sys.stdout.write(full_report + "\n")

    return 0


def handle_info(_: argparse.Namespace) -> int:
    """Display model metadata and benchmark metrics."""
    pipeline = InferencePipeline()
    if not pipeline.is_loaded:
        sys.stderr.write("Error: Inference pipeline artifacts could not be loaded.\n")
        return 1

    metadata = pipeline.metadata or {}
    dataset_info = metadata.get("dataset", {})
    pca_info = metadata.get("pca_diagnostics", {})
    test_eval = metadata.get("test_evaluation", {})

    train_samples = dataset_info.get("train_samples", "N/A")
    train_samples_str = f"{train_samples:,}" if isinstance(train_samples, (int, float)) else str(train_samples)

    sys.stdout.write("=====================================================\n")
    sys.stdout.write("      MODEL ARTIFACT SPECIFICATIONS & BENCHMARK      \n")
    sys.stdout.write("=====================================================\n")
    sys.stdout.write("Algorithm:             IsolationForest (Unsupervised)\n")
    sys.stdout.write(f"Trained Timestamp:     {metadata.get('timestamp', 'N/A')}\n")
    sys.stdout.write(f"Benchmark Dataset:     {dataset_info.get('name', 'NSL-KDD')}\n")
    sys.stdout.write(f"Training Samples:      {train_samples_str}\n")
    sys.stdout.write(f"PCA Latent Dimensions: {pca_info.get('n_components_90_variance', 'N/A')}\n")
    sys.stdout.write("-----------------------------------------------------\n")
    sys.stdout.write("Benchmark Performance (Held-out Test Set):\n")
    sys.stdout.write(f"  - Precision:          {test_eval.get('precision', 0.0):.4f}\n")
    sys.stdout.write(f"  - Recall:             {test_eval.get('recall', 0.0):.4f}\n")
    sys.stdout.write(f"  - F1-Score:           {test_eval.get('f1_score', 0.0):.4f}\n")
    sys.stdout.write(f"  - False Positive Rate:{test_eval.get('false_positive_rate', 0.0):.4f}\n")
    sys.stdout.write("=====================================================\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "score":
        return handle_score(args)
    elif args.command == "report":
        return handle_report(args)
    elif args.command == "info":
        return handle_info(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
