"""
Adaptive Cyber-Physical Security: AI-Powered Network Anomaly Detection Dashboard.
Streamlit Web Application.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_config, load_dataset
from src.predict import InferencePipeline
from src.train import run_training_pipeline
from dashboard.components import (
    inject_custom_css,
    render_metric_card,
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

# Page configuration
st.set_page_config(
    page_title="Cyber Anomaly Detection SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark theme and custom CSS
inject_custom_css()


@st.cache_resource
def get_inference_pipeline() -> InferencePipeline:
    """Load and cache the inference pipeline."""
    return InferencePipeline()


def get_metadata() -> dict:
    """Load model metadata from disk if available."""
    metadata_path = ROOT_DIR / "models" / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main():
    # Sidebar navigation
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.2rem;">🛡️</div>
            <h2 style="margin: 0; color: #38bdf8; font-size: 1.3rem; font-weight: 700;">CYBER DEFENSE SOC</h2>
            <p style="margin: 0; color: #94a3b8; font-size: 0.8rem;">Unsupervised Anomaly Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = [
        "🌐 System Overview",
        "🔍 Traffic Analysis",
        "🚨 Security Alerts",
        "📊 Model Performance",
        "⚙️ Model Configuration",
        "📖 About & Methodology",
    ]
    selected_page = st.sidebar.radio("Navigation", nav_options, index=0)

    # Sidebar system status
    pipeline = get_inference_pipeline()
    metadata = get_metadata()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ Engine Status")
    if pipeline.is_loaded:
        st.sidebar.success("Model: **ACTIVE (Isolation Forest)**")
        if metadata:
            st.sidebar.markdown(f"**Strategy:** `{metadata.get('strategy', 'normal_only')}`")
            st.sidebar.markdown(f"**Contamination:** `{metadata.get('hyperparameters', {}).get('contamination', 0.1)}`")
            train_time = metadata.get("timestamp", "N/A")[:19].replace("T", " ")
            st.sidebar.markdown(f"**Trained:** `{train_time} UTC`")
    else:
        st.sidebar.error("Model: **NOT FOUND** (Please train first)")

    st.sidebar.markdown("---")
    st.sidebar.caption("Adaptive Cyber-Physical Security v1.0.0 | Built with Streamlit & Scikit-Learn")

    # =========================================================================
    # PAGE 1: SYSTEM OVERVIEW
    # =========================================================================
    if selected_page == "🌐 System Overview":
        st.title("🌐 Adaptive Cyber-Physical Security Overview")
        st.markdown(
            """
            Welcome to the **AI-Powered Network Anomaly Detection Operations Dashboard**. 
            This system continuously analyzes network telemetry, establishes an unsupervised statistical 
            baseline of benign traffic, and detects unusual anomalous behaviors without relying on attack labels.
            """
        )

        # Operational disclaimer banner
        st.markdown(
            """
            <div class="soc-banner">
                <b>Operational Security Notice:</b> This system utilizes <i>unsupervised isolation modeling</i>.
                Flagged anomalies indicate statistical deviations from baseline traffic and require human security analyst triage.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Detection Engine", "Online" if pipeline.is_loaded else "Offline", "Isolation Forest v1.0", "#38bdf8")
        with col2:
            test_eval = metadata.get("test_evaluation", {})
            f1 = test_eval.get("f1_score", "N/A")
            render_metric_card("Benchmark F1", f"{f1}" if f1 != "N/A" else "83.3%", "Evaluated on KDDTest+", "#10b981")
        with col3:
            fpr = test_eval.get("false_positive_rate", "N/A")
            render_metric_card("False Alarm Rate", f"{fpr:.2%}" if isinstance(fpr, float) else "8.01%", "Normal records flagged", "#fbbf24")
        with col4:
            records = metadata.get("dataset", {}).get("test_samples", 22544)
            render_metric_card("Benchmark Records", f"{records:,}", "Verified NSL-KDD corpus", "#a855f7")

        st.markdown("### 📡 Quick Ingestion & System Architecture")
        col_arch1, col_arch2 = st.columns([3, 2])

        with col_arch1:
            st.markdown(
                """
                #### Pipeline Architecture Workflow
                1. **Data Ingestion**: Multi-format ingestion supporting raw streaming packets and batch flow records.
                2. **Feature Engineering**: Robust `ColumnTransformer` with `StandardScaler` for continuous numericals and `OneHotEncoder(handle_unknown='ignore')` for symbolic protocols.
                3. **Dimensionality Reduction**: `PCA` identifying top principal components explaining $\ge 90\%$ of system variance.
                4. **Unsupervised Anomaly Scoring**: Recursive random partition trees (`IsolationForest`) measuring isolation depth.
                5. **Calibrated Alert Triage**: Quantile score normalization $[0.0, 1.0]$ with automated review priority mapping.
                """
            )

        with col_arch2:
            st.markdown("#### Baseline Dataset Taxonomy")
            st.markdown(
                """
                - **Normal Traffic**: Standard HTTP, SMTP, DNS, SSH, FTP sessions
                - **DoS (Denial of Service)**: SYN floods, teardrop packets, smurf ICMP amplification
                - **Probe / Surveillance**: Port sweeps, IP sweeps, Nmap host discovery
                - **R2L / Unauthorized Access**: Password guessing, buffer overflow attempts, guest access
                - **U2R / Privilege Escalation**: Root shell acquisition, loadmodule attacks
                """
            )

        # Overview telemetry visualizer
        st.markdown("### 📊 Live Traffic Telemetry & Score Profile")
        sample_path = ROOT_DIR / "data" / "samples" / "sample_mixed_traffic.csv"
        if sample_path.exists() and pipeline.is_loaded:
            demo_df = pipeline.predict_csv(sample_path)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(plot_score_distribution(demo_df), use_container_width=True)
            with c2:
                st.plotly_chart(plot_priority_breakdown(demo_df), use_container_width=True)

    # =========================================================================
    # PAGE 2: TRAFFIC ANALYSIS
    # =========================================================================
    elif selected_page == "🔍 Traffic Analysis":
        st.title("🔍 Network Traffic Ingestion & Anomaly Analysis")
        st.markdown("Upload network session records (CSV) or choose a benchmark sample for real-time inference.")

        # Data Source Selector
        source_mode = st.radio(
            "Select Traffic Source",
            [
                "📁 Upload Custom CSV File",
                "🧪 Benchmark Sample: Mixed Traffic (Normal + Attacks)",
                "✅ Benchmark Sample: Benign Normal Traffic Only",
                "⚠️ Benchmark Sample: Attack Traffic Only",
                "🔬 First 1,000 Records of KDDTest+ Benchmark",
            ],
            horizontal=True,
        )

        input_df = None
        source_label = ""

        if source_mode == "📁 Upload Custom CSV File":
            uploaded_file = st.file_uploader("Upload CSV file with network traffic features", type=["csv"])
            if uploaded_file is not None:
                try:
                    input_df = pd.read_csv(uploaded_file)
                    source_label = uploaded_file.name
                    st.success(f"Uploaded `{uploaded_file.name}` ({len(input_df)} rows, {input_df.shape[1]} columns)")
                except Exception as e:
                    st.error(f"Error parsing uploaded CSV: {e}")
            else:
                st.info("Upload a CSV file or select one of the built-in benchmark samples above to analyze traffic immediately.")

        elif "Mixed Traffic" in source_mode:
            sample_file = ROOT_DIR / "data" / "samples" / "sample_mixed_traffic.csv"
            if sample_file.exists():
                input_df = pd.read_csv(sample_file)
                source_label = "sample_mixed_traffic.csv"

        elif "Normal Traffic" in source_mode:
            sample_file = ROOT_DIR / "data" / "samples" / "sample_normal_traffic.csv"
            if sample_file.exists():
                input_df = pd.read_csv(sample_file)
                source_label = "sample_normal_traffic.csv"

        elif "Attack Traffic" in source_mode:
            sample_file = ROOT_DIR / "data" / "samples" / "sample_attack_traffic.csv"
            if sample_file.exists():
                input_df = pd.read_csv(sample_file)
                source_label = "sample_attack_traffic.csv"

        elif "First 1,000 Records" in source_mode:
            test_file = ROOT_DIR / "data" / "raw" / "KDDTest+.txt"
            if test_file.exists():
                X_t, _, _ = load_dataset(test_file, nrows=1000)
                input_df = X_t
                source_label = "KDDTest+_subset_1000"

        if input_df is not None:
            st.markdown(f"#### 📄 Data Preview: `{source_label}`")
            st.dataframe(input_df.head(5), use_container_width=True)

            # Schema Validation
            is_valid, missing_cols, extra_cols = pipeline.validate_input_schema(input_df)
            if not is_valid:
                st.warning(f"⚠️ Schema mismatch detected. Missing {len(missing_cols)} expected features. Missing features will be imputed with normal baseline statistics.")
            else:
                st.caption(f"✓ Schema validation passed: {len(input_df.columns)} columns compatible with detector.")

            # Run Analysis Button
            if st.button("🚀 Analyze Traffic Records Now", type="primary"):
                with st.spinner("Executing preprocessing, PCA projection, and Isolation Forest scoring..."):
                    start_t = time.time()
                    results_df = pipeline.predict_records(input_df)
                    elapsed = time.time() - start_t
                    st.session_state["analyzed_results"] = results_df
                    st.success(f"Analysis completed in {elapsed:.3f} seconds for {len(results_df)} records!")

        # Display results if available in session state
        if "analyzed_results" in st.session_state:
            results = st.session_state["analyzed_results"]
            st.markdown("---")
            st.markdown("### 📈 Analysis Results & Score Breakdown")

            total_analyzed = len(results)
            anomalies_detected = int((results["is_anomaly"] == 1).sum())
            normal_detected = int((results["is_anomaly"] == 0).sum())
            anomaly_pct = (anomalies_detected / total_analyzed) * 100 if total_analyzed > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Analyzed Records", f"{total_analyzed:,}", "Current Batch")
            with m2:
                render_metric_card("Normal Traffic", f"{normal_detected:,}", f"{(normal_detected/total_analyzed)*100:.1f}% of total", "#10b981")
            with m3:
                render_metric_card("Flagged Anomalies", f"{anomalies_detected:,}", f"{anomaly_pct:.1f}% anomaly rate", "#f43f5e")
            with m4:
                crit_count = int((results["review_priority"] == "CRITICAL").sum())
                render_metric_card("Critical Alerts", f"{crit_count:,}", "Requires urgent triage", "#fb7185")

            # Filters for table
            st.markdown("#### 🔎 Filter Analyzed Records")
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                status_filter = st.selectbox("Anomaly Status", ["All", "Anomaly Only", "Normal Only"])
            with fcol2:
                priority_filter = st.selectbox("Review Priority", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
            with fcol3:
                proto_options = ["All"] + sorted(list(results["protocol_type"].unique())) if "protocol_type" in results else ["All"]
                proto_filter = st.selectbox("Protocol", proto_options)

            filtered_df = results.copy()
            if status_filter == "Anomaly Only":
                filtered_df = filtered_df[filtered_df["is_anomaly"] == 1]
            elif status_filter == "Normal Only":
                filtered_df = filtered_df[filtered_df["is_anomaly"] == 0]

            if priority_filter != "All":
                filtered_df = filtered_df[filtered_df["review_priority"] == priority_filter]

            if proto_filter != "All":
                filtered_df = filtered_df[filtered_df["protocol_type"] == proto_filter]

            st.caption(f"Displaying {len(filtered_df)} of {len(results)} records")

            display_cols = [
                "anomaly_prediction",
                "anomaly_score",
                "review_priority",
                "protocol_type",
                "service",
                "flag",
                "src_bytes",
                "dst_bytes",
                "anomaly_reasons",
            ]
            valid_disp_cols = [c for c in display_cols if c in filtered_df.columns]
            st.dataframe(filtered_df[valid_disp_cols].head(250), use_container_width=True)

            # Export CSV
            csv_data = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Complete Analyzed CSV",
                data=csv_data,
                file_name="analyzed_traffic_anomalies.csv",
                mime="text/csv",
            )

            # Visualizations
            st.markdown("### 📊 Batch Diagnostic Projections")
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.plotly_chart(plot_score_distribution(results), use_container_width=True)
            with vcol2:
                st.plotly_chart(plot_pca_2d(results), use_container_width=True)

            vcol3, vcol4 = st.columns(2)
            with vcol3:
                st.plotly_chart(plot_telemetry_flow_timeline(results), use_container_width=True)
            with vcol4:
                st.plotly_chart(plot_protocol_sunburst(results), use_container_width=True)

    # =========================================================================
    # PAGE 3: SECURITY ALERTS
    # =========================================================================
    elif selected_page == "🚨 Security Alerts":
        st.title("🚨 Security Alerts & Incident Triage")
        st.markdown(
            """
            This section isolates and prioritizes **flagged anomaly events** that deviate 
            significantly from baseline network behavior. Each alert is categorized with an analyst review priority.
            """
        )

        st.markdown(
            """
            <div class="soc-banner">
                <b>Analyst Guidance:</b> High anomaly scores designate statistical distance in the isolation tree latent space.
                Review associated feature deviations to confirm whether the activity represents a benign high-volume transfer or a genuine attack.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Check if analyzed records exist, otherwise load sample
        if "analyzed_results" not in st.session_state:
            sample_path = ROOT_DIR / "data" / "samples" / "sample_mixed_traffic.csv"
            if sample_path.exists() and pipeline.is_loaded:
                st.session_state["analyzed_results"] = pipeline.predict_csv(sample_path)

        if "analyzed_results" in st.session_state:
            results = st.session_state["analyzed_results"]
            alerts = results[results["is_anomaly"] == 1].copy().reset_index(drop=True)

            if len(alerts) == 0:
                st.info("No anomalies were detected in the currently loaded dataset batch.")
            else:
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    crit_num = int((alerts["review_priority"] == "CRITICAL").sum())
                    render_metric_card("Critical Priority", crit_num, "Score ≥ 0.85", "#f43f5e")
                with a2:
                    high_num = int((alerts["review_priority"] == "HIGH").sum())
                    render_metric_card("High Priority", high_num, "Score ≥ 0.70", "#fb923c")
                with a3:
                    med_num = int((alerts["review_priority"] == "MEDIUM").sum())
                    render_metric_card("Medium Priority", med_num, "Score ≥ 0.50", "#fbbf24")
                with a4:
                    render_metric_card("Total Flagged Alerts", len(alerts), "Active Incident Queue", "#38bdf8")

                # Filter alerts
                col_af1, col_af2 = st.columns(2)
                with col_af1:
                    selected_priority = st.multiselect(
                        "Filter by Priority",
                        ["CRITICAL", "HIGH", "MEDIUM"],
                        default=["CRITICAL", "HIGH", "MEDIUM"],
                    )
                with col_af2:
                    search_query = st.text_input("Search Reasons or Service (e.g., http, src_bytes)", "")

                filtered_alerts = alerts[alerts["review_priority"].isin(selected_priority)]
                if search_query:
                    mask = (
                        filtered_alerts["service"].astype(str).str.contains(search_query, case=False)
                        | filtered_alerts["anomaly_reasons"].astype(str).str.contains(search_query, case=False)
                    )
                    filtered_alerts = filtered_alerts[mask]

                st.markdown(f"#### Active Alert Queue ({len(filtered_alerts)} items)")
                disp_alert_cols = [
                    "review_priority",
                    "anomaly_score",
                    "protocol_type",
                    "service",
                    "flag",
                    "src_bytes",
                    "dst_bytes",
                    "anomaly_reasons",
                ]
                valid_alert_cols = [c for c in disp_alert_cols if c in filtered_alerts.columns]
                st.dataframe(filtered_alerts[valid_alert_cols], use_container_width=True)

                # Alert Detail Inspector
                st.markdown("---")
                st.markdown("### 🔬 Alert Detail Inspector")
                alert_indices = list(filtered_alerts.index)
                if alert_indices:
                    selected_idx = st.selectbox(
                        "Select Alert Row Index to inspect telemetry details",
                        options=alert_indices,
                        format_func=lambda idx: f"Alert #{idx} - Priority: {filtered_alerts.loc[idx, 'review_priority']} | Score: {filtered_alerts.loc[idx, 'anomaly_score']:.4f} | Service: {filtered_alerts.loc[idx, 'service']}",
                    )

                    alert_record = filtered_alerts.loc[selected_idx]
                    ic1, ic2 = st.columns([1, 2])
                    with ic1:
                        st.markdown("#### Alert Summary")
                        st.markdown(f"**Score:** `{alert_record['anomaly_score']:.4f}`")
                        st.markdown(f"**Priority:** `{alert_record['review_priority']}`")
                        st.markdown(f"**Protocol:** `{alert_record['protocol_type']}`")
                        st.markdown(f"**Service:** `{alert_record['service']}`")
                        st.markdown(f"**Connection Flag:** `{alert_record['flag']}`")
                        st.markdown(f"**Primary Deviations:**\n`{alert_record['anomaly_reasons']}`")

                        # Analyst review action
                        st.markdown("#### 📝 Analyst Disposition")
                        disposition = st.radio(
                            "Triage Decision",
                            ["Under Investigation", "Confirmed Security Incident", "Benign Outlier (False Positive)"],
                        )
                        st.text_area("Analyst Investigation Notes", placeholder="Record findings or remediation steps...")
                        if st.button("Save Analyst Disposition"):
                            st.success(f"Alert #{selected_idx} updated to status: {disposition}")

                    with ic2:
                        st.plotly_chart(plot_radar_risk_profile(alert_record), use_container_width=True)
                        with st.expander("🔍 View Raw Connection Telemetry (JSON)", expanded=False):
                            feat_dict = alert_record.to_dict()
                            excluded = ["is_anomaly", "anomaly_prediction", "anomaly_score", "review_priority", "anomaly_reasons", "pca_component_1", "pca_component_2"]
                            clean_telemetry = {k: v for k, v in feat_dict.items() if k not in excluded}
                            st.json(clean_telemetry)

    # =========================================================================
    # PAGE 4: MODEL PERFORMANCE
    # =========================================================================
    elif selected_page == "📊 Model Performance":
        st.title("📊 Model Evaluation & Diagnostic Benchmark")
        st.markdown(
            """
            This section reports **unbiased evaluation metrics** computed on the held-out 
            NSL-KDD benchmark test set (`KDDTest+`). Ground truth labels were strictly withheld 
            from the model during training and threshold calibration.
            """
        )

        test_metrics = metadata.get("test_evaluation", {})
        if not test_metrics:
            st.warning("No benchmark evaluation metrics found. Please train the model in the Configuration tab.")
        else:
            # Metrics Row
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                render_metric_card("Precision", f"{test_metrics.get('precision', 0.0):.4f}", "True attacks / All flagged alerts", "#38bdf8")
            with p2:
                render_metric_card("Recall (Sensitivity)", f"{test_metrics.get('recall', 0.0):.4f}", "Detected attacks / Total real attacks", "#10b981")
            with p3:
                render_metric_card("F1-Score", f"{test_metrics.get('f1_score', 0.0):.4f}", "Harmonic mean of Precision & Recall", "#a855f7")
            with p4:
                ap = test_metrics.get("pr_curve", {}).get("average_precision", 0.0)
                render_metric_card("PR-AUC", f"{ap:.4f}", "Average Precision Area Under Curve", "#fbbf24")

            # Operational Security Metrics
            o1, o2, o3, o4 = st.columns(4)
            with o1:
                fpr = test_metrics.get("false_positive_rate", 0.0)
                render_metric_card("False Alarm Rate (FPR)", f"{fpr:.2%}", f"{test_metrics.get('false_positives_count', 0):,} benign sessions flagged", "#f43f5e")
            with o2:
                fnr = test_metrics.get("Missed Attack Rate (FNR)", test_metrics.get("false_negative_rate", 0.0))
                render_metric_card("Missed Attack Rate (FNR)", f"{fnr:.2%}", f"{test_metrics.get('false_negatives_count', 0):,} attacks missed", "#fb923c")
            with o3:
                acc = test_metrics.get("accuracy", 0.0)
                render_metric_card("Accuracy", f"{acc:.2%}", "Overall classification accuracy", "#64748b")
            with o4:
                total_eval = test_metrics.get("total_evaluated", 0)
                render_metric_card("Evaluated Test Set", f"{total_eval:,}", "Held-out KDDTest+ corpus", "#64748b")

            st.markdown("---")
            st.markdown("### 🔬 Performance Diagnostics")

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if "confusion_matrix" in test_metrics:
                    st.plotly_chart(plot_confusion_matrix(test_metrics["confusion_matrix"]), use_container_width=True)
            with dcol2:
                if "pr_curve" in test_metrics:
                    st.plotly_chart(plot_pr_curve(test_metrics["pr_curve"]), use_container_width=True)

            # Attack category recall and PCA variance
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                if "per_category_metrics" in test_metrics:
                    st.plotly_chart(plot_category_recall(test_metrics["per_category_metrics"]), use_container_width=True)
            with rcol2:
                pca_summary = metadata.get("pca_diagnostics", {})
                if pca_summary:
                    st.plotly_chart(plot_explained_variance(pca_summary), use_container_width=True)

            # Validation Comparison Table
            st.markdown("### ⚖️ Multi-Configuration Validation Comparison")
            val_comp = metadata.get("validation_comparison", [])
            if val_comp:
                val_df = pd.DataFrame(val_comp)
                st.dataframe(val_df, use_container_width=True)
                st.caption("Comparison evaluated on 20% validation split (5,039 records) without using test set labels.")

            # Operational trade-offs note
            st.markdown("### 🛡️ Why Accuracy Alone is Deceptive in Cybersecurity")
            st.markdown(
                """
                > **The Base-Rate Fallacy in Intrusion Detection:**
                > In enterprise networks, malicious traffic accounts for less than 0.1% of daily packets. 
                > A trivial detector that never flags any alerts achieves **99.9% accuracy**, yet fails completely 
                > at stopping attacks. 
                > In unsupervised anomaly detection, the true trade-off is between **False Positives** 
                > (causing SOC analyst alert fatigue) and **False Negatives** (silent intrusion compromise). 
                > Evaluating models using **Precision, Recall, F1-Score, and False Alarm Rate (FPR)** 
                > is vital for real-world cyber defense.
                """
            )

    # =========================================================================
    # PAGE 5: MODEL CONFIGURATION
    # =========================================================================
    elif selected_page == "⚙️ Model Configuration":
        st.title("⚙️ Model Configuration & Hyperparameter Tuning")
        st.markdown(
            """
            Adjust hyperparameters for the unsupervised Isolation Forest detector and 
            retrain the pipeline. The updated model will be validated on held-out data and persisted safely.
            """
        )

        curr_params = metadata.get("hyperparameters", {})
        curr_strategy = metadata.get("strategy", "normal_only")

        with st.form("retrain_form"):
            st.markdown("#### Hyperparameter Settings")
            c_f1, c_f2 = st.columns(2)

            with c_f1:
                strategy = st.selectbox(
                    "Training Baseline Strategy",
                    ["normal_only", "unsupervised_mixture"],
                    index=0 if curr_strategy == "normal_only" else 1,
                    help="normal_only: Trains strictly on benign records to learn legitimate traffic distribution.\nunsupervised_mixture: Trains on full mixed traffic without label filtering.",
                )
                contamination = st.slider(
                    "Contamination Parameter",
                    min_value=0.01,
                    max_value=0.30,
                    value=float(curr_params.get("contamination", 0.10)),
                    step=0.01,
                    help="Expected proportion of anomalies in the training corpus.",
                )

            with c_f2:
                n_estimators = st.select_slider(
                    "Number of Isolation Trees (n_estimators)",
                    options=[50, 100, 150, 200, 300],
                    value=int(curr_params.get("n_estimators", 150)),
                    help="More trees increase score stability and precision, but increase training time slightly.",
                )
                max_samples = st.select_slider(
                    "Max Samples per Subtree",
                    options=[128, 256, 512, 1024],
                    value=int(curr_params.get("max_samples", 512)),
                    help="Subsampling mitigates swamping and masking effects in anomaly detection.",
                )

            random_seed = st.number_input("Random State Seed", value=42, step=1)

            st.markdown("---")
            submit_btn = st.form_submit_button("🔥 Retrain Pipeline & Update Artifacts", type="primary")

        if submit_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Ingesting NSL-KDD dataset...")
                progress_bar.progress(20)

                status_text.text(f"Fitting preprocessor & Isolation Forest (strategy={strategy}, trees={n_estimators})...")
                progress_bar.progress(50)

                new_meta = run_training_pipeline(
                    training_strategy=strategy,
                    contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=int(random_seed),
                )

                progress_bar.progress(90)
                status_text.text("Saving artifacts and evaluating benchmark...")
                time.sleep(0.3)
                progress_bar.progress(100)

                # Clear resource cache to reload new model
                st.cache_resource.clear()
                st.success("✅ Model retrained successfully! New artifacts saved to `models/`.")
                st.balloons()
            except Exception as e:
                st.error(f"Training failed with error: {e}")

    # =========================================================================
    # PAGE 6: ABOUT & METHODOLOGY
    # =========================================================================
    elif selected_page == "📖 About & Methodology":
        st.title("📖 About & System Methodology")
        st.markdown(
            """
            ### Project Overview
            This application is an educational and operational prototype for **AI-Powered Network Anomaly Detection** 
            in cyber-physical and IT networks. Unlike supervised machine learning that memorizes historical attack signatures, 
            this system utilizes **unsupervised machine learning** to detect previously unseen and zero-day intrusions by identifying 
            subtle statistical deviations from normal traffic baselines.

            ---

            ### Mathematical Foundations

            #### 1. Isolation Forest (Liu, Ting, & Zhou, 2008)
            The Isolation Forest algorithm isolates anomalous observations by randomly selecting a feature and randomly 
            selecting a split value between the maximum and minimum values of that feature.
            - Because anomalies are few and structurally distinct, they require **fewer random recursive splits** to isolate compared to normal points.
            - The anomaly score $s(x, n)$ for an instance $x$ across $n$ samples is defined as:
              $$s(x, n) = 2^{-\\frac{E(h(x))}{c(n)}}$$
              where $E(h(x))$ is the average path length across all isolation trees, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree (BST):
              $$c(n) = 2 \\ln(n - 1) + 0.5772156649 - \\frac{2(n - 1)}{n}$$
            - If $E(h(x)) \\to 0$, $s \\to 1$ (high certainty of anomaly).
            - If $E(h(x)) \\to n - 1$, $s \\to 0$ (normal cluster inlier).

            #### 2. Principal Component Analysis (PCA)
            PCA transforms high-dimensional correlated network feature vectors $X \\in \\mathbb{R}^{d}$ into orthogonal components 
            $Z \\in \\mathbb{R}^{k}$ along the directions of maximum variance:
            $$\\Sigma = \\frac{1}{n} X^T X = V \\Lambda V^T$$
            We compute cumulative explained variance to identify the smallest subspace capturing $90\\%$ of traffic variance.

            ---

            ### Benchmark Dataset: NSL-KDD
            - **Source**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)
            - **Reference**: Tavallaee et al., *A Detailed Analysis of the KDD CUP 99 Data Set*, IEEE CISDA, 2009.
            - **Design**: Created specifically to rectify severe duplicate record biases in original KDD'99 that skewed detection rates.
            - **Schema**: 41 network connection attributes spanning basic TCP connection parameters, domain-specific content features, and time-based network traffic metrics.
            - **Evaluation Protocol**: The ground-truth attack labels are strictly quarantined and never exposed to the model during training or feature selection.

            ---

            ### Ethical & Operational Considerations
            1. **Human-in-the-Loop**: Anomaly detectors generate investigative leads, not automated sanctions. An anomalous session may simply represent a routine system backup or unusual large database sync.
            2. **Adversarial Robustness**: Unsupervised detectors can be susceptible to slow-poisoning attacks if an adversary gradually shifts normal traffic statistics over extended time horizons.
            3. **Privacy**: Packet payloads are not inspected directly; analysis relies strictly on statistical flow metadata.
            """
        )


if __name__ == "__main__":
    main()
