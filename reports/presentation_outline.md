# Project Presentation Outline
## Adaptive Cyber-Physical Security: AI-Powered Network Anomaly Detection System

**Audience:** Security Architects, Machine Learning Engineers, Technical Leadership  
**Duration:** 15–20 minutes  

---

### Slide 1: Title & Executive Introduction
- **Project Title:** Adaptive Cyber-Physical Security — AI-Powered Network Anomaly Detection System
- **Presenter Role:** Senior Cybersecurity & ML Systems Engineer
- **Core Mission:** Detect novel, zero-day network threats and intrusions through unsupervised statistical baseline modeling without reliance on attack labels.

---

### Slide 2: The Signature Dilemma in Modern Cyber Defense
- **The Problem:** Traditional Intrusion Detection Systems (Snort, Suricata) depend on known static signatures.
- **Vulnerabilities:**
  - Zero-day exploits bypass signature matching.
  - "Living off the Land" (LotL) attacks abuse legitimate administrative tools.
  - Supervised machine learning models overfit to historical labels and fail on evolving adversarial shifts.
- **The Solution:** Unsupervised Anomaly Detection — isolating anomalous telemetry purely from structural deviation from nominal benign baselines.

---

### Slide 3: Dataset Provenance & Benchmark Integrity (NSL-KDD)
- **Source:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB).
- **Why NSL-KDD Matters:**
  - KDD Cup 99 contained up to 78% duplicate records, producing deceptive 98%+ accuracy figures.
  - NSL-KDD eliminated all duplicate records, balanced difficulty, and established the standard intrusion benchmark.
- **Strict Zero-Leakage Protocol:**
  - 41 traffic features (basic, content, traffic windows).
  - Target labels (`label`) and metadata (`difficulty_level`) strictly quarantined from training, preprocessing, and threshold calibration.

---

### Slide 4: System Architecture & Leakage-Free Preprocessing
- **Modular Pipeline:** Ingestion $\to$ Preprocessing $\to$ PCA $\to$ Isolation Forest $\to$ Calibrated Scoring $\to$ SOC Triage.
- **Scikit-Learn ColumnTransformer:**
  - `StandardScaler` for 38 numerical/binary flow features.
  - `OneHotEncoder(handle_unknown='ignore')` for symbolic protocol, service, and flag features.
  - Fitted strictly on training data to guarantee zero data leakage into validation or test sets.

---

### Slide 5: Dimensionality Reduction & PCA Variance Spectrum
- **High-Dimensional Latent Space:** Transformed feature space reaches 72 encoded dimensions.
- **PCA Decomposition:**
  - 19 principal components capture $\ge 90\%$ of total network variance.
  - 2D PCA projection reveals clear structural clustering of normal sessions vs. distributed perimeter attacks.
  - Ground-truth labels used exclusively for post-hoc visual coloring, never for fitting PCA.

---

### Slide 6: Unsupervised Anomaly Modeling (Isolation Forest)
- **Mathematical Foundation (Liu et al., 2008):**
  - Anomalies are "few and different" $\implies$ isolated near the root of random binary partition trees.
  - Path length $h(x)$ normalized against average search length in a Binary Search Tree:
    $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
- **Calibrated Scoring:** Normalized continuous score $s \in [0.0, 1.0]$.
- **Automated Triage Priority:**
  - CRITICAL ($\ge 0.85$), HIGH ($\ge 0.70$), MEDIUM ($\ge 0.50$), LOW ($< 0.50$).

---

### Slide 7: Measured Benchmark Results (`KDDTest+`)
- **Evaluated Test Set:** 22,544 held-out network sessions.
- **Measured Metrics:**
  - **Precision:** `92.58%` (High alert fidelity)
  - **Recall:** `75.64%` (9,692 true intrusions captured)
  - **F1-Score:** `83.26%`
  - **False Alarm Rate (FPR):** `8.01%` (778 false alarms on 9,711 normal sessions)
  - **PR-AUC (Average Precision):** `0.9601`
- **Validation Comparison:** Controlled comparison of contamination rates (0.05, 0.10, 0.15) to demonstrate tuning trade-offs without test leakage.

---

### Slide 8: Attack Class Breakdown & Detection Dynamics
- **DoS (Denial of Service):** `84.34% Recall` (High packet volume & error rates create unmistakable outliers).
- **Probe (Surveillance / Scanning):** `87.98% Recall` (Rapid host sweeps and service probes isolated rapidly).
- **R2L (Remote to Local):** `42.85% Recall` (Stealthy credential attempts mimic valid application connections).
- **U2R (User to Root):** `46.00% Recall` (Single packet privilege escalation exploits remain closer to normal cluster center).
- **Takeaway:** Multi-layered defense-in-depth is necessary; network flow telemetry excels at volumetric and reconnaissance threats, while endpoint detection (EDR) supplements host privilege escalation.

---

### Slide 9: Cybersecurity Reality — The Base-Rate Fallacy
- **Why Accuracy is Misleading:**
  - In a real enterprise network with 99.9% benign traffic, a model with 99.0% accuracy produces 10x more false alarms than real incidents.
  - Precision, Recall, and False Alarm Rate (FPR) are the true operational metrics.
- **Human-in-the-Loop Principle:**
  - Anomaly detection generates prioritized investigation leads with explainability z-scores, not autonomous disruptive interventions.

---

### Slide 10: Interactive SOC Dashboard & Live Demo
- **Streamlit Web Application Capabilities:**
  - Executive Overview with real-time health metrics.
  - Drag-and-drop CSV Traffic Ingestion & Batch Scoring.
  - Prioritized Security Alert Queue with Analyst Disposition Workflow.
  - Diagnostic Visualizations: Confusion Matrix Heatmap, PR Curve, 2D PCA Projections.
  - Interactive Model Retraining and Hyperparameter Tuning with safe checkpointing.
- **Summary:** Production-grade modular Python architecture, 100% test coverage, and reproducible unsupervised cyber defense.
