"""
Automated Security Incident Reporting & MITRE ATT&CK Mapping Module.
Adaptive Cyber-Physical Security System.

Provides:
- Automated extraction of Indicators of Compromise (IoCs)
- MITRE ATT&CK Enterprise Matrix technique mapping
- NIST SP 800-61 Rev. 2 compliant Incident Response (IR) recommendations
- Generation of SOC Incident Investigation Dossiers (Markdown/JSON)
"""

from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


# MITRE ATT&CK mapping for network anomaly indicators
MITRE_ATTACK_MAPPINGS = {
    "dos": {
        "tactic": "Impact (TA0040)",
        "techniques": [
            {"id": "T1498.001", "name": "Network Denial of Service: Direct Network Flood"},
            {"id": "T1499.002", "name": "Endpoint Denial of Service: Service Exhaustion"},
        ],
        "containment_guidance": "Deploy automated rate-limiting or blackhole routing at perimeter routers. Filter aberrant ICMP/UDP flows.",
    },
    "probe": {
        "tactic": "Reconnaissance / Discovery (TA0043 / TA0007)",
        "techniques": [
            {"id": "T1595.001", "name": "Active Scanning: Scanning IP Blocks"},
            {"id": "T1046", "name": "Network Service Discovery"},
        ],
        "containment_guidance": "Block offending source IP at network edge. Review exposed listener ports and close non-essential services.",
    },
    "r2l": {
        "tactic": "Initial Access / Credential Access (TA0001 / TA0006)",
        "techniques": [
            {"id": "T1110.001", "name": "Brute Force: Password Guessing"},
            {"id": "T1190", "name": "Exploit Public-Facing Application"},
        ],
        "containment_guidance": "Enforce MFA and temporary account lockouts. Inspect authentication logs for credential stuffing patterns.",
    },
    "u2r": {
        "tactic": "Privilege Escalation (TA0004)",
        "techniques": [
            {"id": "T1068", "name": "Exploitation for Privilege Escalation"},
            {"id": "T1548", "name": "Abuse Elevation Control Mechanism"},
        ],
        "containment_guidance": "Immediately isolate host endpoint from lateral network subnets. Capture memory dump for forensic analysis.",
    },
    "generic": {
        "tactic": "Unclassified Anomaly (TA0000)",
        "techniques": [
            {"id": "T1071", "name": "Standard Application Layer Protocol Anomaly"},
        ],
        "containment_guidance": "Correlate host telemetry with proxy logs. Verify if activity corresponds to planned administrative change.",
    },
}


def infer_threat_class(record: pd.Series) -> str:
    """Infer likely threat classification based on record attributes."""
    # Check if ground truth or predicted category exists
    if "label" in record:
        lbl = str(record["label"]).lower()
        if "neptune" in lbl or "smurf" in lbl or "teardrop" in lbl or "pod" in lbl:
            return "dos"
        elif "satan" in lbl or "sweep" in lbl or "nmap" in lbl:
            return "probe"
        elif "guess" in lbl or "warez" in lbl or "ftp" in lbl:
            return "r2l"
        elif "root" in lbl or "buffer" in lbl:
            return "u2r"

    # Attribute heuristics for unsupervised triage
    serror = float(record.get("serror_rate", 0.0))
    count = float(record.get("count", 0.0))
    diff_srv = float(record.get("diff_srv_rate", 0.0))
    num_compromised = float(record.get("num_compromised", 0.0))
    root_shell = float(record.get("root_shell", 0.0))

    if root_shell > 0 or num_compromised > 0:
        return "u2r"
    if serror > 0.5 and count > 100:
        return "dos"
    if diff_srv > 0.5:
        return "probe"
    return "generic"


def generate_incident_dossier(
    record: pd.Series,
    alert_id: Optional[str | int] = None,
    analyst_notes: Optional[str] = None,
    disposition: Optional[str] = None,
) -> str:
    """Generate a formal NIST SP 800-61 compliant SOC Incident Dossier in Markdown.
    
    Args:
        record: Pandas Series representing the anomalous connection.
        alert_id: Identifier for the alert.
        analyst_notes: Optional human investigation notes.
        disposition: Analyst disposition status.
        
    Returns:
        Formatted Markdown incident report string.
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    aid = f"SEC-INC-{alert_id}" if alert_id is not None else "SEC-INC-UNASSIGNED"
    score = float(record.get("anomaly_score", 0.0))
    priority = str(record.get("review_priority", "HIGH"))
    threat_class = infer_threat_class(record)
    mitre = MITRE_ATTACK_MAPPINGS.get(threat_class, MITRE_ATTACK_MAPPINGS["generic"])

    proto = str(record.get("protocol_type", "unknown")).upper()
    service = str(record.get("service", "unknown"))
    flag = str(record.get("flag", "unknown"))
    src_bytes = record.get("src_bytes", 0)
    dst_bytes = record.get("dst_bytes", 0)
    reasons = str(record.get("anomaly_reasons", "Multivariate latent distance threshold exceeded"))

    report = f"""# 🛡️ SOC Incident Investigation Dossier
**Incident Identifier:** `{aid}`  
**Generated Timestamp:** `{now_utc}`  
**Classification:** Operational Security Event — Human Review Required  

---

## 1. Executive Incident Summary
- **Triage Severity Priority:** `{priority}`
- **Normalized Anomaly Score:** `{score:.4f}` / 1.0000
- **Inferred Threat Family:** `{threat_class.upper()}`
- **Current Analyst Disposition:** `{disposition or "Pending Tier-2 Verification"}`

> **Statistical Anomaly Finding:**  
> The network connection exhibited significant multivariate deviation from legitimate baseline profiles.  
> **Key Outlier Deviations:** `{reasons}`

---

## 2. Technical Connection Telemetry
| Telemetry Parameter | Value | Baseline Reference |
| :--- | :--- | :--- |
| **Transport Protocol** | `{proto}` | Standard TCP/UDP/ICMP stack |
| **Target Service** | `{service}` | Expected network service port |
| **TCP Connection Flag** | `{flag}` | Normal connection completion (`SF`) |
| **Source Data Volume** | `{src_bytes:,} bytes` | Normal mean: ~1,500 bytes |
| **Destination Data Volume** | `{dst_bytes:,} bytes` | Normal mean: ~2,500 bytes |
| **Connection Rate Counter** | `{record.get('count', 'N/A')}` | Normal rate: < 10 conns / window |
| **SYN Error Rate** | `{record.get('serror_rate', 0.0):.2f}` | Expected normal: 0.00 |

---

## 3. MITRE ATT&CK® Threat Mapping
- **Tactical Category:** `{mitre['tactic']}`
- **Mapped Techniques:**
"""
    for tech in mitre["techniques"]:
        report += f"  - **{tech['id']}**: {tech['name']}\n"

    report += f"""
---

## 4. Recommended Incident Containment Procedures (NIST SP 800-61)
1. **Immediate Containment:**
   - {mitre['containment_guidance']}
2. **Telemetry Correlation:**
   - Query perimeter firewall logs for concurrent egress attempts from target service `{service}`.
   - Inspect local host authorization logs for anomalous lateral movements.
3. **Eradication & Recovery:**
   - Verify service integrity and validate if configuration adjustments are required.

---

## 5. Security Analyst Disposition & Audit Trail
- **Assigned Analyst Notes:**
> {analyst_notes if analyst_notes else "No notes recorded. Initial automated dossier generation."}

*Report generated by Adaptive Cyber-Physical Security System v1.0.0*
"""
    return report
