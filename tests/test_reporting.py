"""
Unit Tests for Incident Reporting and MITRE ATT&CK Mapping Module.
"""

import pandas as pd
import pytest

from src.reporting import (
    MITRE_ATTACK_MAPPINGS,
    generate_incident_dossier,
    infer_threat_class,
)


@pytest.fixture
def sample_attack_record():
    """Create a sample DoS-like attack record."""
    return pd.Series({
        "protocol_type": "tcp",
        "service": "private",
        "flag": "S0",
        "src_bytes": 0,
        "dst_bytes": 0,
        "count": 280,
        "serror_rate": 1.0,
        "diff_srv_rate": 0.0,
        "anomaly_score": 0.9421,
        "review_priority": "CRITICAL",
        "anomaly_reasons": "count (+8.4σ); serror_rate (+12.1σ)",
    })


@pytest.fixture
def sample_probe_record():
    """Create a sample Probe-like attack record."""
    return pd.Series({
        "protocol_type": "tcp",
        "service": "other",
        "flag": "REJ",
        "src_bytes": 0,
        "dst_bytes": 0,
        "count": 12,
        "serror_rate": 0.0,
        "diff_srv_rate": 0.85,
        "anomaly_score": 0.7840,
        "review_priority": "HIGH",
        "anomaly_reasons": "diff_srv_rate (+6.2σ)",
    })


def test_infer_threat_class(sample_attack_record, sample_probe_record):
    """Test threat class inference logic."""
    assert infer_threat_class(sample_attack_record) == "dos"
    assert infer_threat_class(sample_probe_record) == "probe"


def test_generate_incident_dossier(sample_attack_record):
    """Test incident dossier Markdown report generation."""
    dossier = generate_incident_dossier(
        record=sample_attack_record,
        alert_id="42",
        analyst_notes="Verified SYN flood against external interface.",
        disposition="Confirmed Security Incident",
    )

    assert isinstance(dossier, str)
    assert "SEC-INC-42" in dossier
    assert "CRITICAL" in dossier
    assert "0.9421" in dossier
    assert "MITRE ATT&CK" in dossier
    assert "T1498" in dossier
    assert "NIST SP 800-61" in dossier
    assert "Verified SYN flood against external interface." in dossier


def test_mitre_mapping_structure():
    """Test integrity of MITRE ATT&CK taxonomy entries."""
    for data in MITRE_ATTACK_MAPPINGS.values():
        assert "tactic" in data
        assert "techniques" in data
        assert len(data["techniques"]) > 0
        assert "containment_guidance" in data
