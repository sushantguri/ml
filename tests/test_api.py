"""
Unit Tests for FastAPI Scoring Microservice.
"""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_root_endpoint():
    """Test API root status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint():
    """Test health check and model loading state."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["dimensions"] > 0


def test_metadata_endpoint():
    """Test retrieval of model metadata."""
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "strategy" in data
    assert "hyperparameters" in data


def test_score_single_normal_connection():
    """Test scoring a typical benign HTTP session."""
    payload = {
        "duration": 0.0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 215.0,
        "dst_bytes": 450.0,
        "count": 2.0,
        "srv_count": 2.0,
        "serror_rate": 0.0,
        "diff_srv_rate": 0.0,
        "dst_host_count": 50.0,
        "dst_host_srv_count": 50.0,
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert data["review_priority"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def test_score_single_anomalous_connection():
    """Test scoring an anomalous high-rate flood connection."""
    payload = {
        "duration": 0.0,
        "protocol_type": "tcp",
        "service": "private",
        "flag": "S0",
        "src_bytes": 0.0,
        "dst_bytes": 0.0,
        "count": 511.0,
        "srv_count": 511.0,
        "serror_rate": 1.0,
        "diff_srv_rate": 0.0,
        "dst_host_count": 255.0,
        "dst_host_srv_count": 1.0,
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] == 1
    assert data["anomaly_score"] >= 0.50
    assert data["review_priority"] in {"CRITICAL", "HIGH"}


def test_score_batch_connections():
    """Test batch scoring endpoint with multiple connection records."""
    batch_payload = {
        "records": [
            {
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF",
                "src_bytes": 200.0,
                "dst_bytes": 500.0,
            },
            {
                "protocol_type": "tcp",
                "service": "private",
                "flag": "S0",
                "src_bytes": 0.0,
                "count": 500.0,
                "serror_rate": 1.0,
            },
        ]
    }
    response = client.post("/api/v1/score/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["results"]) == 2
    assert "anomalies_detected" in data
