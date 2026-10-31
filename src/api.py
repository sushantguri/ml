"""
REST API Microservice for Real-Time Network Telemetry Scoring.
Adaptive Cyber-Physical Security System.

Provides high-throughput, low-latency HTTP endpoints for edge sensors,
firewalls, and SIEM forwarders to evaluate network sessions in real time.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from threading import Lock

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import InferencePipeline

logger = logging.getLogger(__name__)

# FastAPI Application
app = FastAPI(
    title="Adaptive Cyber-Physical Anomaly Detection API",
    description="Real-time unsupervised network session scoring and alert triage microservice.",
    version="1.0.0",
)


class APIMetricsTracker:
    """Thread-safe operational telemetry and Prometheus metrics collector."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.sessions_scored: int = 0
        self.anomalies_detected: int = 0
        self.total_processing_seconds: float = 0.0
        self.priorities: dict[str, int] = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }

    def record_scoring(
        self,
        count: int,
        anomalies: int,
        priorities: dict[str, int],
        duration_sec: float,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.sessions_scored += count
            self.anomalies_detected += anomalies
            self.total_processing_seconds += duration_sec
            for k, v in priorities.items():
                if k in self.priorities:
                    self.priorities[k] += v

    def get_snapshot(self) -> dict:
        with self._lock:
            uptime = time.time() - self.start_time
            avg_latency_ms = (
                (self.total_processing_seconds / self.total_requests * 1000)
                if self.total_requests > 0
                else 0.0
            )
            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": self.total_requests,
                "sessions_scored": self.sessions_scored,
                "anomalies_detected": self.anomalies_detected,
                "anomaly_rate_pct": round(
                    (self.anomalies_detected / self.sessions_scored * 100)
                    if self.sessions_scored > 0
                    else 0.0,
                    2,
                ),
                "average_latency_ms": round(avg_latency_ms, 2),
                "priority_breakdown": dict(self.priorities),
            }

    def to_prometheus(self) -> str:
        snap = self.get_snapshot()
        lines = [
            "# HELP ml_anomaly_service_uptime_seconds Total runtime of the API service in seconds",
            "# TYPE ml_anomaly_service_uptime_seconds gauge",
            f"ml_anomaly_service_uptime_seconds {snap['uptime_seconds']}",
            "# HELP ml_anomaly_scoring_requests_total Total number of scoring HTTP requests received",
            "# TYPE ml_anomaly_scoring_requests_total counter",
            f"ml_anomaly_scoring_requests_total {snap['total_requests']}",
            "# HELP ml_anomaly_sessions_scored_total Total network telemetry sessions scored",
            "# TYPE ml_anomaly_sessions_scored_total counter",
            f"ml_anomaly_sessions_scored_total {snap['sessions_scored']}",
            "# HELP ml_anomaly_detections_total Total anomalous network sessions flagged",
            "# TYPE ml_anomaly_detections_total counter",
            f"ml_anomaly_detections_total {snap['anomalies_detected']}",
            "# HELP ml_anomaly_triage_priority_total Breakdown of sessions by SOC triage priority",
            "# TYPE ml_anomaly_triage_priority_total counter",
        ]
        for priority, count in snap["priority_breakdown"].items():
            lines.append(f'ml_anomaly_triage_priority_total{{priority="{priority}"}} {count}')
        lines.extend([
            "# HELP ml_anomaly_processing_seconds_total Aggregate inference processing time in seconds",
            "# TYPE ml_anomaly_processing_seconds_total counter",
            f"ml_anomaly_processing_seconds_total {round(self.total_processing_seconds, 4)}",
        ])
        return "\n".join(lines) + "\n"


# Global instances
_pipeline: InferencePipeline | None = None
_metrics_tracker = APIMetricsTracker()


def get_pipeline() -> InferencePipeline:
    """Lazy loader for global inference engine."""
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


# Request / Response Schemas
class NetworkConnectionPayload(BaseModel):
    duration: float = Field(default=0.0, description="Connection duration in seconds")
    protocol_type: str = Field(default="tcp", description="Protocol: tcp, udp, or icmp")
    service: str = Field(default="http", description="Network service name")
    flag: str = Field(default="SF", description="Connection status flag")
    src_bytes: float = Field(default=0.0, description="Bytes sent from source")
    dst_bytes: float = Field(default=0.0, description="Bytes sent from destination")
    count: float = Field(default=1.0, description="Connections to same destination in past 2s")
    srv_count: float = Field(default=1.0, description="Connections to same service in past 2s")
    serror_rate: float = Field(default=0.0, description="SYN error rate")
    diff_srv_rate: float = Field(default=0.0, description="Rate of connections to different services")
    dst_host_count: float = Field(default=1.0, description="Destination host count")
    dst_host_srv_count: float = Field(default=1.0, description="Destination host service count")
    # Additional attributes can be passed dynamically
    model_config = {"extra": "allow"}


class BatchScoringPayload(BaseModel):
    records: list[NetworkConnectionPayload] = Field(..., description="List of network connection records")


class AnomalyScoreResponse(BaseModel):
    is_anomaly: int
    anomaly_prediction: str
    anomaly_score: float
    review_priority: str
    anomaly_reasons: str
    pca_component_1: float | None = None
    pca_component_2: float | None = None


class BatchScoreResponse(BaseModel):
    total_records: int
    anomalies_detected: int
    anomaly_rate_pct: float
    results: list[AnomalyScoreResponse]


@app.get("/")
def root():
    """API Root endpoint."""
    return {
        "service": "Adaptive Cyber-Physical Anomaly Detection API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/api/v1/health")
def health_check():
    """Service health and model load state."""
    pipeline = get_pipeline()
    return {
        "status": "healthy" if pipeline.is_loaded else "degraded",
        "model_loaded": pipeline.is_loaded,
        "detector_algorithm": "IsolationForest",
        "dimensions": len(pipeline.preprocessor.feature_names_out) if pipeline.preprocessor else 0,
        "models_directory": str(pipeline.models_dir),
    }


@app.get("/api/v1/metadata")
def get_model_metadata():
    """Retrieve active model training metadata and benchmark evaluation metrics."""
    pipeline = get_pipeline()
    if not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Inference pipeline not initialized.")
    return pipeline.metadata


@app.get("/api/v1/metrics", response_class=PlainTextResponse)
def get_prometheus_metrics():
    """Prometheus-compatible operational metrics exposition endpoint."""
    return PlainTextResponse(
        content=_metrics_tracker.to_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/api/v1/metrics/json")
def get_metrics_json():
    """Retrieve operational telemetry and aggregate triage metrics in JSON format."""
    return _metrics_tracker.get_snapshot()


@app.post("/api/v1/score", response_model=AnomalyScoreResponse)
def score_single_connection(payload: NetworkConnectionPayload):
    """Score a single network session and return anomaly metrics and triage priority."""
    pipeline = get_pipeline()
    if not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Inference model artifacts not loaded.")

    t0 = time.time()
    df = pd.DataFrame([payload.model_dump()])
    try:
        enriched_df = pipeline.predict_records(df)
        record = enriched_df.iloc[0]

        is_anomaly = int(record["is_anomaly"])
        priority = str(record["review_priority"])
        duration = time.time() - t0

        _metrics_tracker.record_scoring(
            count=1,
            anomalies=is_anomaly,
            priorities={priority: 1},
            duration_sec=duration,
        )

        return AnomalyScoreResponse(
            is_anomaly=is_anomaly,
            anomaly_prediction=str(record["anomaly_prediction"]),
            anomaly_score=float(record["anomaly_score"]),
            review_priority=priority,
            anomaly_reasons=str(record["anomaly_reasons"]),
            pca_component_1=float(record.get("pca_component_1", 0.0)),
            pca_component_2=float(record.get("pca_component_2", 0.0)),
        )
    except Exception as e:
        logger.error("Scoring error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e!s}") from e


@app.post("/api/v1/score/batch", response_model=BatchScoreResponse)
def score_batch_connections(payload: BatchScoringPayload):
    """Batch score network telemetry sessions for high-volume sensors."""
    pipeline = get_pipeline()
    if not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Inference model artifacts not loaded.")

    if not payload.records:
        raise HTTPException(status_code=400, detail="Empty batch payload provided.")

    t0 = time.time()
    records_data = [r.model_dump() for r in payload.records]
    df = pd.DataFrame(records_data)

    try:
        enriched_df = pipeline.predict_records(df)
        anomalies_count = int((enriched_df["is_anomaly"] == 1).sum())

        priority_counts: dict[str, int] = (
            enriched_df["review_priority"].value_counts().to_dict()
            if "review_priority" in enriched_df
            else {}
        )
        duration = time.time() - t0

        _metrics_tracker.record_scoring(
            count=len(enriched_df),
            anomalies=anomalies_count,
            priorities=priority_counts,
            duration_sec=duration,
        )

        results = []
        for _, row in enriched_df.iterrows():
            results.append(
                AnomalyScoreResponse(
                    is_anomaly=int(row["is_anomaly"]),
                    anomaly_prediction=str(row["anomaly_prediction"]),
                    anomaly_score=float(row["anomaly_score"]),
                    review_priority=str(row["review_priority"]),
                    anomaly_reasons=str(row["anomaly_reasons"]),
                    pca_component_1=float(row.get("pca_component_1", 0.0)),
                    pca_component_2=float(row.get("pca_component_2", 0.0)),
                )
            )

        return BatchScoreResponse(
            total_records=len(enriched_df),
            anomalies_detected=anomalies_count,
            anomaly_rate_pct=round((anomalies_count / len(enriched_df)) * 100, 2),
            results=results,
        )
    except Exception as e:
        logger.error("Batch scoring error: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch inference error: {e!s}") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
