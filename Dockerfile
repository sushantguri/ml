# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL maintainer="Sushant Guri <sushantv355@gmail.com>"
LABEL description="Adaptive Cyber-Physical Security - AI Network Anomaly Detection System"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# Install system utilities for healthchecks and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application codebase and artifacts
COPY config/ ./config/
COPY models/ ./models/
COPY data/samples/ ./data/samples/
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY reports/ ./reports/
COPY app.py demo.py pytest.ini ./

EXPOSE 8000 8501

# Default command launches the interactive Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
