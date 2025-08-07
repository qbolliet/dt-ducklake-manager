# Updater - Lightweight image for model updates only
FROM python:3.11-slim

# Minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only essential requirements for updating
COPY requirements/base.txt /tmp/
RUN pip install --no-cache-dir \
    duckdb \
    pandas \
    numpy \
    pyyaml

# Copy only the necessary modules for updating
COPY dashboard_template_database/operations/ ./dashboard_template_database/operations/
COPY dashboard_template_database/manager/ ./dashboard_template_database/manager/
COPY dashboard_template_database/utils/ ./dashboard_template_database/utils/
COPY dashboard_template_database/__init__.py ./dashboard_template_database/

# Lightweight update script
COPY scripts/model_updater.py ./

# Create directories
RUN mkdir -p /data/databases /data/input /data/logs

# Environment
ENV PYTHONUNBUFFERED=1 \
    DATABASE_PATH=/data/databases \
    INPUT_PATH=/data/input \
    LOG_PATH=/data/logs

# Volume for database access
VOLUME ["/data"]

# Focused entrypoint for updates
ENTRYPOINT ["python", "model_updater.py"]

# Labels
LABEL purpose="database-updater" \
      description="Lightweight updater for model results" \
      size="minimal"