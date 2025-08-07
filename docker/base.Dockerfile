# Base image with common dependencies
FROM python:3.13-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy base requirements
COPY requirements/base.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the package
COPY dashboard_template_database/ ./dashboard_template_database/
COPY setup.py .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /data/databases /data/logs /data/temp

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DATABASE_PATH=/data/databases \
    LOG_PATH=/data/logs