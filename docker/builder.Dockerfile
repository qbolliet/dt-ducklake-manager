# Builder - For initial database creation and full operations
FROM base as builder

# Install additional dependencies for building
COPY requirements/builder.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/builder.txt

# Copy all scripts
COPY scripts/ ./scripts/
COPY config.yaml .
COPY parameters/ ./parameters/

# Volume for data persistence
VOLUME ["/data"]

# Default entrypoint for building
ENTRYPOINT ["python", "-m", "scripts.main"]
CMD ["--help"]

# Labels
LABEL purpose="database-builder" \
      description="Full database creation and management"