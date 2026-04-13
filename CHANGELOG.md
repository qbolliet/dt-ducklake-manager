# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-13

### Features

- Build DuckDB databases from tabular data with schema inference
- Fact table, metadata table and dimension tables architecture
- Atomic update, delete and merge operations
- Schema persistence and inference from Polars/PyArrow frames
- Database maintenance : auditing, compaction and recovery
- Indexes support for query optimization
- DuckLake-compatible manager (`DtDucklakeManager`)
- Structured logging via `dt_ducklake_manager.utils.logger`
