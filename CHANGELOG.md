# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0](https://github.com/qbolliet/dt-ducklake-manager/compare/v0.1.0...v0.2.0) (2026-05-29)


### Features

* add multi-schema support ([d40488a](https://github.com/qbolliet/dt-ducklake-manager/commit/d40488a7b74c6e596441110dfbe626030d76b44d))
* add postgres catalog backend option ([698ea79](https://github.com/qbolliet/dt-ducklake-manager/commit/698ea799954299a97f184ee013fc0ed40c3d1cba))
* postgres catalog backend + CI tooling and automated releases ([064d1d9](https://github.com/qbolliet/dt-ducklake-manager/commit/064d1d90da114c595e1b9253dc8406f13a09d3e9))


### Bug Fixes

* null dimension labels ([9786726](https://github.com/qbolliet/dt-ducklake-manager/commit/9786726a81f675d2965dcc3d2bdc83580a0fb99e))
* null dimension labels ([357a0a0](https://github.com/qbolliet/dt-ducklake-manager/commit/357a0a05060c2f5a1c09a866e96336340af0d949))
* remove ipykernel and upgrade dependencies to fix security vulnerabilities ([21083a8](https://github.com/qbolliet/dt-ducklake-manager/commit/21083a89ebaffc009eb147a9a6900999ffc12ebd))


### Documentation

* add new architecture section ([792665a](https://github.com/qbolliet/dt-ducklake-manager/commit/792665a0aad2629a7d3b2161a82d4d27e6c26310))
* add new architecture section ([3ba28cf](https://github.com/qbolliet/dt-ducklake-manager/commit/3ba28cf908c49157a41f6d7f1b0a90949c162a76))
* fix import handler -&gt; inventories ([e44976a](https://github.com/qbolliet/dt-ducklake-manager/commit/e44976a42c7b64b545e58498b2155b9d184bba5b))
* fix import handler -&gt; inventories ([bc216a6](https://github.com/qbolliet/dt-ducklake-manager/commit/bc216a612654734803fd2f463a86d4dda22a6a9d))
* fix import python handler + linux paths ([b3fa030](https://github.com/qbolliet/dt-ducklake-manager/commit/b3fa030efa51b5abb4b0e7f861d31578295d3b1c))
* fix import python handler + linux paths ([34b39f7](https://github.com/qbolliet/dt-ducklake-manager/commit/34b39f745659ea182322efd2f7a9a322fb12044a))
* move import to plugin level ([70c95b2](https://github.com/qbolliet/dt-ducklake-manager/commit/70c95b29a453dd58552e518ee8d3e5f78b96d414))
* move import to plugin level ([a257f26](https://github.com/qbolliet/dt-ducklake-manager/commit/a257f26bddda5c27aed7ef1145bc2a60e3a16b69))
* schema architecture ([4288ab6](https://github.com/qbolliet/dt-ducklake-manager/commit/4288ab62aa3caf7d5ac9a0266464cc3c32d9d1ea))

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
