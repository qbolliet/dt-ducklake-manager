# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0](https://github.com/qbolliet/dt-ducklake-manager/compare/v0.3.1...v0.4.0) (2026-09-27)


### Features

* label_for logic ([fb84f9e](https://github.com/qbolliet/dt-ducklake-manager/commit/fb84f9e85485c97778ab3b57a19efd84e5c31657))


### Bug Fixes

* record physical SQL types and UTC timestamps in the metadata contract ([17b2ec4](https://github.com/qbolliet/dt-ducklake-manager/commit/17b2ec4664bd0600013f885341da25b8f209b390))
* update + delete bugs + simplify maintenance logic ([fbe766b](https://github.com/qbolliet/dt-ducklake-manager/commit/fbe766b8fe7873e73224abb9a9817dd80168dd7e))


### Documentation

* label for behavior in database builder ([9c29946](https://github.com/qbolliet/dt-ducklake-manager/commit/9c29946f2497ea3faec0ae4b4d359f6063effed7))
* label logic ([82fae86](https://github.com/qbolliet/dt-ducklake-manager/commit/82fae869a4184d81c3dc9ccf7ee24e2d3eb200cf))
* label specification + prompts ([90ce954](https://github.com/qbolliet/dt-ducklake-manager/commit/90ce9549c912f1d66249b094811521c7b6efa08a))
* remove specification markdowns ([71d8fc9](https://github.com/qbolliet/dt-ducklake-manager/commit/71d8fc9d275a2d5afa783d495e0aecb9a36d9a7f))
* review example notebooks ([6b196a5](https://github.com/qbolliet/dt-ducklake-manager/commit/6b196a5ce7773306ba16f880b1c11ac031391305))

## [0.3.1](https://github.com/qbolliet/dt-ducklake-manager/compare/v0.3.0...v0.3.1) (2026-09-18)


### Bug Fixes

* duplicate remove logic of null columns ([6dc8c07](https://github.com/qbolliet/dt-ducklake-manager/commit/6dc8c07c62618ed85fa69cb11cc8c2989b84c7dc))
* example notebooks ([b03b042](https://github.com/qbolliet/dt-ducklake-manager/commit/b03b04265b69438ee30865ad7596b1106aa5ec90))
* homogenous number types between sql and python ([1ba2b9d](https://github.com/qbolliet/dt-ducklake-manager/commit/1ba2b9dca1e77a611164a1e0b01df3931a9d7b3f))
* maintenance logic ([ad0faed](https://github.com/qbolliet/dt-ducklake-manager/commit/ad0faed571cb53d03728d874fbb4dc38b4308886))
* optimization parametrisation ([331b134](https://github.com/qbolliet/dt-ducklake-manager/commit/331b134bb3cec3c242eb9746906b58ad5c9c695c))
* quote columns + schema validation + logger path ([821a87d](https://github.com/qbolliet/dt-ducklake-manager/commit/821a87df52c00f90d2d1d9eaa49e51661f919336))
* type checking + metadata and categorical status updates ([0a1c252](https://github.com/qbolliet/dt-ducklake-manager/commit/0a1c25226719bc4424f6e89e42764a4723dcc4bb))


### Documentation

* readme Release-As: 0.4.0 ([54e2d55](https://github.com/qbolliet/dt-ducklake-manager/commit/54e2d5533de441ec84a5e129f88ec377964256e8))

## [0.3.0](https://github.com/qbolliet/dt-ducklake-manager/compare/v0.2.0...v0.3.0) (2026-08-28)


### Features

* add db creation if not exists ([bdeeaf3](https://github.com/qbolliet/dt-ducklake-manager/commit/bdeeaf3a40cb9d5196203b4eebee52751dce0166))
* safe default and dedicated admin role for catalog DB creation ([05424aa](https://github.com/qbolliet/dt-ducklake-manager/commit/05424aae079f491ddef1632edaee7928319c0a0b))
* safe default and dedicated admin role for catalog DB creation ([0d2d1e4](https://github.com/qbolliet/dt-ducklake-manager/commit/0d2d1e49aa6baae565f119a9b2bcd39c618a6270))


### Bug Fixes

* s3 + postgres sql connection ([b9c0d7a](https://github.com/qbolliet/dt-ducklake-manager/commit/b9c0d7a353d515bd64a1ddf223c1f5264cf3ba50))

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
