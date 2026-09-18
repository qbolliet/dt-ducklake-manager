"""
Physical maintenance of a DuckLake catalog.

DuckLake files are immutable : every UPDATE/DELETE
produces new Parquet delta files and, for a partial deletion, a
``-delete.parquet`` tombstone file, while the old file stays referenced by prior
snapshots (time travel). Left alone, small files and tombstones accumulate and
degrade read performance. Per procedure — effect / when / risk:

- ``rewrite_data_files(delete_threshold)`` — rewrites files whose deleted-row
  share exceeds the threshold; **without an explicit threshold it is a no-op**.
  After every update/delete
  (``delete_threshold`` 0.1-0.3). Risk: none (old files stay readable via time
  travel).
- ``merge_files(min_file_size)`` — merges adjacent files smaller than
  ``min_file_size``. Once many small batches have accumulated; after
  ``recluster``. Risk: none.
- ``flush_inlined_data`` — writes inlined catalog rows out to Parquet. Planned
  maintenance; before reading files directly. Risk: none.
- ``recluster(order_by)`` — rewrites the whole table in
  ``cluster_by`` order. When file overlap degrades pruning, typically after N
  updates. Risk: full rewrite, doubles space until cleanup.
- ``repartition`` — changes partitioning and rewrites. Filter strategy change.
  Risk: full rewrite, doubles space until cleanup.
- ``expire_snapshots(older_than)`` — makes snapshots older than the cutoff
  unreachable. **Planned maintenance only**, explicit retention (days). Risk:
  **destroys time travel** beyond the retention.
- ``cleanup_files`` — deletes files no snapshot references anymore. After
  ``expire_snapshots``; the only step that actually frees space (measured).
  Risk: irreversible.
- ``delete_orphaned_files`` — deletes files under ``data_path`` unknown to the
  catalog. After an incident (interrupted transaction, manual copy); always
  ``dry_run`` first. Risk: irreversible.

Cycle: *rewrite* (rewrite/merge/flush) after every write -> *expire* and
*cleanup* only in planned maintenance. Deciding which step is worth running is the
job of :meth:`DuckLakeMaintenance.maintain`: it reads
:meth:`DuckLakeMaintenance.storage_report` (file/delete/small-file counts, inlined
rows, snapshots, file-range overlap on the first ``cluster_by`` column) and only
runs a step when its indicator justifies it under a :class:`MaintenancePolicy`,
logging every skipped step and why. The write operations (``DatabaseUpdater``/
``DatabaseDeleter``) only run the safe post-write steps through
:meth:`DuckLakeMaintenance.compact` (merge + rewrite) and never call
``expire_snapshots``/``cleanup_files``/``delete_orphaned_files``.

DuckLake has no index: pruning relies on per-file min/max statistics
(``ducklake_file_column_stats``, used by the planner — ``Total Files Read`` in
``EXPLAIN ANALYZE``) and on Parquet row-group statistics. Both only help when the
data is physically grouped, which successive updates degrade — hence
:meth:`DuckLakeMaintenance.recluster`.
"""

# Importation des modules
# Modules de base
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

# DuckDB
import duckdb

# Rapport d'opération
from ..reporting import (
    OperationReport,
    _current_snapshot_id,
    _set_commit_message,
    _table_info,
)

# Module d'initialisation du logger
from ..utils.logger import _init_logger
from ..utils.sql import qualify_table, quote_ident

# Taille de repli (octets) d'un fichier cible lorsque ``target_file_size`` n'est pas
# lisible dans ``ducklake_options`` (valeur par défaut de DuckLake : 512MB n'est pas
# garantie selon les versions, d'où un repli explicite aligné sur la recommandation)
_FALLBACK_TARGET_FILE_SIZE: int = 100_000_000


# Dataclass du rapport d'état du stockage d'une table
@dataclass
class StorageReport:
    """
    Physical storage state of one DuckLake table, used to decide on maintenance.

    Every indicator is measured from the catalog (``ducklake_table_info``, the
    internal ``__ducklake_metadata_<alias>`` tables, ``ducklake_snapshots``), never
    estimated. A measurement that cannot be read (no DuckLake catalog attached,
    unknown table) keeps its "nothing to do" default.

    Attributes:
        schema (str): DuckLake schema of the table.
        table (str): Bare table name.
        file_count (int): Active Parquet data files.
        total_bytes (int): Total size (bytes) of the active data files.
        delete_file_count (int): Active delete (tombstone) files.
        delete_bytes (int): Total size (bytes) of the active delete files.
        delete_ratio (float): ``delete_bytes / total_bytes`` (``0.0`` for an empty
            table) — a size-based proxy of the deleted share.
        small_file_count (int): Active data files smaller than
            ``min_file_size_bytes`` (empty residual files included).
        min_file_size_bytes (int): Threshold used for ``small_file_count``.
        inlined_rows (int | None): Rows still inlined in the catalog (not flushed
            to Parquet), or ``None`` when only the boolean indicator could be read.
        has_inlined_data (bool): Whether unflushed inlined rows exist.
        snapshot_count (int): Snapshots in the catalog (catalog-wide).
        oldest_snapshot_age_days (float | None): Age, in days, of the oldest
            snapshot still reachable (catalog-wide).
        overlap_ratio (float | None): Share of active non-empty files whose
            ``[min, max]`` range on ``cluster_column`` overlaps another file's
            range (strict inequalities: files sharing only a boundary value do not
            overlap; identical ranges do). ``None`` without ``cluster_by``.
        cluster_column (str | None): First ``cluster_by`` column, used for
            ``overlap_ratio``.

    Examples:
        >>> r = StorageReport(schema="main", table="fact_table", file_count=3)
        >>> r.summary().startswith("storage main.fact_table: 3 files")
        True
    """

    schema: str
    table: str
    file_count: int = 0
    total_bytes: int = 0
    delete_file_count: int = 0
    delete_bytes: int = 0
    delete_ratio: float = 0.0
    small_file_count: int = 0
    min_file_size_bytes: int = 100_000_000
    inlined_rows: int | None = None
    has_inlined_data: bool = False
    snapshot_count: int = 0
    oldest_snapshot_age_days: float | None = None
    overlap_ratio: float | None = None
    cluster_column: str | None = None

    # Méthode de construction de la ligne de synthèse lisible
    def summary(self) -> str:
        """
        Build a one-line, human-readable summary of the storage state.

        Returns:
            str: e.g. ``"storage main.fact_table: 3 files (2.4 MB), 0 delete files
            (0.0%), 3 small files, 0 inlined rows, 12 snapshots (oldest 4.2 days),
            overlap 0.67 on k"``.

        Examples:
            >>> StorageReport(schema="main", table="t").summary()  # doctest: +ELLIPSIS
            'storage main.t: 0 files (0.0 MB), 0 delete files (0.0%), ...'
        """
        # Lignes inlinées : nombre exact ou simple indicateur
        if self.inlined_rows is not None:
            inlined = f"{self.inlined_rows} inlined rows"
        else:
            inlined = "inlined data" if self.has_inlined_data else "no inlined data"
        # Âge du plus ancien snapshot
        age = (
            f" (oldest {self.oldest_snapshot_age_days:.1f} days)"
            if self.oldest_snapshot_age_days is not None
            else ""
        )
        # Recouvrement
        overlap = (
            f"overlap {self.overlap_ratio:.2f} on {self.cluster_column}"
            if self.overlap_ratio is not None
            else "overlap n/a (no cluster_by)"
        )
        return (
            f"storage {self.schema}.{self.table}: {self.file_count} files"
            f" ({self.total_bytes / 1_000_000:.1f} MB), {self.delete_file_count}"
            f" delete files ({self.delete_ratio:.1%}), {self.small_file_count} small"
            f" files, {inlined}, {self.snapshot_count} snapshots{age}, {overlap}"
        )


# Dataclass de la politique de maintenance
@dataclass
class MaintenancePolicy:
    """
    Thresholds deciding which maintenance steps :meth:`DuckLakeMaintenance.maintain`
    runs.

    Every destructive or costly behaviour is opt-in: by default snapshots never
    expire, orphaned files are never deleted and the table is never reclustered.

    Attributes:
        delete_threshold (float): Passed to ``rewrite_data_files``; the rewrite step
            runs when at least one delete file exists. Defaults to ``0.1``.
        min_file_size_bytes (int): A data file below this size (bytes) is "small";
            also passed to ``merge_files``. Defaults to ``100_000_000`` (aligned on
            the recommended ``target_file_size``).
        max_small_files (int): The merge step runs when more than this many small
            files exist. Defaults to ``10``.
        flush_inlined (bool): Flush inlined rows when some exist. Defaults to
            ``True``.
        max_overlap_ratio (float): The recluster step runs when ``overlap_ratio``
            exceeds it (and ``recluster`` is True). Defaults to ``0.5``.
        recluster (bool): Allow the (full rewrite) recluster step. Defaults to
            ``False``: opt-in.
        retention_days (int | None): Snapshot retention for expire + cleanup;
            ``None`` (default) never expires anything.
        delete_orphaned (bool): Run ``delete_orphaned_files``. Defaults to ``False``.
        dry_run (bool): Log what would run without modifying anything
            (expire/cleanup/delete_orphaned are called with ``dry_run=True``, which
            only lists). Defaults to ``False``.

    Raises:
        ValueError: If a ratio is outside ``[0, 1]`` or a count/size/retention is
            negative.

    Examples:
        >>> MaintenancePolicy().recluster
        False
        >>> MaintenancePolicy(retention_days=30, recluster=True).retention_days
        30
    """

    delete_threshold: float = 0.1
    min_file_size_bytes: int = 100_000_000
    max_small_files: int = 10
    flush_inlined: bool = True
    max_overlap_ratio: float = 0.5
    recluster: bool = False
    retention_days: int | None = None
    delete_orphaned: bool = False
    dry_run: bool = False

    # Validation des bornes
    def __post_init__(self) -> None:
        """Validate the policy thresholds.

        Raises:
            ValueError: If a ratio is outside ``[0, 1]`` or a count is negative.
        """
        # Vérification des ratios
        for name in ("delete_threshold", "max_overlap_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        # Vérification des entiers positifs
        for name in ("min_file_size_bytes", "max_small_files"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.retention_days is not None and self.retention_days < 0:
            raise ValueError(
                f"retention_days must be >= 0 or None, got {self.retention_days}"
            )


# Classe de maintenance d'un catalogue DuckLake
class DuckLakeMaintenance:
    """
    Runs maintenance operations on a DuckLake catalog to keep read performance optimal.

    DuckLake writes small Parquet delta files and delete tombstone files for every
    UPDATE/DELETE/MERGE operation. Over time, many small files accumulate and degrade
    sequential read performance. This class wraps DuckLake's maintenance procedures
    that compact those files back into larger, efficient Parquet files, flush inlined
    rows, and (planned maintenance only) reclaim space from expired snapshots.

    All methods are non-fatal: errors are logged as warnings and execution continues,
    so a failure in one step does not prevent the remaining steps from running.

    Attributes:
        conn (duckdb.DuckDBPyConnection): DuckDB connection with the DuckLake catalog
            already attached.
        catalog_alias (str): Alias used in the ``ATTACH`` statement (default ``'db'``).
        schema (str): DuckLake schema this maintenance instance is associated with,
            carried alongside ``catalog_alias`` so the catalog alias and the schema
            always travel together (parity with the other schema-aware managers).
        logger (logging.Logger): Logger instance.

    Examples:
        >>> from dt_ducklake_manager.connection import DuckLakeConnector
        >>> from dt_ducklake_manager.maintenance import DuckLakeMaintenance
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
        >>> maint = DuckLakeMaintenance(conn)
        >>> maint.maintain(MaintenancePolicy(retention_days=30))
        >>> # Instance liée à un schéma dédié du même catalogue
        >>> maint = DuckLakeMaintenance(conn, schema='predictions')
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        catalog_alias: str = "db",
        schema: str = "main",
        log_filename: str | os.PathLike[str] | None = None,
    ) -> None:
        """
        Initialize the DuckLakeMaintenance manager.

        Args:
            connection (duckdb.DuckDBPyConnection): DuckDB connection with the
                DuckLake catalog attached. The connection must have been created
                via ``DuckLakeConnector.connect()`` or equivalent.
            catalog_alias (str): Alias used in the ``ATTACH`` statement. Must match
                the alias passed to ``DuckLakeConnector``. Defaults to ``'db'``.
            schema (str): DuckLake schema this instance is associated with, carried
                alongside ``catalog_alias`` so the catalog alias and the schema
                always travel together. Defaults to ``'main'``.
            log_filename (Optional[os.PathLike]): Path to the log file.

        Examples:
            >>> maint = DuckLakeMaintenance(conn)
            >>> maint = DuckLakeMaintenance(conn, catalog_alias='my_lake')
            >>> maint = DuckLakeMaintenance(conn, schema='predictions')
        """
        # Stockage de la connexion, de l'alias du catalogue et du schéma associé
        self.conn = connection
        self.catalog_alias = catalog_alias
        self.schema = schema

        # Initialisation du logger nommé.
        # Chemin par défaut centralisé dans utils.logger : <cwd>/logs/<name>.log.
        self.logger = _init_logger(filename=log_filename, name="ducklake_maintenance")

    # ---------------------------------------------------------------------------
    # Méthodes de maintenance individuelles
    # ---------------------------------------------------------------------------

    # Fusion des petits fichiers Parquet adjacents
    def merge_files(
        self,
        schema: str,
        table: str,
        min_file_size: int = 100_000_000,
        max_file_size: int = 500_000_000,
        max_compacted_files: int = 100,
    ) -> tuple[str, str, int, int]:
        """
        Merge small adjacent Parquet files into larger files.

        Each INSERT/UPDATE/MERGE in DuckLake produces a small Parquet file.
        Over time, a table may consist of hundreds of tiny files, which forces
        DuckDB to open many file handles during a sequential scan. This procedure
        merges files smaller than ``min_file_size`` into larger chunks, reducing
        scan overhead.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).
            table (str): Table name to compact (e.g. ``'fact_table'``).
            min_file_size (int): Files smaller than this (in **bytes** — DuckLake
                rejects a value with a unit, e.g. ``'1KB'``) are candidates for
                merging. Defaults to 100 000 000 (100MB), aligned with the
                recommended ``target_file_size`` (:data:`RECOMMENDED_DUCKLAKE_OPTIONS`
                in ``connection.connector``): files below the target get merged.
            max_file_size (int): Cap, in bytes, on the size of a merged output file.
                Defaults to 500 000 000 (500MB, 5x the target) to leave room for
                combining several under-target files without producing oversized
                ones.
            max_compacted_files (int): Maximum number of small files combined into
                one output file. Defaults to 100.

        Returns:
            tuple[str, str, int, int]: ``(schema_name, table_name, files_processed,
            files_created)`` as reported by DuckLake, or ``(schema, table, 0, 0)`` if
            the call fails (logged as a warning, non-fatal).

        Examples:
            >>> maint.merge_files('main', 'fact_table')
            >>> maint.merge_files('main', 'fact_table', min_file_size=50_000_000)
        """
        try:
            # Exécution de la fusion des fichiers.
            # min_file_size / max_file_size / max_compacted_files / schema sont des
            # paramètres nommés uniquement.
            result = self.conn.execute(
                f"SELECT * FROM ducklake_merge_adjacent_files('{self.catalog_alias}',"
                f" '{table}', min_file_size := {min_file_size}, max_file_size :="
                f" {max_file_size}, max_compacted_files := {max_compacted_files},"
                f" schema := '{schema}')"
            ).fetchone()
            schema_name, table_name, files_processed, files_created = (
                result if result is not None else (schema, table, 0, 0)
            )
            # Logging : un zéro est toujours explicité
            if files_processed == 0:
                self.logger.info(
                    f"merge_files {schema}.{table} : 0 file merged (no"
                    f" file under the threshold min_file_size={min_file_size})"
                )
            else:
                self.logger.info(
                    f"merge_files {schema}.{table} : {files_processed} file(s)"
                    f" processed, {files_created} files(s) created(s)"
                )
            return schema_name, table_name, files_processed, files_created
        except Exception as e:
            # Logging
            self.logger.warning(f"merge_files failed for {schema}.{table} : {e}")
            return schema, table, 0, 0

    # Réécriture des fichiers contenant des suppressions
    def rewrite_data_files(
        self,
        schema: str,
        table: str,
        delete_threshold: float = 0.1,
    ) -> tuple[str, str, int, int]:
        """
        Rewrite data files to remove deleted rows from Parquet files.

        DuckLake represents DELETE and UPDATE operations as separate delete-tombstone
        files. These tombstones accumulate and must be applied as a filter on every
        read. This procedure rewrites files whose deleted-row share exceeds
        ``delete_threshold``, physically removing deleted rows.

        **Without an explicit ``delete_threshold`` this procedure is a true no-op**
        (measured: an empty result set, even at 25% deletions) — this is why it is
        always passed here rather than left to the engine default.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).
            table (str): Table name to rewrite (e.g. ``'fact_table'``).
            delete_threshold (float): Rewrite files whose deleted-row share exceeds
                this fraction (0-1). Defaults to 0.1 (per the specification's
                recommended 0.1-0.3 range for after-write compaction).

        Returns:
            tuple[str, str, int, int]: ``(schema_name, table_name, files_processed,
            files_created)`` as reported by DuckLake, or ``(schema, table, 0, 0)`` if
            no file crosses the threshold or the call fails (the latter logged as a
            warning, non-fatal).

        Examples:
            >>> maint.rewrite_data_files('main', 'fact_table')
            >>> maint.rewrite_data_files('main', 'fact_table', delete_threshold=0.3)
        """
        try:
            # Exécution de la réécriture des fichiers.
            # delete_threshold et schema sont des paramètres nommés uniquement.
            result = self.conn.execute(
                f"SELECT * FROM ducklake_rewrite_data_files('{self.catalog_alias}',"
                f" '{table}', delete_threshold := {delete_threshold}, schema :="
                f" '{schema}')"
            ).fetchone()
            # Cas où le résultat est vide
            if result is None:
                # No-op réel : aucun fichier ne dépasse le seuil de suppression
                self.logger.info(
                    f"rewrite_data_files {schema}.{table} : 0 rewriten file"
                    f" (deletion threshold {delete_threshold} unreached)"
                )
                return schema, table, 0, 0
            # Extractions des composantes du résultat
            schema_name, table_name, files_processed, files_created = result
            # Logging
            self.logger.info(
                f"rewrite_data_files {schema}.{table} : {files_processed} file(s)"
                f" processed, {files_created} file(s) created"
                f" (delete_threshold={delete_threshold})"
            )
            return schema_name, table_name, files_processed, files_created
        except Exception as e:
            # Loggin
            self.logger.warning(f"rewrite_data_files failed for {schema}.{table} : {e}")
            return schema, table, 0, 0

    # Écriture en Parquet des lignes inlinées dans le catalogue
    def flush_inlined_data(
        self, table: str | None = None, schema: str | None = None
    ) -> list[tuple[str, str, int]]:
        """
        Write inlined catalog rows out to Parquet files.

        Data inlining is active by default: a small
        ``INSERT`` produces no Parquet file at all, the rows living in the catalog
        instead. This procedure flushes them out to Parquet — required before
        reading the data path's files directly, and recommended in planned
        maintenance.

        Args:
            table (Optional[str]): Table to flush, in this instance's ``schema``.
                Defaults to None, flushing every table of the whole catalog.
            schema (Optional[str]): Schema of ``table``. Defaults to None (this
                instance's ``schema``).

        Returns:
            list[tuple[str, str, int]]: ``(schema_name, table_name, rows_flushed)``
            rows as reported by DuckLake — empty when there was nothing inlined, or
            if the call fails (logged as a warning, non-fatal).

        Examples:
            >>> maint.flush_inlined_data()
            >>> maint.flush_inlined_data('fact_table')
        """
        try:
            if table is not None:
                # table_name / schema_name sont des paramètres nommés uniquement.
                query = (
                    f"SELECT * FROM ducklake_flush_inlined_data('{self.catalog_alias}',"
                    f" table_name := '{table}', schema_name :="
                    f" '{schema or self.schema}')"
                )
            else:
                # Aucune table : vidage de l'ensemble du catalogue
                query = (
                    f"SELECT * FROM ducklake_flush_inlined_data('{self.catalog_alias}')"
                )
            rows = self.conn.execute(query).fetchall()
            total_rows = sum(r[2] for r in rows)
            # Logging : un zéro est toujours explicité
            if not rows:
                self.logger.info(
                    f"flush_inlined_data ({table or 'for all tables'}) : nothing to"
                    f" flush (no ilined row)"
                )
            else:
                self.logger.info(
                    f"flush_inlined_data ({table or 'for all tables'}) :"
                    f" {int(total_rows)} line(s) flushed to Parquet"
                )
            return rows
        except Exception as e:
            # Logging
            self.logger.warning(f"flush_inlined_data failed for {table} : {e}")
            return []

    # Suppression des fichiers du data_path inconnus du catalogue
    def delete_orphaned_files(
        self,
        older_than: datetime | str | None = None,
        dry_run: bool = True,
    ) -> list[str]:
        """
        Delete files under ``data_path`` that are unknown to the catalog.

        Reserved for after an incident (interrupted transaction, manual file copy) —
        run with ``dry_run=True`` first to review what would be deleted. This is an
        **irreversible** operation, unrelated to snapshot expiration.

        Args:
            older_than (datetime | str | None): Only consider files older than this
                cutoff. Defaults to None (no age filter).
            dry_run (bool): When True (the default), list the files that would be
                deleted without deleting them.

        Returns:
            list[str]: Paths deleted (or that would be deleted, under ``dry_run``).

        Examples:
            >>> maint.delete_orphaned_files()  # dry_run=True by default : safe review
            >>> maint.delete_orphaned_files(dry_run=False)  # actually deletes
        """
        try:
            # older_than n'est ajouté que s'il est fourni : le passer explicitement à
            # NULL provoque une erreur interne DuckDB (mesuré).
            parts = [f"dry_run := {str(dry_run).lower()}"]
            if older_than is not None:
                ts = (
                    older_than
                    if isinstance(older_than, str)
                    else older_than.strftime("%Y-%m-%d %H:%M:%S")
                )
                parts.append(f"older_than := TIMESTAMPTZ '{ts}'")
            # Construction de la requête
            query = (
                f"SELECT * FROM ducklake_delete_orphaned_files('{self.catalog_alias}',"
                f" {', '.join(parts)})"
            )
            rows = self.conn.execute(query).fetchall()
            paths = [r[0] for r in rows]
            # Logging
            mode = "dry_run" if dry_run else "supprimé(s)"
            self.logger.info(f"delete_orphaned_files : {len(paths)} file(s) ({mode})")
            return paths
        except Exception as e:
            # Logging
            self.logger.warning(f"delete_orphaned_files failed : {e}")
            return []

    # Expiration des anciens snapshots du catalogue
    def expire_snapshots(
        self,
        schema: str,
        older_than_days: int = 30,
        dry_run: bool = False,
    ) -> list[tuple[Any, ...]]:
        """
        Expire old snapshots to free catalog space.

        DuckLake retains every committed snapshot indefinitely by default, enabling
        time travel but consuming catalog space. This procedure marks snapshots older
        than ``older_than_days`` as expired. Expired snapshots can no longer be
        queried via ``AT (VERSION => n)`` or ``AT (TIMESTAMP => t)``.

        **Reserved for planned maintenance with an explicit retention** — never
        called from ``DatabaseUpdater``/``DatabaseDeleter`` after a normal write,
        since it destroys time travel beyond the retention.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).
            older_than_days (int): Snapshots older than this many days will be expired.
                Defaults to 30.
            dry_run (bool): When True, list the snapshots that would be expired
                without expiring them. Defaults to False.

        Returns:
            list[tuple]: The expired (or, under ``dry_run``, would-be-expired)
            snapshot rows, or an empty list if the call fails (logged as a warning).

        Examples:
            >>> maint.expire_snapshots('main', older_than_days=30)
            >>> maint.expire_snapshots('main', older_than_days=7, dry_run=True)
        """
        # Calcul du timestamp de coupure à partir du nombre de jours
        cutoff: datetime = datetime.now() - timedelta(days=older_than_days)
        cutoff_str: str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        try:
            # Exécution de la requête
            rows = self.conn.execute(
                f"SELECT * FROM ducklake_expire_snapshots('{self.catalog_alias}', "
                f"older_than := TIMESTAMPTZ '{cutoff_str}',"
                f" dry_run := {str(dry_run).lower()})"
            ).fetchall()
            # Logging
            self.logger.info(
                f"expire_snapshots {schema} (cutoff={cutoff_str}, dry_run={dry_run}) :"
                f" {len(rows)} snapshot(s)"
            )
            return rows
        except Exception as e:
            # Logging
            self.logger.warning(f"expire_snapshots failed for {schema} : {e}")
            return []

    # Nettoyage des fichiers Parquet orphelins
    def cleanup_files(self, schema: str, dry_run: bool = False) -> list[str]:
        """
        Remove files no longer referenced by any live snapshot.

        After expiring snapshots, the Parquet data files they referenced remain on
        disk until this procedure is called — the only step that actually frees disk
        space (measured). **Reserved for planned maintenance**, after
        ``expire_snapshots``.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).
            dry_run (bool): When True, list the files that would be deleted without
                deleting them. Defaults to False.

        Returns:
            list[str]: Paths deleted (or that would be deleted, under ``dry_run``),
            or an empty list if the call fails (logged as a warning).

        Examples:
            >>> maint.cleanup_files('main')
            >>> maint.cleanup_files('main', dry_run=True)
        """
        try:
            # Exécution de la suppression des fichiers orphelins
            rows = self.conn.execute(
                f"SELECT * FROM ducklake_cleanup_old_files('{self.catalog_alias}',"
                f" dry_run := {str(dry_run).lower()})"
            ).fetchall()
            paths = [r[0] for r in rows]
            # Logging
            mode = "dry_run" if dry_run else "supprimé(s)"
            self.logger.info(
                f"cleanup_files {schema} : {len(paths)} fichier(s) ({mode})"
            )
            return paths
        except Exception as e:
            # Logging
            self.logger.warning(f"cleanup_files failed for {schema} : {e}")
            return []

    # ---------------------------------------------------------------------------
    # Méthodes de gestion du partitionnement
    # ---------------------------------------------------------------------------

    # Définition ou remplacement du partitionnement d'une table
    def set_partitioned_by(
        self,
        table: str,
        partition_by: list[str],
        schema: str = "main",
    ) -> None:
        """
        Set or replace the partition keys of an existing DuckLake table.

        Executes ``ALTER TABLE … SET PARTITIONED BY`` with the provided keys.
        Only data written *after* this call will be stored in the new partition
        layout; existing files are unaffected (see ``repartition`` to force a
        physical rewrite).

        Supported partition expressions (passed as plain strings):

        - column name — ``'country'``
        - time transforms — ``'year(ts)'``, ``'month(ts)'``,
                            ``'day(ts)'``, ``'hour(ts)'``
        - hash distribution — ``'bucket(8, user_id)'``

        Args:
            table (str): Table name (e.g. ``'fact_table'``).
            partition_by (list[str]): Non-empty list of partition expressions.
            schema (str): DuckLake schema name. Defaults to ``'main'``.

        Raises:
            ValueError: If ``partition_by`` is empty.

        Examples:
            >>> maint.set_partitioned_by('fact_table', ['country'])
            >>> maint.set_partitioned_by('fact_table', ['year(date_col)', 'country'])
            >>> maint.set_partitioned_by('fact_table', ['bucket(8, user_id)'])
        """
        # Validation : au moins une clé de partitionnement est requise
        if not partition_by:
            raise ValueError(
                "partition_by ne peut pas être vide. "
                "Pour supprimer le partitionnement, utilisez reset_partitioned_by()."
            )

        # Construction et exécution du DDL
        cols = ", ".join(partition_by)
        self.conn.execute(f"ALTER TABLE {schema}.{table} SET PARTITIONED BY ({cols})")

        # Logging
        self.logger.info(f"Partitionning defined on {schema}.{table} : ({cols})")

    # Suppression du partitionnement d'une table
    def reset_partitioned_by(
        self,
        table: str,
        schema: str = "main",
    ) -> None:
        """
        Remove all partition keys from an existing DuckLake table.

        Executes ``ALTER TABLE … RESET PARTITIONED BY``. Subsequent writes
        will produce unpartitioned files; existing files are unaffected.

        Args:
            table (str): Table name (e.g. ``'fact_table'``).
            schema (str): DuckLake schema name. Defaults to ``'main'``.

        Examples:
            >>> maint.reset_partitioned_by('fact_table')
        """
        # Suppression de la définition de partitionnement
        self.conn.execute(f"ALTER TABLE {schema}.{table} RESET PARTITIONED BY")

        # Logging
        self.logger.info(f"Partitionning removed on {schema}.{table}")

    # Réinitialisation et application d'un nouveau partitionnement
    def repartition(
        self,
        table: str,
        partition_by: list[str] | None = None,
        schema: str = "main",
        run_maintenance: bool = True,
    ) -> None:
        """
        Reset the current partitioning and optionally apply a new one.

        Orchestrates three steps in order:

        1. ``reset_partitioned_by`` — clears the existing partition definition.
        2. ``set_partitioned_by`` — applies the new keys (skipped if
           ``partition_by`` is ``None``).
        3. ``merge_files`` + ``rewrite_data_files`` — rewrites existing Parquet
           files so they adopt the new layout (only when ``run_maintenance=True``).

        Note: DuckLake only partitions *newly written* data by default.
        Pass ``run_maintenance=True`` (the default) to also rewrite the existing
        files into the new partition structure.

        Args:
            table (str): Table name (e.g. ``'fact_table'``).
            partition_by (Optional[list[str]]): New partition keys. Pass ``None``
                to remove partitioning without defining a replacement.
            schema (str): DuckLake schema name. Defaults to ``'main'``.
            run_maintenance (bool): Whether to trigger ``merge_files`` and
                ``rewrite_data_files`` after changing the partition definition.
                Defaults to ``True``.

        Examples:
            >>> # Chang partition keys
            >>> maint.repartition(
            >>>     'fact_table',
            >>>     partition_by=['year(date_col)', 'country']
            >>> )
            >>> # Remove partitioning
            >>> maint.repartition('fact_table', partition_by=None)
            >>> # Change without rewriting the existing files
            >>> maint.repartition('fact_table', ['country'], run_maintenance=False)
        """
        # Logging de début d'opération
        self.logger.info(
            f"Begin the repartition of {schema}.{table} "
            f"— new partition keys : {partition_by}"
        )

        # Étape 1 : suppression du partitionnement courant
        self.reset_partitioned_by(table, schema=schema)

        # Étape 2 : application du nouveau partitionnement (si fourni)
        if partition_by is not None:
            self.set_partitioned_by(table, partition_by, schema=schema)

        # Étape 3 : réécriture physique des fichiers existants dans la nouvelle
        # structure, afin que les données déjà présentes bénéficient également du
        # nouveau layout de partitionnement.
        if run_maintenance:
            self.merge_files(schema, table)
            self.rewrite_data_files(schema, table)

        # Logging de fin d'opération
        self.logger.info(f"The reparttition of {schema}.{table} is completed")

    # ---------------------------------------------------------------------------
    # Méthode de compaction post-écriture
    # ---------------------------------------------------------------------------

    # Compaction légère après une écriture (merge + rewrite)
    def compact(
        self,
        table: str = "fact_table",
        schema: str | None = None,
        delete_threshold: float = 0.1,
        report: OperationReport | None = None,
    ) -> dict[str, int]:
        """
        Run the lightweight post-write compaction: merge then rewrite.

        Merges small adjacent Parquet files and rewrites files whose deleted-row
        share exceeds ``delete_threshold``. Called by the write operations
        (``update_database``, ``add_columns``, ``delete_rows``) right after their
        commit. Never raises: ``merge_files``/``rewrite_data_files`` already log and
        swallow their own failures.

        Never calls ``expire_snapshots``, ``cleanup_files`` or
        ``delete_orphaned_files``: those destroy time travel or are irreversible and
        belong to planned maintenance (:meth:`maintain` with an explicit
        ``MaintenancePolicy``).

        Args:
            table (str): Bare table name. Defaults to ``'fact_table'``.
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).
            delete_threshold (float): Rewrite files whose deleted-row share exceeds
                this fraction (0-1). Defaults to 0.1 — without an explicit value
                ``ducklake_rewrite_data_files`` is a measured no-op.
            report (OperationReport | None): When given, its ``maintenance`` dict
                receives the four counters (zeros included).

        Returns:
            dict[str, int]: ``merge_files_processed``, ``merge_files_created``,
            ``rewrite_files_processed`` and ``rewrite_files_created``.

        Examples:
            >>> maint.compact()
            >>> maint.compact('fact_table', 'predictions', delete_threshold=0.05)
        """
        # Résolution du schéma cible
        schema = schema or self.schema
        # Fusion des petits fichiers puis réécriture des fichiers trop supprimés
        _, _, merge_processed, merge_created = self.merge_files(schema, table)
        _, _, rewrite_processed, rewrite_created = self.rewrite_data_files(
            schema, table, delete_threshold=delete_threshold
        )
        counters = {
            "merge_files_processed": merge_processed,
            "merge_files_created": merge_created,
            "rewrite_files_processed": rewrite_processed,
            "rewrite_files_created": rewrite_created,
        }
        # Logging
        self.logger.info(
            f"Compaction DuckLake finished for '{schema}.{table}' : merge"
            f" {merge_processed} -> {merge_created} file(s), rewrite"
            f" {rewrite_processed} -> {rewrite_created} file(s)"
            f" (delete_threshold={delete_threshold})"
        )
        # Ajout au rapport
        if report is not None:
            report.maintenance.update(counters)
        return counters

    # Exécution de l'ensemble des opérations de maintenance dans l'ordre recommandé
    # ---------------------------------------------------------------------------
    # Méthodes de diagnostic du stockage et de réordonnancement
    # ---------------------------------------------------------------------------

    # Diagnostic de l'état physique du stockage d'une table
    def storage_report(
        self,
        table: str = "fact_table",
        schema: str | None = None,
        min_file_size_bytes: int = 100_000_000,
    ) -> StorageReport:
        """
        Measure the physical storage state of a table.

        Sources: ``ducklake_table_info`` (files, bytes, delete files); the internal
        ``__ducklake_metadata_<alias>`` tables for small files, inlined rows and
        per-file column statistics (``ducklake_data_file`` joined to
        ``ducklake_file_column_stats``, active files only, bounds cast to the
        column's type); ``ducklake_snapshots`` for the snapshot count and age.
        Every measurement is non-fatal (warning logged, default kept).

        Args:
            table (str): Bare table name. Defaults to ``'fact_table'``.
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).
            min_file_size_bytes (int): Size (bytes) under which a data file counts
                as small. Defaults to 100 000 000.

        Returns:
            StorageReport: The measured indicators (``overlap_ratio`` is ``None``
            when ``dataset_metadata.cluster_by`` is absent).

        Examples:
            >>> report = maint.storage_report()
            >>> report.overlap_ratio
            0.75
            >>> print(report.summary())
        """
        # Résolution du schéma et initialisation du rapport
        schema = schema or self.schema
        report = StorageReport(
            schema=schema, table=table, min_file_size_bytes=min_file_size_bytes
        )

        # Fichiers de données et de suppression
        info = _table_info(self.conn, self.catalog_alias, schema, table, self.logger)
        if info is not None:
            (
                report.file_count,
                report.total_bytes,
                report.delete_file_count,
                report.delete_bytes,
            ) = info
            report.delete_ratio = (
                report.delete_bytes / report.total_bytes if report.total_bytes else 0.0
            )

        # Petits fichiers (fichiers vides résiduels compris) et fichiers de suppression
        # réellement vivants
        try:
            # Identifiant de la table
            table_id = self._table_id(schema, table)
            if table_id is not None:
                # ducklake_table_info compte aussi les fichiers de suppression dont le
                # fichier de données a été retiré par un DELETE intégral (mesuré après
                # recluster : end_snapshot reste NULL, rewrite_data_files n'y peut
                # rien) ; seuls ceux visant un fichier de données actif comptent
                row = self.conn.execute(
                    f"""
                    SELECT count(*), coalesce(sum(d.file_size_bytes), 0)
                    FROM {self._metadata_catalog()}.ducklake_delete_file d
                    JOIN {self._metadata_catalog()}.ducklake_data_file f
                        ON f.data_file_id = d.data_file_id
                    WHERE d.table_id = ? AND d.end_snapshot IS NULL
                      AND f.end_snapshot IS NULL
                    """,
                    [table_id],
                ).fetchone()
                # Population du rapport
                if row is not None:
                    report.delete_file_count = int(row[0])
                    report.delete_bytes = int(row[1])
                    report.delete_ratio = (
                        report.delete_bytes / report.total_bytes
                        if report.total_bytes
                        else 0.0
                    )
                meta = self._metadata_catalog()
                # Comptage des petits fichiers
                row = self.conn.execute(
                    f"SELECT count(*) FROM {meta}.ducklake_data_file"
                    " WHERE table_id = ? AND end_snapshot IS NULL"
                    " AND file_size_bytes < ?",
                    [table_id, min_file_size_bytes],
                ).fetchone()
                report.small_file_count = int(row[0]) if row is not None else 0
        except Exception as e:
            # Logging
            self.logger.warning(f"storage_report {schema}.{table}: small files: {e}")

        # Lignes inlinées non vidangées
        report.inlined_rows, report.has_inlined_data = self._inlined_state(
            schema, table
        )

        # Snapshots : nombre et âge du plus ancien. L'âge est calculé en SQL (epoch)
        # car la récupération d'un TIMESTAMPTZ en Python exige pytz (mesuré).
        try:
            row = self.conn.execute(
                "SELECT count(*), epoch(now()) - epoch(min(snapshot_time))"
                f" FROM ducklake_snapshots('{self.catalog_alias}')"
            ).fetchone()
            if row is not None:
                report.snapshot_count = int(row[0])
                if row[1] is not None:
                    report.oldest_snapshot_age_days = float(row[1]) / 86_400
        except Exception as e:
            self.logger.warning(f"storage_report {schema}.{table}: snapshots: {e}")

        # Recouvrement des plages de fichiers sur la première colonne de cluster_by
        cluster_by = self._cluster_by(schema)
        if cluster_by:
            report.cluster_column = cluster_by[0]
            report.overlap_ratio = self._safe_overlap_ratio(
                schema, table, cluster_by[0]
            )

        # Logging
        self.logger.info(report.summary())
        return report

    # Réordonnancement physique complet d'une table
    def recluster(
        self,
        table: str = "fact_table",
        order_by: list[str] | None = None,
        schema: str | None = None,
        merge_min_file_size: int | None = None,
        merge_max_file_size: int | None = None,
    ) -> OperationReport:
        """
        Physically rewrite a table in sort order to restore file pruning.

        DuckLake prunes files with per-file min/max statistics, which only works
        when the data is physically grouped. Each update writes new files sorted
        within the batch but not merged into the global order, so file ranges
        progressively overlap. This method rewrites the whole table sorted by
        ``order_by`` in a single transaction: ``CREATE TEMP TABLE _recluster AS
        SELECT * FROM <table>``, ``DELETE FROM <table>`` (a full delete creates no
        delete file, measured), ``INSERT INTO <table> SELECT * FROM _recluster ORDER
        BY …``, ``DROP TABLE _recluster``, ``COMMIT``. The table keeps its identity
        (table id, partitioning, history/time travel).

        After commit, ``merge_adjacent_files`` absorbs small or empty residual
        files (adjacent files keep the order). ``expire_snapshots``/
        ``cleanup_files`` are never run here: they belong to planned maintenance.

        **Cost**: a full rewrite of the table, single-threaded; storage doubles
        until the previous files are released by ``expire_snapshots`` +
        ``cleanup_files``. **When**: after N updates, when
        ``storage_report().overlap_ratio`` exceeds a threshold (see
        ``MaintenancePolicy.max_overlap_ratio`` and :meth:`maintain`).

        Args:
            table (str): Bare table name. Defaults to ``'fact_table'``.
            order_by (list[str] | None): Sort columns. Defaults to None
                (``dataset_metadata.cluster_by`` of the schema).
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).
            merge_min_file_size (int | None): ``min_file_size`` (bytes) of the
                post-commit merge. Defaults to None (a quarter of the catalog's
                ``target_file_size``).
            merge_max_file_size (int | None): ``max_file_size`` (bytes) of the
                post-commit merge. Defaults to None (the ``target_file_size``).

        Returns:
            OperationReport: ``operation='recluster'``, rows/files/bytes/snapshot
            before and after, ``maintenance`` with the merge counters and
            ``overlap_ratio_before``/``overlap_ratio_after``.

        Raises:
            ValueError: If the table does not exist, no sort key is available
                (no ``order_by`` and no ``cluster_by``), or a sort column is unknown.
            Exception: Any failure of the rewrite, re-raised after ``ROLLBACK``
                (the table is left untouched).

        Examples:
            >>> report = maint.recluster()  # ORDER BY dataset_metadata.cluster_by
            >>> report.maintenance['overlap_ratio_after']
            0.0
            >>> maint.recluster('fact_table', order_by=['date', 'region'])
        """
        # Résolution du schéma et validation de la table
        schema = schema or self.schema
        columns = self._table_columns(schema, table)
        if not columns:
            raise ValueError(f"Table {schema}.{table} does not exist")

        # Résolution et validation de la clé de tri
        sort_columns = (
            list(order_by) if order_by is not None else (self._cluster_by(schema) or [])
        )
        if not sort_columns:
            raise ValueError(
                f"No sort key for {schema}.{table}: pass order_by or set"
                " dataset_metadata.cluster_by"
            )
        # Colonnes absentes de la liste des colonnes de la table
        unknown = [c for c in sort_columns if c not in columns]
        if unknown:
            raise ValueError(
                f"order_by columns {unknown} do not exist in {schema}.{table}"
            )
        qualified = qualify_table(table, schema, self.catalog_alias)
        order_clause = ", ".join(quote_ident(c) for c in sort_columns)

        # État avant réordonnancement
        start_time = time.time()
        info_before = _table_info(
            self.conn, self.catalog_alias, schema, table, self.logger
        )
        files_before, bytes_before, _, _ = info_before or (0, 0, 0, 0)
        report = OperationReport(
            operation="recluster",
            schema=schema,
            run_id=None,
            started_at=datetime.now(),
            duration_seconds=0.0,
            rows_before=self._count_rows(schema, table),
            files_before=files_before,
            bytes_before=bytes_before,
            snapshot_before=_current_snapshot_id(
                self.conn, self.catalog_alias, self.logger
            ),
        )
        overlap_before = self._safe_overlap_ratio(schema, table, sort_columns[0])
        if overlap_before is not None:
            report.maintenance["overlap_ratio_before"] = overlap_before
        # Logging
        self.logger.info(
            f"recluster {schema}.{table}: ORDER BY {sort_columns}"
            f" (overlap before: {overlap_before})"
        )

        # Valeur courante de threads, restaurée en finally (SET non transactionnel)
        row = self.conn.execute("SELECT current_setting('threads')").fetchone()
        previous_threads = int(row[0]) if row is not None else None

        # Nettoyage d'une éventuelle table temporaire laissée par un échec antérieur
        self.conn.execute("DROP TABLE IF EXISTS temp.main._recluster")
        try:
            self.conn.execute("BEGIN")
            try:
                # Matérialisation, vidage puis réinsertion triée
                self.conn.execute(
                    f"CREATE TEMP TABLE _recluster AS SELECT * FROM {qualified}"
                )
                self.conn.execute(f"DELETE FROM {qualified}")
                # Un seul thread : fichiers disjoints et monotones (mesuré)
                self.conn.execute("SET threads = 1")
                self.conn.execute(
                    f"INSERT INTO {qualified} SELECT * FROM temp.main._recluster"
                    f" ORDER BY {order_clause}"
                )
                self.conn.execute("DROP TABLE temp.main._recluster")
                _set_commit_message(
                    self.conn,
                    self.catalog_alias,
                    None,
                    f"recluster {schema}.{table}",
                    {
                        "operation": "recluster",
                        "schema": schema,
                        "table": table,
                        "order_by": sort_columns,
                    },
                    self.logger,
                )
                self.conn.execute("COMMIT")
            except Exception as e:
                # Annulation : la table reste intacte
                try:
                    self.conn.execute("ROLLBACK")
                except Exception as rollback_error:
                    self.logger.debug(f"recluster rollback failed: {rollback_error}")
                report.warnings.append(f"recluster failed: {e}")
                report.duration_seconds = time.time() - start_time
                # Logging
                self.logger.error(
                    f"recluster {schema}.{table} failed and was rolled back: {e}"
                )
                raise
        finally:
            # Restauration du nombre de threads, succès comme échec
            if previous_threads is not None:
                self.conn.execute(f"SET threads = {previous_threads}")

        # Fusion des petits fichiers ou fichiers vides résiduels (jamais expire/cleanup)
        target_size = self._target_file_size()
        _, _, merge_processed, merge_created = self.merge_files(
            schema,
            table,
            min_file_size=(
                merge_min_file_size
                if merge_min_file_size is not None
                else target_size // 4
            ),
            max_file_size=(
                merge_max_file_size if merge_max_file_size is not None else target_size
            ),
        )
        report.maintenance["merge_files_processed"] = merge_processed
        report.maintenance["merge_files_created"] = merge_created

        # État après réordonnancement
        report.rows_after = self._count_rows(schema, table)
        info_after = _table_info(
            self.conn, self.catalog_alias, schema, table, self.logger
        )
        report.files_after, report.bytes_after, _, _ = info_after or (0, 0, 0, 0)
        report.snapshot_after = _current_snapshot_id(
            self.conn, self.catalog_alias, self.logger
        )
        overlap_after = self._safe_overlap_ratio(schema, table, sort_columns[0])
        if overlap_after is not None:
            report.maintenance["overlap_ratio_after"] = overlap_after
        if report.rows_after != report.rows_before:
            message = (
                f"recluster row count changed: {report.rows_before} ->"
                f" {report.rows_after}"
            )
            self.logger.warning(message)
            report.warnings.append(message)
        report.duration_seconds = time.time() - start_time

        # Logging
        self.logger.info(
            f"recluster {schema}.{table}: overlap {overlap_before} -> {overlap_after}"
        )
        self.logger.info(report.summary())
        return report

    # Maintenance conditionnelle pilotée par une politique
    def maintain(
        self,
        policy: MaintenancePolicy | None = None,
        table: str = "fact_table",
        schema: str | None = None,
    ) -> OperationReport:
        """
        Run only the maintenance steps justified by the storage indicators.

        Reads :meth:`storage_report` then considers, in order, flush → rewrite →
        merge → recluster → expire → cleanup → delete_orphaned. Each step runs only
        when its indicator justifies it; otherwise it is skipped and the reason is
        logged explicitly (e.g. ``"recluster skipped — overlap 0.12 <=
        max_overlap_ratio 0.5"``) and flagged as ``<step>_skipped`` in
        ``report.maintenance``. The storage report is re-read after each step that
        modified the table.

        ============== ==========================================================
        Step           Runs when
        ============== ==========================================================
        flush          ``flush_inlined`` and unflushed inlined rows exist
        rewrite        at least one delete file exists (DuckLake then applies
                       ``delete_threshold`` per file)
        merge          ``small_file_count > max_small_files``
        recluster      ``recluster`` and ``overlap_ratio > max_overlap_ratio``
        expire         ``retention_days is not None``
        cleanup        ``retention_days is not None``
        delete_orphaned ``delete_orphaned``
        ============== ==========================================================

        Under ``dry_run`` the modifying steps (flush, rewrite, merge, recluster)
        are only logged as "would run"; expire/cleanup/delete_orphaned are called
        with ``dry_run=True``, which only lists. Every step is non-fatal (failure
        logged and added to ``report.warnings``).

        Args:
            policy (MaintenancePolicy | None): Thresholds. Defaults to None
                (``MaintenancePolicy()``: nothing destructive).
            table (str): Bare table name. Defaults to ``'fact_table'``.
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).

        Returns:
            OperationReport: ``operation='maintenance'``; ``maintenance`` holds the
            counters of the steps that ran, ``<step>_skipped``/``<step>_planned``
            flags, and ``recluster_*`` counters when reclustering ran.

        Examples:
            >>> maint.maintain()  # safe defaults
            >>> maint.maintain(MaintenancePolicy(recluster=True, retention_days=30))
            >>> maint.maintain(MaintenancePolicy(dry_run=True, recluster=True))
        """
        # Résolution de la politique et du schéma
        policy = policy if policy is not None else MaintenancePolicy()
        schema = schema or self.schema
        prefix = f"maintain {schema}.{table}{' [dry_run]' if policy.dry_run else ''}"
        self.logger.info(f"{prefix}: start with {policy}")

        # État initial
        start_time = time.time()
        started_at = datetime.now()
        storage = self.storage_report(table, schema, policy.min_file_size_bytes)
        report = OperationReport(
            operation="maintenance",
            schema=schema,
            run_id=None,
            started_at=started_at,
            duration_seconds=0.0,
            rows_before=self._count_rows(schema, table),
            files_before=storage.file_count,
            bytes_before=storage.total_bytes,
            snapshot_before=_current_snapshot_id(
                self.conn, self.catalog_alias, self.logger
            ),
        )

        # Journalisation d'une étape sautée
        def skip(step: str, reason: str) -> None:
            self.logger.info(f"{prefix}: {step} skipped — {reason}")
            report.maintenance[f"{step}_skipped"] = 1

        # Exécution non fatale d'une étape ; retourne True si la table a été modifiée
        def run(
            step: str, reason: str, action: Callable[[], None], modifies: bool = True
        ) -> bool:
            if policy.dry_run and modifies:
                self.logger.info(f"{prefix}: {step} would run — {reason}")
                report.maintenance[f"{step}_planned"] = 1
                return False
            self.logger.info(f"{prefix}: {step} runs — {reason}")
            try:
                action()
                return modifies
            except Exception as e:
                self.logger.warning(f"{prefix}: {step} failed: {e}")
                report.warnings.append(f"{step} failed: {e}")
                return False

        # Étape 1 : vidange des lignes inlinées
        def flush() -> None:
            flushed = self.flush_inlined_data(table, schema=schema)
            report.maintenance["flush_inlined_rows"] = int(sum(r[2] for r in flushed))

        if not policy.flush_inlined:
            skip("flush_inlined_data", "flush_inlined=False")
        elif not storage.has_inlined_data:
            skip("flush_inlined_data", "no inlined rows")
        elif run(
            "flush_inlined_data",
            f"{storage.inlined_rows if storage.inlined_rows is not None else 'some'}"
            " inlined rows",
            flush,
        ):
            storage = self.storage_report(table, schema, policy.min_file_size_bytes)

        # Étape 2 : réécriture des fichiers portant des suppressions
        def rewrite() -> None:
            _, _, processed, created = self.rewrite_data_files(
                schema, table, delete_threshold=policy.delete_threshold
            )
            report.maintenance["rewrite_files_processed"] = processed
            report.maintenance["rewrite_files_created"] = created

        if storage.delete_file_count == 0:
            skip("rewrite_data_files", "no delete file")
        elif run(
            "rewrite_data_files",
            f"{storage.delete_file_count} delete file(s)"
            f" (delete_threshold={policy.delete_threshold})",
            rewrite,
        ):
            storage = self.storage_report(table, schema, policy.min_file_size_bytes)

        # Étape 3 : fusion des petits fichiers
        def merge() -> None:
            _, _, processed, created = self.merge_files(
                schema,
                table,
                min_file_size=policy.min_file_size_bytes,
                max_file_size=max(500_000_000, policy.min_file_size_bytes),
            )
            report.maintenance["merge_files_processed"] = processed
            report.maintenance["merge_files_created"] = created

        if storage.small_file_count <= policy.max_small_files:
            skip(
                "merge_files",
                f"{storage.small_file_count} small file(s) <= max_small_files"
                f" {policy.max_small_files}",
            )
        elif run(
            "merge_files",
            f"{storage.small_file_count} small file(s) > max_small_files"
            f" {policy.max_small_files}",
            merge,
        ):
            storage = self.storage_report(table, schema, policy.min_file_size_bytes)

        # Étape 4 : réordonnancement (opt-in)
        def recluster() -> None:
            sub_report = self.recluster(table, schema=schema)
            for key, value in sub_report.maintenance.items():
                report.maintenance[f"recluster_{key}"] = value
            report.warnings.extend(sub_report.warnings)

        if not policy.recluster:
            skip(
                "recluster",
                f"recluster=False (opt-in), overlap {storage.overlap_ratio}",
            )
        elif storage.overlap_ratio is None:
            skip("recluster", "overlap unknown (no cluster_by)")
        elif storage.overlap_ratio <= policy.max_overlap_ratio:
            skip(
                "recluster",
                f"overlap {storage.overlap_ratio:.2f} <= max_overlap_ratio"
                f" {policy.max_overlap_ratio}",
            )
        else:
            run(
                "recluster",
                f"overlap {storage.overlap_ratio:.2f} > max_overlap_ratio"
                f" {policy.max_overlap_ratio}",
                recluster,
            )

        # Étapes 5 et 6 : expiration puis nettoyage (rétention explicite uniquement)
        retention_days = policy.retention_days
        if retention_days is None:
            skip("expire_snapshots", "retention_days=None (snapshots never expire)")
            skip("cleanup_files", "retention_days=None (nothing expired to clean up)")
        else:

            def expire() -> None:
                expired = self.expire_snapshots(
                    schema, older_than_days=retention_days, dry_run=policy.dry_run
                )
                report.maintenance["expired_snapshots"] = len(expired)

            def cleanup() -> None:
                cleaned = self.cleanup_files(schema, dry_run=policy.dry_run)
                report.maintenance["cleaned_files"] = len(cleaned)

            # dry_run transmis aux procédures : simple listage, aucune modification
            run(
                "expire_snapshots",
                f"retention_days={retention_days}",
                expire,
                modifies=False,
            )
            run("cleanup_files", "after expire_snapshots", cleanup, modifies=False)

        # Étape 7 : suppression des fichiers orphelins (opt-in)
        def delete_orphaned() -> None:
            orphaned = self.delete_orphaned_files(dry_run=policy.dry_run)
            report.maintenance["orphaned_files"] = len(orphaned)

        if not policy.delete_orphaned:
            skip("delete_orphaned_files", "delete_orphaned=False")
        else:
            run(
                "delete_orphaned_files",
                "delete_orphaned=True",
                delete_orphaned,
                modifies=False,
            )

        # État final
        report.rows_after = self._count_rows(schema, table)
        info_after = _table_info(
            self.conn, self.catalog_alias, schema, table, self.logger
        )
        report.files_after, report.bytes_after, _, _ = info_after or (0, 0, 0, 0)
        report.snapshot_after = _current_snapshot_id(
            self.conn, self.catalog_alias, self.logger
        )
        report.duration_seconds = time.time() - start_time

        # Logging
        skipped = [
            key.removesuffix("_skipped")
            for key in report.maintenance
            if key.endswith("_skipped")
        ]
        self.logger.info(f"{prefix}: done, skipped steps: {skipped}")
        self.logger.info(report.summary())
        return report

    # ---------------------------------------------------------------------------
    # Méthodes auxiliaires de lecture du catalogue DuckLake
    # ---------------------------------------------------------------------------

    # Identifiant quoté du catalogue de métadonnées interne
    def _metadata_catalog(self) -> str:
        """Return the quoted ``__ducklake_metadata_<alias>`` catalog identifier.

        Returns:
            str: e.g. ``'"__ducklake_metadata_db"'``.
        """
        return quote_ident(f"__ducklake_metadata_{self.catalog_alias}")

    # Lecture de la clé de tri physique du schéma
    def _cluster_by(self, schema: str) -> list[str] | None:
        """Read ``dataset_metadata.cluster_by`` of ``schema``.

        Same logic as ``BaseSchemaManager._get_cluster_by_columns``, read directly
        since this class does not inherit from it.

        Args:
            schema: DuckLake schema name.

        Returns:
            list[str] | None: The sort columns, or None when ``dataset_metadata``
            is absent or ``cluster_by`` is NULL/empty.
        """
        try:
            row = self.conn.execute(
                "SELECT cluster_by FROM"
                f" {qualify_table('dataset_metadata', schema, self.catalog_alias)}"
            ).fetchone()
        except Exception:
            return None
        if row is None or row[0] is None:
            return None
        decoded: list[str] = json.loads(row[0])
        return decoded or None

    # Colonnes et types d'une table
    def _table_columns(self, schema: str, table: str) -> dict[str, str]:
        """Map each column of a table to its DuckDB type.

        Args:
            schema: DuckLake schema name.
            table: Bare table name.

        Returns:
            dict[str, str]: ``column_name -> data_type`` in column order, empty when
            the table does not exist.
        """
        rows = self.conn.execute(
            "SELECT column_name, data_type FROM duckdb_columns()"
            " WHERE database_name = ? AND schema_name = ? AND table_name = ?"
            " ORDER BY column_index",
            [self.catalog_alias, schema, table],
        ).fetchall()
        return {name: data_type for name, data_type in rows}

    # Identifiant DuckLake d'une table active
    def _table_id(self, schema: str, table: str) -> int | None:
        """Resolve the DuckLake ``table_id`` of an active table.

        Args:
            schema: DuckLake schema name.
            table: Bare table name.

        Returns:
            int | None: The ``table_id``, or None when not found.
        """
        meta = self._metadata_catalog()
        row = self.conn.execute(
            f"""
            SELECT t.table_id
            FROM {meta}.ducklake_table t
            JOIN {meta}.ducklake_schema s ON s.schema_id = t.schema_id
            WHERE t.table_name = ? AND s.schema_name = ?
              AND t.end_snapshot IS NULL AND s.end_snapshot IS NULL
            """,
            [table, schema],
        ).fetchone()
        return int(row[0]) if row is not None else None

    # Identifiant DuckLake d'une colonne active
    def _column_id(self, table_id: int, column: str) -> int | None:
        """Resolve the DuckLake ``column_id`` of an active column.

        Args:
            table_id: DuckLake ``table_id``.
            column: Column name.

        Returns:
            int | None: The ``column_id``, or None when not found.
        """
        row = self.conn.execute(
            f"SELECT column_id FROM {self._metadata_catalog()}.ducklake_column"
            " WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
            [table_id, column],
        ).fetchone()
        return int(row[0]) if row is not None else None

    # Requête SQL des plages [min, max] typées des fichiers actifs
    def _file_ranges_query(self, schema: str, table: str, column: str) -> str | None:
        """Build the query listing active files' typed ``[lo, hi]`` on ``column``.

        Args:
            schema: DuckLake schema name.
            table: Bare table name.
            column: Column whose statistics are read.

        Returns:
            str | None: A query returning ``(data_file_id, record_count, lo, hi)``,
            or None when the table or column cannot be resolved.

        Raises:
            ValueError: If ``column`` does not exist in the table.
        """
        columns = self._table_columns(schema, table)
        if column not in columns:
            raise ValueError(f"Column {column!r} does not exist in {schema}.{table}")
        table_id = self._table_id(schema, table)
        if table_id is None:
            return None
        column_id = self._column_id(table_id, column)
        if column_id is None:
            return None
        # Statistiques stockées en VARCHAR : conversion au type de la colonne pour
        # comparer dans l'ordre du type (et non lexicographique)
        data_type = columns[column]
        meta = self._metadata_catalog()
        return f"""
            SELECT f.data_file_id, f.record_count,
                   TRY_CAST(s.min_value AS {data_type}) AS lo,
                   TRY_CAST(s.max_value AS {data_type}) AS hi
            FROM {meta}.ducklake_data_file f
            JOIN {meta}.ducklake_file_column_stats s
                ON s.data_file_id = f.data_file_id
            WHERE f.table_id = {table_id} AND s.column_id = {column_id}
              AND f.end_snapshot IS NULL
        """

    # Plages [min, max] des fichiers actifs (diagnostic, tests)
    def _file_ranges(
        self, schema: str, table: str, column: str
    ) -> list[tuple[int, int, Any, Any]]:
        """List the active files' ``(data_file_id, record_count, lo, hi)`` ranges.

        Args:
            schema: DuckLake schema name.
            table: Bare table name.
            column: Column whose statistics are read.

        Returns:
            list[tuple[int, int, Any, Any]]: One row per active file, ordered by
            ``data_file_id``.
        """
        query = self._file_ranges_query(schema, table, column)
        if query is None:
            return []
        rows = self.conn.execute(f"{query} ORDER BY f.data_file_id").fetchall()
        return [tuple(r) for r in rows]

    # Part des fichiers dont la plage chevauche celle d'un autre fichier
    def _overlap_ratio(self, schema: str, table: str, column: str) -> float | None:
        """Compute the share of active files whose range overlaps another's.

        Empty files and files without statistics are excluded. Overlap uses strict
        inequalities (``a.lo < b.hi AND b.lo < a.hi``) so files sharing a single
        boundary value — the normal outcome of a sorted rewrite with duplicate keys
        — do not count; identical ranges always do.
        Computed in SQL, so no typed value is fetched into Python.

        Args:
            schema: DuckLake schema name.
            table: Bare table name.
            column: Column whose statistics are compared.

        Returns:
            float | None: Ratio in ``[0, 1]`` (``0.0`` with at most one file), or
            None when the table/column cannot be resolved.
        """
        query = self._file_ranges_query(schema, table, column)
        if query is None:
            return None
        row = self.conn.execute(
            f"""
            WITH r AS (
                SELECT * FROM ({query})
                WHERE record_count > 0 AND lo IS NOT NULL AND hi IS NOT NULL
            )
            SELECT
                (SELECT count(*) FROM r),
                (SELECT count(DISTINCT a.data_file_id) FROM r a JOIN r b
                    ON a.data_file_id <> b.data_file_id
                    AND ((b.lo < a.hi AND a.lo < b.hi)
                         OR (a.lo = b.lo AND a.hi = b.hi)))
            """
        ).fetchone()
        if row is None or row[0] <= 1:
            return 0.0
        return float(row[1]) / float(row[0])

    # Calcul non fatal du recouvrement
    def _safe_overlap_ratio(self, schema: str, table: str, column: str) -> float | None:
        """Non-fatal wrapper of :meth:`_overlap_ratio` (warning logged on failure).

        Args:
            schema: DuckLake schema name.
            table: Bare table name.
            column: Column whose statistics are compared.

        Returns:
            float | None: The ratio, or None on failure.
        """
        try:
            return self._overlap_ratio(schema, table, column)
        except Exception as e:
            self.logger.warning(f"overlap ratio of {schema}.{table}.{column}: {e}")
            return None

    # État des lignes inlinées d'une table
    def _inlined_state(self, schema: str, table: str) -> tuple[int | None, bool]:
        """Measure the unflushed inlined rows of a table.

        Counts the live rows (``end_snapshot IS NULL``) of every inlined-data table
        registered for the table in ``ducklake_inlined_data_tables`` (measured:
        emptied by ``flush_inlined_data``). Falls back to a catalog-wide boolean
        read from ``ducklake_snapshots.changes`` (an ``inlined_insert`` more recent
        than the last ``flushed_inlined``).

        Args:
            schema: DuckLake schema name.
            table: Bare table name.

        Returns:
            tuple[int | None, bool]: ``(inlined_rows, has_inlined_data)``;
            ``inlined_rows`` is None when only the boolean fallback was available.
        """
        meta = self._metadata_catalog()
        try:
            table_id = self._table_id(schema, table)
            if table_id is None:
                return 0, False
            names = self.conn.execute(
                f"SELECT table_name FROM {meta}.ducklake_inlined_data_tables"
                " WHERE table_id = ?",
                [table_id],
            ).fetchall()
            total = 0
            for (name,) in names:
                row = self.conn.execute(
                    f"SELECT count(*) FROM {meta}.{quote_ident(name)}"
                    " WHERE end_snapshot IS NULL"
                ).fetchone()
                total += int(row[0]) if row is not None else 0
            return total, total > 0
        except Exception as e:
            self.logger.debug(f"inlined rows of {schema}.{table} unreadable: {e}")
        # Repli : indicateur booléen depuis l'historique des snapshots
        try:
            row = self.conn.execute(
                "SELECT max(snapshot_id) FILTER (WHERE changes::VARCHAR LIKE"
                " '%inlined_insert%'), max(snapshot_id) FILTER (WHERE"
                " changes::VARCHAR LIKE '%flushed_inlined%')"
                f" FROM ducklake_snapshots('{self.catalog_alias}')"
            ).fetchone()
        except Exception as e:
            self.logger.warning(f"inlined state of {schema}.{table} unreadable: {e}")
            return None, False
        if row is None or row[0] is None:
            return None, False
        return None, row[1] is None or row[0] > row[1]

    # Taille cible des fichiers du catalogue
    def _target_file_size(self) -> int:
        """Read the catalog's ``target_file_size`` option, in bytes.

        Returns:
            int: The option value (stored in bytes, measured), or
            ``_FALLBACK_TARGET_FILE_SIZE`` when unreadable.
        """
        try:
            rows = self.conn.execute(
                f"SELECT * FROM ducklake_options('{self.catalog_alias}')"
            ).fetchall()
            for row in rows:
                if row[0] == "target_file_size":
                    return int(row[2])
        except Exception as e:
            self.logger.debug(f"target_file_size unreadable: {e}")
        return _FALLBACK_TARGET_FILE_SIZE

    # Méthode auxiliaire de comptage des lignes d'une table
    def _count_rows(self, schema: str, table: str) -> int:
        """Count the rows of a table, qualified by ``schema``.

        Args:
            schema: DuckLake schema name.
            table: Bare table name.

        Returns:
            int: Row count, or ``0`` if it cannot be read.
        """
        qualified = qualify_table(table, schema, self.catalog_alias)
        try:
            row = self.conn.execute(f"SELECT COUNT(*) FROM {qualified}").fetchone()
            return int(row[0]) if row is not None else 0
        except Exception:
            return 0
