"""
Storage diagnostic and policy-driven maintenance of a DuckLake table.

:class:`DuckLakeMaintenance` extends the thin procedure wrappers of
:mod:`dt_ducklake_manager.maintenance.procedures` with the three methods that
decide *whether* a procedure is worth running:

- :meth:`DuckLakeMaintenance.storage_report` measures the physical state of a table
  (file/delete/small-file counts, inlined rows, snapshots, file-range overlap on
  the first ``cluster_by`` column);
- :meth:`DuckLakeMaintenance.recluster` rewrites a table in its sort order to
  restore file pruning;
- :meth:`DuckLakeMaintenance.maintain` reads the storage report and only runs a
  step when its indicator justifies it under a :class:`MaintenancePolicy`, logging
  every skipped step and why.

DuckLake has no index: pruning relies on per-file min/max statistics
(``ducklake_file_column_stats``, used by the planner — ``Total Files Read`` in
``EXPLAIN ANALYZE``) and on Parquet row-group statistics. Both only help when the
data is physically grouped, which successive updates degrade — hence ``recluster``.
File-level pruning only starts to matter once a table spans several files, i.e.
beyond a few million rows with the recommended ``target_file_size`` of 100MB; below
that, row-group statistics already do the job.
"""

# Importation des modules
# Modules de base
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

# Rapport d'opération
from ..reporting import (
    OperationReport,
    _current_snapshot_id,
    _set_commit_message,
    _table_info,
)
from ..utils.sql import qualify_table, quote_ident, quote_literal

# Procédures DuckLake enveloppées
from .procedures import (
    DEFAULT_DELETE_THRESHOLD,
    DEFAULT_TARGET_FILE_SIZE_BYTES,
    DuckLakeProcedures,
)


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

    # Initialisation des attributs
    schema: str
    table: str
    file_count: int = 0
    total_bytes: int = 0
    delete_file_count: int = 0
    delete_bytes: int = 0
    delete_ratio: float = 0.0
    small_file_count: int = 0
    min_file_size_bytes: int = DEFAULT_TARGET_FILE_SIZE_BYTES
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
        min_file_size_bytes (int): A data file below this size (bytes) counts as
            "small" in the storage report. Only used to decide whether the merge
            step runs: the merge itself combines adjacent files up to the catalog's
            ``target_file_size``. Defaults to ``100_000_000`` (the recommended
            ``target_file_size``).
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

    # Initialisation des attributs
    delete_threshold: float = DEFAULT_DELETE_THRESHOLD
    min_file_size_bytes: int = DEFAULT_TARGET_FILE_SIZE_BYTES
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
class DuckLakeMaintenance(DuckLakeProcedures):
    """
    Maintenance of a DuckLake catalog: procedures, storage diagnostic and policy.

    Inherits every procedure wrapper of :class:`DuckLakeProcedures` (merge, rewrite,
    flush, partitioning, expire, cleanup, orphaned files, post-write ``compact``)
    and adds the storage diagnostic (:meth:`storage_report`), the physical
    reordering (:meth:`recluster`) and the policy-driven maintenance
    (:meth:`maintain`).

    Attributes:
        conn (duckdb.DuckDBPyConnection): DuckDB connection with the DuckLake
            catalog already attached.
        catalog_alias (str): Alias used in the ``ATTACH`` statement.
        schema (str): Default DuckLake schema of the table-level methods.
        logger (logging.Logger): Logger instance.

    Examples:
        >>> from dt_ducklake_manager.connection import DuckLakeConnector
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
        >>> maint = DuckLakeMaintenance(conn)
        >>> maint.maintain(MaintenancePolicy(retention_days=30))
        >>> maint = DuckLakeMaintenance(conn, schema='predictions')
    """

    # ---------------------------------------------------------------------------
    # Diagnostic du stockage
    # ---------------------------------------------------------------------------

    # Diagnostic de l'état physique du stockage d'une table
    def storage_report(
        self,
        table: str = "fact_table",
        *,
        schema: str | None = None,
        min_file_size_bytes: int = DEFAULT_TARGET_FILE_SIZE_BYTES,
    ) -> StorageReport:
        """
        Measure the physical storage state of a table.

        Sources: ``ducklake_table_info`` (files, bytes); the internal
        ``__ducklake_metadata_<alias>`` tables for live delete files, small files,
        inlined rows and per-file column statistics (``ducklake_data_file`` joined
        to ``ducklake_file_column_stats``, active files only, bounds cast to the
        column's type); ``ducklake_snapshots`` for the snapshot count and age.
        The DuckLake ``table_id`` is resolved once and shared by every
        measurement. Every measurement is non-fatal (warning logged, default kept).

        Args:
            table (str): Table to measure. Defaults to ``'fact_table'``.
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

        # Identifiant DuckLake de la table, partagé par les mesures suivantes.
        # Catalogue interne illisible (aucun catalogue DuckLake attaché) et table
        # inconnue d'un catalogue lisible sont distingués : seul le premier cas
        # justifie le repli sur l'historique des snapshots pour les lignes inlinées.
        try:
            table_id = self._table_id(table, schema)
            catalog_readable = True
        except Exception as e:
            self.logger.debug(f"table_id of {schema}.{table} unreadable: {e}")
            table_id = None
            catalog_readable = False

        # Fichiers de suppression réellement vivants et petits fichiers
        if table_id is not None:
            self._measure_files(report, table_id)

        # Lignes inlinées non vidangées (aucune pour une table inconnue)
        if table_id is not None or not catalog_readable:
            report.inlined_rows, report.has_inlined_data = self._inlined_state(
                table_id, table, schema
            )
        else:
            report.inlined_rows, report.has_inlined_data = 0, False

        # Snapshots : nombre et âge du plus ancien. L'âge est calculé en SQL (epoch)
        # car la récupération d'un TIMESTAMPTZ en Python exige pytz (mesuré).
        try:
            row = self.conn.execute(
                "SELECT count(*), epoch(now()) - epoch(min(snapshot_time))"
                f" FROM ducklake_snapshots({quote_literal(self.catalog_alias)})"
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
                table, schema, cluster_by[0], table_id
            )

        # Logging
        self.logger.debug(report.summary())
        return report

    # ---------------------------------------------------------------------------
    # Réordonnancement physique
    # ---------------------------------------------------------------------------

    # Réordonnancement physique complet d'une table
    def recluster(
        self,
        table: str = "fact_table",
        *,
        order_by: list[str] | None = None,
        schema: str | None = None,
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

        After commit, :meth:`merge_files` absorbs the small or empty residual files
        (adjacent files keep the order). ``expire_snapshots``/``cleanup_files`` are
        never run here: they belong to planned maintenance.

        **Cost**: a full rewrite of the table, single-threaded; storage doubles
        until the previous files are released by ``expire_snapshots`` +
        ``cleanup_files``. **When**: after N updates, when
        ``storage_report().overlap_ratio`` exceeds a threshold (see
        ``MaintenancePolicy.max_overlap_ratio`` and :meth:`maintain`), and only on
        a table large enough to span several files (beyond a few million rows with
        the recommended ``target_file_size``).

        Args:
            table (str): Table to rewrite. Defaults to ``'fact_table'``.
            order_by (list[str] | None): Sort columns. Defaults to None
                (``dataset_metadata.cluster_by`` of the schema).
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).
            merge_max_file_size (int | None): Upper bound (bytes) on the files
                produced by the post-commit merge. Defaults to None (the catalog's
                ``target_file_size``).

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
        columns = self._table_columns(table, schema)
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
        # Colonnes absentes de la table
        unknown = [c for c in sort_columns if c not in columns]
        if unknown:
            raise ValueError(
                f"order_by columns {unknown} do not exist in {schema}.{table}"
            )
        qualified = qualify_table(table, schema, self.catalog_alias)
        order_clause = ", ".join(quote_ident(c) for c in sort_columns)

        # État avant réordonnancement
        start_time = time.time()
        table_id = self._safe_table_id(table, schema)
        files_before, bytes_before, _, _ = _table_info(
            self.conn, self.catalog_alias, schema, table, self.logger
        ) or (0, 0, 0, 0)
        report = OperationReport(
            operation="recluster",
            schema=schema,
            run_id=None,
            started_at=datetime.now(),
            duration_seconds=0.0,
            rows_before=self._count_rows(table, schema),
            files_before=files_before,
            bytes_before=bytes_before,
            snapshot_before=_current_snapshot_id(
                self.conn, self.catalog_alias, self.logger
            ),
        )
        overlap_before = self._safe_overlap_ratio(
            table, schema, sort_columns[0], table_id
        )
        if overlap_before is not None:
            report.maintenance["overlap_ratio_before"] = overlap_before
        # Logging
        self.logger.debug(
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
                # Un seul thread : fichiers disjoints et monotones
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
        _, _, merge_processed, merge_created = self.merge_files(
            table, schema=schema, max_file_size=merge_max_file_size
        )
        report.maintenance["merge_files_processed"] = merge_processed
        report.maintenance["merge_files_created"] = merge_created

        # État après réordonnancement
        report.rows_after = self._count_rows(table, schema)
        report.files_after, report.bytes_after, _, _ = _table_info(
            self.conn, self.catalog_alias, schema, table, self.logger
        ) or (0, 0, 0, 0)
        report.snapshot_after = _current_snapshot_id(
            self.conn, self.catalog_alias, self.logger
        )
        overlap_after = self._safe_overlap_ratio(
            table, schema, sort_columns[0], table_id
        )
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
            f"{report.summary()}, overlap {overlap_before} -> {overlap_after}"
        )
        return report

    # ---------------------------------------------------------------------------
    # Maintenance pilotée par une politique
    # ---------------------------------------------------------------------------

    # Maintenance conditionnelle pilotée par une politique
    def maintain(
        self,
        policy: MaintenancePolicy | None = None,
        table: str = "fact_table",
        *,
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
            table (str): Table to maintain. Defaults to ``'fact_table'``.
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
        # Logging
        self.logger.debug(f"{prefix}: start with {policy}")

        # Relecture de l'état du stockage selon la politique
        def measure() -> StorageReport:
            return self.storage_report(
                table, schema=schema, min_file_size_bytes=policy.min_file_size_bytes
            )

        # État initial
        start_time = time.time()
        started_at = datetime.now()
        storage = measure()
        report = OperationReport(
            operation="maintenance",
            schema=schema,
            run_id=None,
            started_at=started_at,
            duration_seconds=0.0,
            rows_before=self._count_rows(table, schema),
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
            storage = measure()

        # Étape 2 : réécriture des fichiers portant des suppressions
        def rewrite() -> None:
            _, _, processed, created = self.rewrite_data_files(
                table, schema=schema, delete_threshold=policy.delete_threshold
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
            storage = measure()

        # Étape 3 : fusion des petits fichiers, jusqu'à la taille cible du catalogue
        def merge() -> None:
            _, _, processed, created = self.merge_files(table, schema=schema)
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
            storage = measure()

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
                    older_than_days=retention_days, dry_run=policy.dry_run
                )
                report.maintenance["expired_snapshots"] = len(expired)

            def cleanup() -> None:
                cleaned = self.cleanup_files(dry_run=policy.dry_run)
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
        report.rows_after = self._count_rows(table, schema)
        report.files_after, report.bytes_after, _, _ = _table_info(
            self.conn, self.catalog_alias, schema, table, self.logger
        ) or (0, 0, 0, 0)
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
        self.logger.info(f"{report.summary()}, skipped steps: {skipped}")
        return report

    # ---------------------------------------------------------------------------
    # Méthodes auxiliaires de lecture du catalogue DuckLake
    # ---------------------------------------------------------------------------

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
    def _table_columns(self, table: str, schema: str) -> dict[str, str]:
        """Map each column of a table to its DuckDB type.

        Args:
            table: Bare table name.
            schema: DuckLake schema name.

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
    def _table_id(self, table: str, schema: str) -> int | None:
        """Resolve the DuckLake ``table_id`` of an active table.

        Args:
            table: Bare table name.
            schema: DuckLake schema name.

        Returns:
            int | None: The ``table_id``, or None when not found.

        Raises:
            duckdb.Error: When the internal metadata catalog cannot be read (no
                DuckLake catalog attached).
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

    # Résolution non fatale de l'identifiant DuckLake d'une table
    def _safe_table_id(self, table: str, schema: str) -> int | None:
        """Non-fatal wrapper of :meth:`_table_id` (DEBUG line on failure).

        Args:
            table: Bare table name.
            schema: DuckLake schema name.

        Returns:
            int | None: The ``table_id``, or None when not found or unreadable.
        """
        try:
            return self._table_id(table, schema)
        except Exception as e:
            self.logger.debug(f"table_id of {schema}.{table} unreadable: {e}")
            return None

    # Fichiers de suppression vivants et petits fichiers d'une table
    def _measure_files(self, report: StorageReport, table_id: int) -> None:
        """Fill the live delete files and the small-file count of ``report``.

        ``ducklake_table_info`` also counts the delete files whose data file was
        removed by a full ``DELETE`` (measured after ``recluster``: their
        ``end_snapshot`` stays NULL and ``rewrite_data_files`` cannot clear them),
        so only the delete files targeting an active data file are counted here.

        Args:
            report: Storage report updated in place.
            table_id: DuckLake ``table_id`` of the measured table.
        """
        meta = self._metadata_catalog()
        try:
            # Fichiers de suppression visant un fichier de données actif
            row = self.conn.execute(
                f"""
                SELECT count(*), coalesce(sum(d.file_size_bytes), 0)
                FROM {meta}.ducklake_delete_file d
                JOIN {meta}.ducklake_data_file f
                    ON f.data_file_id = d.data_file_id
                WHERE d.table_id = ? AND d.end_snapshot IS NULL
                  AND f.end_snapshot IS NULL
                """,
                [table_id],
            ).fetchone()
            if row is not None:
                report.delete_file_count = int(row[0])
                report.delete_bytes = int(row[1])
                report.delete_ratio = (
                    report.delete_bytes / report.total_bytes
                    if report.total_bytes
                    else 0.0
                )
            # Fichiers de données actifs sous le seuil de taille
            row = self.conn.execute(
                f"SELECT count(*) FROM {meta}.ducklake_data_file"
                " WHERE table_id = ? AND end_snapshot IS NULL"
                " AND file_size_bytes < ?",
                [table_id, report.min_file_size_bytes],
            ).fetchone()
            report.small_file_count = int(row[0]) if row is not None else 0
        except Exception as e:
            # Logging
            self.logger.warning(
                f"storage_report {report.schema}.{report.table}: files: {e}"
            )

    # Requête SQL des plages [min, max] typées des fichiers actifs
    def _file_ranges_query(
        self, table: str, schema: str, column: str, table_id: int | None
    ) -> str | None:
        """Build the query listing active files' typed ``[lo, hi]`` on ``column``.

        Args:
            table: Bare table name.
            schema: DuckLake schema name.
            column: Column whose statistics are read.
            table_id: DuckLake ``table_id`` of the table, or None when unresolved.

        Returns:
            str | None: A query returning ``(data_file_id, record_count, lo, hi)``,
            or None when the table or column cannot be resolved.

        Raises:
            ValueError: If ``column`` does not exist in the table.
        """
        columns = self._table_columns(table, schema)
        if column not in columns:
            raise ValueError(f"Column {column!r} does not exist in {schema}.{table}")
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
            WHERE f.table_id = {int(table_id)} AND s.column_id = {int(column_id)}
              AND f.end_snapshot IS NULL
        """

    # Identifiant DuckLake d'une colonne active
    def _column_id(self, table_id: int, column: str) -> int | None:
        """Resolve the DuckLake ``column_id`` of an active column.

        Args:
            table_id: DuckLake ``table_id``.
            column: Column name.

        Returns:
            int | None: The ``column_id``, or None when not found.
        """
        # Extraction de l'identifiant de la colonne
        row = self.conn.execute(
            f"SELECT column_id FROM {self._metadata_catalog()}.ducklake_column"
            " WHERE table_id = ? AND column_name = ? AND end_snapshot IS NULL",
            [table_id, column],
        ).fetchone()
        return int(row[0]) if row is not None else None

    # Part des fichiers dont la plage chevauche celle d'un autre fichier
    def _overlap_ratio(
        self, table: str, schema: str, column: str, table_id: int | None
    ) -> float | None:
        """Compute the share of active files whose range overlaps another's.

        Empty files and files without statistics are excluded. Overlap uses strict
        inequalities (``a.lo < b.hi AND b.lo < a.hi``) so files sharing a single
        boundary value — the normal outcome of a sorted rewrite with duplicate keys
        — do not count; identical ranges always do. Computed in SQL, so no typed
        value is fetched into Python.

        Args:
            table: Bare table name.
            schema: DuckLake schema name.
            column: Column whose statistics are compared.
            table_id: DuckLake ``table_id`` of the table, or None when unresolved.

        Returns:
            float | None: Ratio in ``[0, 1]`` (``0.0`` with at most one file), or
            None when the table/column cannot be resolved.
        """
        # Construction de la requête de l'intervalle de la colonne
        query = self._file_ranges_query(table, schema, column, table_id)
        if query is None:
            return None
        # Exécution de la requête
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
    def _safe_overlap_ratio(
        self, table: str, schema: str, column: str, table_id: int | None
    ) -> float | None:
        """Non-fatal wrapper of :meth:`_overlap_ratio` (warning logged on failure).

        Args:
            table: Bare table name.
            schema: DuckLake schema name.
            column: Column whose statistics are compared.
            table_id: DuckLake ``table_id`` of the table, or None when unresolved.

        Returns:
            float | None: The ratio, or None on failure.
        """
        try:
            # Calcul de ratio
            return self._overlap_ratio(table, schema, column, table_id)
        except Exception as e:
            # Logging
            self.logger.warning(f"overlap ratio of {schema}.{table}.{column}: {e}")
            return None

    # État des lignes inlinées d'une table
    def _inlined_state(
        self, table_id: int | None, table: str, schema: str
    ) -> tuple[int | None, bool]:
        """Measure the unflushed inlined rows of a table.

        Counts the live rows (``end_snapshot IS NULL``) of every inlined-data table
        registered for the table in ``ducklake_inlined_data_tables`` (measured:
        emptied by ``flush_inlined_data``). Falls back to a catalog-wide boolean
        read from ``ducklake_snapshots.changes`` (an ``inlined_insert`` more recent
        than the last ``flushed_inlined``) when the internal tables are unreadable.

        Args:
            table_id: DuckLake ``table_id`` of the table, or None when unresolved.
            table: Bare table name (for the log lines).
            schema: DuckLake schema name (for the log lines).

        Returns:
            tuple[int | None, bool]: ``(inlined_rows, has_inlined_data)``;
            ``inlined_rows`` is None when only the boolean fallback was available.
        """
        # Extraction des métadonnées du catalogue
        meta = self._metadata_catalog()
        if table_id is not None:
            try:
                # Sélection des noms de tables
                names = self.conn.execute(
                    f"SELECT table_name FROM {meta}.ducklake_inlined_data_tables"
                    " WHERE table_id = ?",
                    [table_id],
                ).fetchall()
                # Initialisation du nombre de lignes total
                # associé à l'identifiant de table
                total = 0
                # Parcours des noms de table
                for (name,) in names:
                    # Comptable des snapshots actifs associés
                    row = self.conn.execute(
                        f"SELECT count(*) FROM {meta}.{quote_ident(name)}"
                        " WHERE end_snapshot IS NULL"
                    ).fetchone()
                    total += int(row[0]) if row is not None else 0
                return total, total > 0
            except Exception as e:
                # Logging
                self.logger.debug(f"inlined rows of {schema}.{table} unreadable: {e}")
        # Repli : indicateur booléen depuis l'historique des snapshots
        try:
            row = self.conn.execute(
                "SELECT max(snapshot_id) FILTER (WHERE changes::VARCHAR LIKE"
                " '%inlined_insert%'), max(snapshot_id) FILTER (WHERE"
                " changes::VARCHAR LIKE '%flushed_inlined%')"
                f" FROM ducklake_snapshots({quote_literal(self.catalog_alias)})"
            ).fetchone()
        except Exception as e:
            # Logging
            self.logger.warning(f"inlined state of {schema}.{table} unreadable: {e}")
            return None, False
        if row is None or row[0] is None:
            return None, False
        return None, row[1] is None or row[0] > row[1]


# Réexportation des procédures pour les imports depuis ce module
__all__ = [
    "DEFAULT_DELETE_THRESHOLD",
    "DEFAULT_TARGET_FILE_SIZE_BYTES",
    "DuckLakeMaintenance",
    "MaintenancePolicy",
    "StorageReport",
]
