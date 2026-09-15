"""
Operation reports and DuckLake run traceability.

Every public write operation (``build_schema``, ``update_database``, ``add_columns``,
``delete_rows``, ``delete_columns``, ``DuckLakeMaintenance.full_maintenance``) builds
an :class:`OperationReport` describing what it actually did, measured from DuckLake's
public introspection functions rather than estimated in Python:

- ``ducklake_table_info(catalog)`` : ``file_count``/``file_size_bytes``/
  ``delete_file_count``/``delete_file_size_bytes``, before and after the operation.
- ``ducklake_snapshots(catalog)`` : the current ``snapshot_id``.
- ``ducklake_table_changes(catalog, schema, table, start, end)`` : exact row counts by
  ``change_type`` between two snapshots. **Measured**: ``start_snapshot`` is inclusive,
  so the range that captures exactly one operation's own changes is
  ``(snapshot_before + 1, snapshot_after)``, not ``(snapshot_before, snapshot_after)``.
- The result tuples already returned by the maintenance procedures
  (``merge_files``/``rewrite_data_files``/``flush_inlined_data``/...).

On a connection with no DuckLake catalog attached (the in-memory connections used by
unit tests), every DuckLake-only measurement gracefully falls back to ``None``/``0``
with a DEBUG log line, and ``ducklake_set_commit_message`` is skipped the same way —
the report is still built and returned, only its DuckLake-specific fields stay empty.

``ducklake_table_info`` does not expose a schema *name*, only a ``schema_id``, and it
does not accept a table/schema filter (measured via ``duckdb_functions()``): resolving
``schema_id -> schema_name`` therefore goes through the internal
``__ducklake_metadata_<alias>.ducklake_schema`` table, the same way the specification's
own annexe A resorts to internal tables for column statistics when no public function
exposes them.
"""

# Importation des modules
# Modules de base
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

# DuckDB
import duckdb

__all__ = ["OperationReport"]


# Dataclass du rapport d'une opération d'écriture ou de maintenance
@dataclass
class OperationReport:
    """
    Report of what a single write or maintenance operation actually did.

    Built from DuckLake's public introspection functions (see the module docstring),
    never from Python-side estimates. Fields default to their "nothing happened yet"
    value so a partial report (before-state known, after-state still at default) can be
    built and logged even when the operation fails mid-way.

    Attributes:
        operation (str): Operation kind, e.g. ``'build'``, ``'update'``,
            ``'add_columns'``, ``'delete_columns'``, ``'delete_rows'``,
            ``'maintenance'``.
        schema (str): DuckLake schema the operation targeted.
        run_id (str | None): Caller-supplied run identifier, also recorded in the
            DuckLake commit message (``ducklake_set_commit_message``) when a real
            catalog is attached.
        started_at (datetime): Wall-clock time the operation started.
        duration_seconds (float): Wall-clock duration of the operation.
        rows_inserted (int): Rows inserted, from ``ducklake_table_changes``
            (``change_type = 'insert'``). ``0`` when no DuckLake catalog is attached.
        rows_updated (int): Rows updated, from ``ducklake_table_changes``
            (``change_type = 'update_postimage'``).
        rows_deleted (int): Rows deleted, from ``ducklake_table_changes``
            (``change_type = 'delete'``).
        rows_before (int): ``COUNT(*)`` of the fact table before the operation.
            Computable without a real DuckLake catalog.
        rows_after (int): ``COUNT(*)`` of the fact table after the operation.
        columns_added (list[str]): Value columns added to the fact table.
        columns_dropped (list[str]): Value columns dropped from the fact table.
        metadata_changes (list[str]): Human-readable metadata changes, e.g.
            ``"is_categorical(region): False -> True"`` or a type-widening line.
        snapshot_before (int | None): DuckLake ``snapshot_id`` before the operation.
        snapshot_after (int | None): DuckLake ``snapshot_id`` after the operation
            (and after any post-commit compaction the caller ran).
        files_before (int): Parquet file count before the operation.
        files_after (int): Parquet file count after the operation (and after any
            post-commit compaction).
        bytes_before (int): Total Parquet file size (bytes) before the operation.
        bytes_after (int): Total Parquet file size (bytes) after the operation.
        maintenance (dict[str, int]): Counters returned by post-commit maintenance
            procedures (e.g. ``{'merge_files_processed': 0, 'merge_files_created':
            0, 'rewrite_files_processed': 3, 'rewrite_files_created': 1}``). A step
            that ran and changed nothing is present with a ``0`` value, never
            omitted.
        warnings (list[str]): Non-fatal warnings collected during the operation,
            each also logged at WARNING as it is produced.
    """

    operation: str
    schema: str
    run_id: str | None
    started_at: datetime
    duration_seconds: float
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_deleted: int = 0
    rows_before: int = 0
    rows_after: int = 0
    columns_added: list[str] = field(default_factory=list)
    columns_dropped: list[str] = field(default_factory=list)
    metadata_changes: list[str] = field(default_factory=list)
    snapshot_before: int | None = None
    snapshot_after: int | None = None
    files_before: int = 0
    files_after: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    maintenance: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Méthode de construction de la ligne de synthèse lisible (log INFO)
    def summary(self) -> str:
        """
        Build the one-line, human-readable summary logged at INFO on success.

        Only non-trivial parts are included (a zero row delta, or no column
        change, is simply omitted from the line — the underlying counters, not the
        summary line, are where zeros are always explicit, per the maintenance
        procedures' own logging).

        Returns:
            str: A single summary line, e.g. ``"update main [run-42]: +1240 rows,
            ~380 updated, +2 columns (score, rank), 3 -> 2 files (48.2 -> 31.7 MB),
            snapshot 17 -> 18, 4.1s"``.

        Examples:
            >>> from datetime import datetime
            >>> r = OperationReport(
            ...     operation="update", schema="main", run_id="run-42",
            ...     started_at=datetime(2026, 1, 1), duration_seconds=4.1,
            ...     rows_inserted=1240, rows_updated=380,
            ...     files_before=3, files_after=2,
            ...     bytes_before=48_200_000, bytes_after=31_700_000,
            ...     snapshot_before=17, snapshot_after=18,
            ... )
            >>> r.summary()  # doctest: +NORMALIZE_WHITESPACE
            'update main [run-42]: +1240 rows, ~380 updated, 3 -> 2 files
            (48.2 -> 31.7 MB), snapshot 17 -> 18, 4.1s'
        """
        # En-tête : opération, schéma, run_id éventuel
        head = f"{self.operation} {self.schema}"
        if self.run_id is not None:
            head += f" [{self.run_id}]"

        # Parties optionnelles, omises quand triviales (delta nul)
        parts: list[str] = []
        # Lignes insérées
        if self.rows_inserted:
            parts.append(f"+{self.rows_inserted} rows")
        # Lignes mises à jour
        if self.rows_updated:
            parts.append(f"~{self.rows_updated} updated")
        # Lignes supprimées
        if self.rows_deleted:
            parts.append(f"-{self.rows_deleted} rows")
        # Colonnes ajoutées
        if self.columns_added:
            added = ", ".join(self.columns_added)
            parts.append(f"+{len(self.columns_added)} columns ({added})")
        # Colonnes supprimées
        if self.columns_dropped:
            dropped = ", ".join(self.columns_dropped)
            parts.append(f"-{len(self.columns_dropped)} columns ({dropped})")
        # Nombre de fichiers avant/après
        if self.files_before or self.files_after:
            before_mb = self.bytes_before / 1_000_000
            after_mb = self.bytes_after / 1_000_000
            parts.append(
                f"{self.files_before} -> {self.files_after} files"
                f" ({before_mb:.1f} -> {after_mb:.1f} MB)"
            )
        # Snapshots
        if self.snapshot_before is not None or self.snapshot_after is not None:
            parts.append(f"snapshot {self.snapshot_before} -> {self.snapshot_after}")
        parts.append(f"{self.duration_seconds:.1f}s")

        return f"{head}: " + ", ".join(parts)

    # Méthode de conversion en dictionnaire sérialisable
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the report to a plain, JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Every field of the dataclass, with ``started_at``
            rendered as an ISO-8601 string.

        Examples:
            >>> from datetime import datetime
            >>> r = OperationReport(
            ...     operation="update", schema="main", run_id=None,
            ...     started_at=datetime(2026, 1, 1), duration_seconds=1.0,
            ... )
            >>> r.to_dict()["operation"]
            'update'
            >>> r.to_dict()["started_at"]
            '2026-01-01T00:00:00'
        """
        # dataclasses.asdict copie récursivement les listes/dicts mutables
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        return data


# ---------------------------------------------------------------------------
# Fonctions privées de collecte, indépendantes de toute classe : réutilisables
# aussi bien par BaseSchemaManager (opérations) que par DuckLakeTablesBuilder
# (construction), qui n'hérite pas de BaseSchemaManager.
# ---------------------------------------------------------------------------


# Fonction de lecture des statistiques de fichiers d'une table (avant/après)
def _table_info(
    conn: duckdb.DuckDBPyConnection,
    catalog_alias: str | None,
    schema: str,
    table: str,
    logger: logging.Logger,
) -> tuple[int, int, int, int] | None:
    """
    Read ``(file_count, file_size_bytes, delete_file_count, delete_file_size_bytes)``
    for one table via ``ducklake_table_info``.

    ``ducklake_table_info(catalog)`` returns one row per table of the *whole*
    catalog (every schema) and exposes only a ``schema_id`` (measured via
    ``duckdb_functions()`` : no schema/table filter parameter exists), so resolving
    it to a schema name goes through the internal
    ``__ducklake_metadata_<alias>.ducklake_schema`` table — the public function
    itself is still preferred over the internal ``ducklake_data_file`` tables for
    the measurements themselves, per the specification.

    Args:
        conn: DuckDB connection with the catalog attached.
        catalog_alias: Effective attached catalog alias (``None`` skips the call —
            no real DuckLake catalog, e.g. an in-memory test connection).
        schema: DuckLake schema of the target table.
        table: Bare table name (e.g. ``'fact_table'``).
        logger: Logger for the DEBUG skip line / WARNING on failure.

    Returns:
        tuple[int, int, int, int] | None: ``(file_count, file_size_bytes,
        delete_file_count, delete_file_size_bytes)``, or ``None`` when no real
        DuckLake catalog is attached or the table cannot be found.
    """
    # Vérification qu'un catalogue est spécifié
    if catalog_alias is None:
        # Logging
        logger.debug(
            f"_table_info skipped for {schema}.{table}: no DuckLake catalog attached"
        )
        return None
    # Exécution de la requête de collecte des informations
    try:
        row = conn.execute(
            f"""
            SELECT ti.file_count, ti.file_size_bytes, ti.delete_file_count,
                   ti.delete_file_size_bytes
            FROM ducklake_table_info('{catalog_alias}') ti
            JOIN __ducklake_metadata_{catalog_alias}.ducklake_schema s
                ON s.schema_id = ti.schema_id
            WHERE ti.table_name = ? AND s.schema_name = ?
            """,
            [table, schema],
        ).fetchone()
        if row is None:
            return None
        return (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
    except Exception as e:
        # Logging
        logger.warning(f"_table_info failed for {schema}.{table}: {e}")
        return None


# Fonction de lecture du snapshot_id courant du catalogue
def _current_snapshot_id(
    conn: duckdb.DuckDBPyConnection,
    catalog_alias: str | None,
    logger: logging.Logger,
) -> int | None:
    """
    Read the current (highest) DuckLake ``snapshot_id`` of the catalog.

    Args:
        conn: DuckDB connection with the catalog attached.
        catalog_alias: Effective attached catalog alias (``None`` skips the call).
        logger: Logger for the DEBUG skip line / WARNING on failure.

    Returns:
        int | None: The current ``snapshot_id``, or ``None`` when no real DuckLake
        catalog is attached or the call fails.
    """
    # Vérification qu'un catalogue est spécifié
    if catalog_alias is None:
        # Logging
        logger.debug("_current_snapshot_id skipped: no DuckLake catalog attached")
        return None
    # Exécution de la requête d'extraction de l'identifiant du snapshot actuel
    try:
        row = conn.execute(
            f"SELECT max(snapshot_id) FROM ducklake_snapshots('{catalog_alias}')"
        ).fetchone()
        return int(row[0]) if row is not None and row[0] is not None else None
    except Exception as e:
        # Logging
        logger.warning(f"_current_snapshot_id failed: {e}")
        return None


# Fonction de comptage exact des changements de lignes entre deux snapshots
def _table_changes_counts(
    conn: duckdb.DuckDBPyConnection,
    catalog_alias: str | None,
    schema: str,
    table: str,
    snapshot_before: int | None,
    snapshot_after: int | None,
    logger: logging.Logger,
) -> dict[str, int]:
    """
    Count row changes by ``change_type`` for exactly one operation's own commit.

    **Measured** (``ducklake_table_changes``): ``start_snapshot`` is inclusive, so
    the range capturing only this operation's own changes — excluding whatever
    already produced ``snapshot_before`` — is ``(snapshot_before + 1,
    snapshot_after)``, not ``(snapshot_before, snapshot_after)``.

    Args:
        conn: DuckDB connection with the catalog attached.
        catalog_alias: Effective attached catalog alias (``None`` skips the call).
        schema: DuckLake schema of the target table.
        table: Bare table name (e.g. ``'fact_table'``).
        snapshot_before: Snapshot id before the operation. ``None`` skips the call
            (no real DuckLake catalog, or the table did not exist yet).
        snapshot_after: Snapshot id right after the operation's own commit.
        logger: Logger for the DEBUG skip line / WARNING on failure.

    Returns:
        dict[str, int]: Raw ``change_type -> count`` mapping (``insert``,
        ``delete``, ``update_preimage``, ``update_postimage``), or ``{}`` when
        unavailable.
    """
    # Vérification qu'un catalogue est spécifié
    if catalog_alias is None or snapshot_before is None or snapshot_after is None:
        # Logging
        logger.debug(
            f"_table_changes_counts skipped for {schema}.{table}: "
            "no DuckLake catalog attached or missing snapshot bound"
        )
        return {}
    # Exécution de la requête de comptage des modifications entre deux snapshots
    try:
        rows = conn.execute(
            f"""
            SELECT change_type, count(*) FROM ducklake_table_changes(
                '{catalog_alias}', '{schema}', '{table}', ?, ?
            ) GROUP BY 1
            """,
            [snapshot_before + 1, snapshot_after],
        ).fetchall()
        return {change_type: int(count) for change_type, count in rows}
    except Exception as e:
        # Logging
        logger.warning(f"_table_changes_counts failed for {schema}.{table}: {e}")
        return {}


# Fonction d'enregistrement du message de commit DuckLake (traçabilité des runs)
def _set_commit_message(
    conn: duckdb.DuckDBPyConnection,
    catalog_alias: str | None,
    run_id: str | None,
    commit_message: str | None,
    extra_info: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Record ``run_id``/``commit_message``/``extra_info`` on the current snapshot.

    Must be called **inside** the open transaction, before ``COMMIT`` — DuckLake
    then exposes ``author``, ``commit_message`` and ``commit_extra_info`` on the
    resulting row of ``ducklake_snapshots()``. Never raises: on a connection with no
    DuckLake catalog attached the call is skipped with a DEBUG line (the common case
    in unit tests, which mostly run against a plain in-memory DuckDB connection);
    any other failure is logged as a WARNING and swallowed, since a traceability
    side-channel must never fail the write it annotates. ``run_id``/``commit_message``
    are passed through even when ``None`` (measured: ``ducklake_set_commit_message``
    accepts ``NULL`` for both positional arguments).

    Args:
        conn: DuckDB connection with the catalog attached, inside an open
            transaction.
        catalog_alias: Effective attached catalog alias (``None`` skips the call).
        run_id: Caller-supplied run identifier, recorded as the snapshot's
            ``author``.
        commit_message: Caller-supplied commit message.
        extra_info: JSON-serializable payload, always carrying at least
            ``operation`` and ``schema`` (set by the caller).
        logger: Logger for the DEBUG skip line / WARNING on failure.
    """
    # Vérification qu'un catalogue est spécifié
    if catalog_alias is None:
        logger.debug(
            "_set_commit_message skipped: no DuckLake catalog attached"
            f" (run_id={run_id!r})"
        )
        return
    # Commit message
    try:
        conn.execute(
            f"CALL ducklake_set_commit_message('{catalog_alias}', ?, ?,"
            " extra_info := ?)",
            [run_id, commit_message, json.dumps(extra_info)],
        )
    except Exception as e:
        # Logging
        logger.warning(f"_set_commit_message failed (run_id={run_id!r}): {e}")
