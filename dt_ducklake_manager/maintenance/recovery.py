"""
Recovery of a result set through DuckLake time travel.

DuckLake keeps every committed snapshot until it is explicitly expired, so the
recovery mechanism of the package is time travel, not an application-level
backup: a failed operation is already rolled back by its own DuckDB transaction,
and a committed but unwanted state is undone by copying a previous snapshot back.

The procedure, implemented by :class:`DatabaseRecoveryManager`:

1. :meth:`DatabaseRecoveryManager.list_ducklake_snapshots` lists the snapshots,
   with the ``author`` (``run_id``), ``commit_message`` and ``commit_extra_info``
   recorded by each write, to identify the snapshot to go back to;
2. :meth:`DatabaseRecoveryManager.restore_snapshot` copies the tables of the
   result set as they were at that snapshot
   (``SELECT * FROM <table> AT (VERSION => n)``) back into the current catalog,
   in a single transaction that produces a new snapshot: the history is kept, and
   the restoration itself can be undone the same way.
"""

# Importation des modules
# Modules de base
import os
import time
from datetime import datetime
from typing import Any

# DuckDB
import duckdb
import narwhals as nw

# Import des utilitaires
from ..reporting import (
    OperationReport,
    _current_snapshot_id,
    _set_commit_message,
    _table_changes_counts,
)
from ..utils.logger import _init_logger
from ..utils.sql import SchemaScoped, quote_ident, quote_literal, resolve_catalog

# Tables d'un jeu de résultats restaurées par défaut
RESTORABLE_TABLES: tuple[str, ...] = ("fact_table", "metadata", "dataset_metadata")


# Classe de récupération de la base de données
class DatabaseRecoveryManager(SchemaScoped):
    """
    Lists the DuckLake snapshots and restores a result set to one of them.

    Recovery from a bad write is **not** an application-level restore: DuckLake
    persists the full snapshot history, so the mechanism is to list the snapshots
    (:meth:`list_ducklake_snapshots`), pick one, and copy its tables back
    (:meth:`restore_snapshot`). No backup file is written, and none is needed.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection.
        catalog_alias (str): Alias of the attached DuckLake catalog.
        schema (str): DuckLake schema to recover.
        logger: Logger instance for recovery tracking.

    Examples:
        >>> recovery = DatabaseRecoveryManager(conn, schema='predictions')
        >>> print(recovery.list_ducklake_snapshots())
        >>> report = recovery.restore_snapshot(17, run_id='rollback-run-42')
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        log_filename: str | os.PathLike[str] | None = None,
        catalog_alias: str = "db",
        schema: str = "main",
    ):
        """
        Initialize the database recovery manager.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            log_filename: Path to log file.
            catalog_alias: Alias of the attached DuckLake catalog, used to query
                the snapshots via ``ducklake_snapshots()``. Defaults to ``'db'``.
            schema: DuckLake schema to recover. A catalog can host several schemas;
                only this one is restored. Defaults to ``'main'``.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> recovery = DatabaseRecoveryManager(conn, schema='predictions')
        """
        # Initialisation de la connexion DuckLake.
        self.conn = connection if connection is not None else duckdb.connect(":memory:")
        self.catalog_alias = catalog_alias
        self.schema = schema

        # Alias de catalogue effectif : qualification par le catalogue uniquement
        # s'il est réellement attaché (None pour les connexions in-memory des tests).
        self._catalog = resolve_catalog(self.conn, self.catalog_alias)

        # Initialisation du logger nommé.
        # Chemin par défaut centralisé dans utils.logger : <cwd>/logs/<name>.log.
        self.logger = _init_logger(filename=log_filename, name="database_recovery")

    # Méthode publique de consultation de l'historique des snapshots DuckLake
    def list_ducklake_snapshots(self) -> nw.DataFrame[Any] | None:
        """Return the full DuckLake snapshot history as a narwhals DataFrame.

        Convenience wrapper around the ``ducklake_snapshots(catalog)`` table
        function, and the entry point of the recovery procedure: pick a
        ``snapshot_id`` here, then pass it to :meth:`restore_snapshot`. The history
        is catalog-wide, covering every schema it holds; ``author`` carries the
        ``run_id`` of the write that produced each snapshot.

        Returns:
            nw.DataFrame | None: One row per snapshot (``snapshot_id``,
            ``snapshot_time``, ``schema_version``, ``changes``, ``author``,
            ``commit_message``, ``commit_extra_info``), sorted by ``snapshot_id``
            descending (pyarrow backend: ``.to_native()`` gives the
            ``pyarrow.Table``). None if the catalog cannot be queried (e.g. a plain
            in-memory connection with no DuckLake catalog attached).

        Example:
            >>> snapshots = recovery.list_ducklake_snapshots()
            >>> if snapshots is not None:
            ...     print(snapshots.select('snapshot_id', 'author', 'commit_message'))
        """
        try:
            # Inventaire des snapshots
            snapshots = self.conn.execute(
                "SELECT * FROM ducklake_snapshots("
                f"{quote_literal(self.catalog_alias)}) ORDER BY snapshot_id DESC"
            ).to_arrow_table()
        except Exception as e:
            # Logging
            self.logger.error(f"Could not list the DuckLake snapshots: {e}")
            return None
        history: nw.DataFrame[Any] = nw.from_native(snapshots, eager_only=True)
        return history

    # Méthode de restauration des tables du jeu de résultats à un snapshot
    def restore_snapshot(
        self,
        snapshot_id: int,
        tables: tuple[str, ...] | list[str] = RESTORABLE_TABLES,
        run_id: str | None = None,
        commit_message: str | None = None,
    ) -> OperationReport:
        """
        Restore the tables of the result set to their state at ``snapshot_id``.

        For each table, the rows it held at the snapshot are first copied into a
        temporary table (``CREATE TEMP TABLE … AS SELECT * FROM <table> AT
        (VERSION => n)``), outside the transaction: read inside the same
        transaction as the ``DELETE`` that empties the table, the time-travel read
        returns no row (measured). Then, in a single transaction, every table is
        emptied and refilled from its copy, and the commit is annotated with
        ``run_id``/``commit_message``. The restoration is a new snapshot: the
        states in between stay readable by time travel, and the restoration can
        itself be undone.

        Only the rows are restored, not the table structure: a table whose columns
        changed since the snapshot (``add_columns``, ``delete_columns``,
        ``allow_new_columns``) is refused before anything is modified, since
        re-creating it would lose its DuckLake identity (table id, history). Such a
        state is read directly with ``SELECT * FROM <table> AT (VERSION => n)``.

        Args:
            snapshot_id: Identifier of the snapshot to restore, as listed by
                :meth:`list_ducklake_snapshots`.
            tables: Bare names of the tables of the schema to restore. Defaults to
                the three tables of a result set.
            run_id: Run identifier recorded as the ``author`` of the restoration
                snapshot.
            commit_message: Commit message recorded on the restoration snapshot.
                Defaults to ``'restore <schema> to snapshot <n>'``.

        Returns:
            OperationReport: ``operation='restore_snapshot'``, with the fact table
            rows before/after, the snapshots and, on a real catalog, the exact
            row changes of the fact table.

        Raises:
            ValueError: If ``tables`` is empty, if no DuckLake catalog is attached,
                if the snapshot does not exist (or has expired), if a table did not
                exist at the snapshot, or if its columns changed since.
            duckdb.Error: If the restoration fails; the transaction is rolled back
                and the tables are left untouched.

        Examples:
            >>> recovery.restore_snapshot(17)
            >>> recovery.restore_snapshot(17, tables=['fact_table'],
            ...     run_id='rollback-run-42')
        """
        # Validation des arguments
        if not tables:
            raise ValueError("tables must name at least one table to restore")
        if self._catalog is None:
            raise ValueError(
                f"No DuckLake catalog {self.catalog_alias!r} is attached: time travel"
                " is unavailable"
            )

        # Existence du snapshot visé
        snapshots = self.conn.execute(
            "SELECT snapshot_id FROM ducklake_snapshots("
            f"{quote_literal(self.catalog_alias)}) WHERE snapshot_id = ?",
            [int(snapshot_id)],
        ).fetchone()
        if snapshots is None:
            raise ValueError(
                f"Snapshot {snapshot_id} does not exist in catalog"
                f" {self.catalog_alias!r} (unknown or expired)"
            )

        # Structure identique entre le snapshot et l'état courant, table par table
        for table in tables:
            self._check_same_columns(table, int(snapshot_id))

        # Avant-état
        start_time = time.time()
        report = OperationReport(
            operation="restore_snapshot",
            schema=self.schema,
            run_id=run_id,
            started_at=datetime.now(),
            duration_seconds=0.0,
            rows_before=self._count_rows("fact_table"),
            snapshot_before=_current_snapshot_id(self.conn, self._catalog, self.logger),
        )

        # Copie des états passés dans des tables temporaires, hors transaction
        copies = {table: f"_restore_{table}" for table in tables}
        try:
            for table, copy in copies.items():
                self.conn.execute(
                    f"CREATE OR REPLACE TEMP TABLE {quote_ident(copy)} AS SELECT *"
                    f" FROM {self._qualified(table)} AT (VERSION => {int(snapshot_id)})"
                )

            # Vidage et remplissage de toutes les tables dans une seule transaction
            self.conn.begin()
            try:
                for table, copy in copies.items():
                    self.conn.execute(f"DELETE FROM {self._qualified(table)}")
                    self.conn.execute(
                        f"INSERT INTO {self._qualified(table)}"
                        f" SELECT * FROM temp.main.{quote_ident(copy)}"
                    )
                _set_commit_message(
                    self.conn,
                    self._catalog,
                    run_id,
                    commit_message
                    or f"restore {self.schema} to snapshot {snapshot_id}",
                    {
                        "operation": "restore_snapshot",
                        "schema": self.schema,
                        "snapshot_id": int(snapshot_id),
                        "tables": list(tables),
                    },
                    self.logger,
                )
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                report.warnings.append(f"restore_snapshot failed: {e}")
                report.duration_seconds = time.time() - start_time
                self.logger.error(
                    f"restore_snapshot {self.schema} to {snapshot_id} failed and was"
                    f" rolled back: {e}"
                )
                raise
        finally:
            # Suppression des copies temporaires, succès comme échec
            for copy in copies.values():
                self.conn.execute(f"DROP TABLE IF EXISTS temp.main.{quote_ident(copy)}")

        # Après-état et comptages exacts des changements de la table des faits
        report.rows_after = self._count_rows("fact_table")
        report.snapshot_after = _current_snapshot_id(
            self.conn, self._catalog, self.logger
        )
        if "fact_table" in tables:
            changes = _table_changes_counts(
                self.conn,
                self._catalog,
                self.schema,
                "fact_table",
                report.snapshot_before,
                report.snapshot_after,
                self.logger,
            )
            report.rows_inserted = changes.get("insert", 0)
            report.rows_deleted = changes.get("delete", 0)
        report.duration_seconds = time.time() - start_time

        # Logging
        self.logger.info(f"{report.summary()} (restored snapshot {snapshot_id})")
        return report

    # Méthode de contrôle de la stabilité des colonnes d'une table
    def _check_same_columns(self, table: str, snapshot_id: int) -> None:
        """Check that ``table`` has the same columns now as at ``snapshot_id``.

        Args:
            table: Bare table name.
            snapshot_id: Snapshot the table is about to be restored to.

        Raises:
            ValueError: If the table did not exist at the snapshot, does not exist
                anymore, or if its column names, order or types changed since.
        """
        qualified = self._qualified(table)
        # Colonnes courantes
        try:
            current = self.conn.execute(f"DESCRIBE {qualified}").fetchall()
        except duckdb.Error as e:
            raise ValueError(f"Table {table!r} does not exist anymore: {e}") from e
        # Colonnes au snapshot
        try:
            past = self.conn.execute(
                f"DESCRIBE SELECT * FROM {qualified} AT (VERSION => {snapshot_id})"
            ).fetchall()
        except duckdb.Error as e:
            raise ValueError(
                f"Table {table!r} cannot be read at snapshot {snapshot_id}: {e}"
            ) from e
        # Comparaison des noms et types, dans l'ordre
        current_columns = [(row[0], row[1]) for row in current]
        past_columns = [(row[0], row[1]) for row in past]
        if current_columns != past_columns:
            raise ValueError(
                f"The columns of {table!r} changed since snapshot {snapshot_id}"
                f" ({past_columns} -> {current_columns}); restore_snapshot only"
                f" restores rows. Read the past state with SELECT * FROM"
                f" {qualified} AT (VERSION => {snapshot_id})"
            )
