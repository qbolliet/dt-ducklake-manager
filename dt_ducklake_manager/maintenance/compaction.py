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
*cleanup* only in planned maintenance. ``full_maintenance`` runs the safe,
always-after-write steps (flush -> merge -> rewrite) plus expire/cleanup with an
explicit retention — callers that only want the safe steps should call
``merge_files``/``rewrite_data_files``/``flush_inlined_data`` directly instead
(see ``DatabaseUpdater``/``DatabaseDeleter``, which never call
``expire_snapshots``/``cleanup_files``/``delete_orphaned_files``).
"""

# Importation des modules
# Modules de base
import os
import time
from datetime import datetime, timedelta
from typing import Any

# DuckDB
import duckdb

# Rapport d'opération
from ..reporting import OperationReport, _current_snapshot_id, _table_info

# Module d'initialisation du logger
from ..utils.logger import _init_logger
from ..utils.sql import quote_ident


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
        >>> maint.full_maintenance('main', 'fact_table')
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
        self, table: str | None = None
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
                    f" table_name := '{table}', schema_name := '{self.schema}')"
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
    # Méthode de maintenance complète
    # ---------------------------------------------------------------------------

    # Exécution de l'ensemble des opérations de maintenance dans l'ordre recommandé
    def full_maintenance(
        self,
        schema: str,
        table: str,
        older_than_days: int = 30,
    ) -> OperationReport:
        """
        Run all maintenance operations in the recommended order.

        Executes in sequence: ``flush_inlined_data`` → ``merge_files`` →
        ``rewrite_data_files`` → ``expire_snapshots`` → ``cleanup_files``. Each step
        is wrapped in a ``try/except`` so a failure in one step does not block the
        others. Includes ``expire_snapshots``/``cleanup_files`` — this is planned
        maintenance with an explicit retention, unlike the after-write compaction run
        by ``DatabaseUpdater``/``DatabaseDeleter`` (which only run the first three,
        safe steps). Unlike the write operations, this method does not accept
        ``run_id``/``commit_message``: it is a maintenance sequence, not a single
        traceable commit — each step already commits (and logs) on its own.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).
            table (str): Table name to compact (passed to ``flush_inlined_data``,
                ``merge_files`` and ``rewrite_data_files``).
            older_than_days (int): Passed to ``expire_snapshots``. Defaults to 30.

        Returns:
            OperationReport: ``report.maintenance`` carries every step's counters
            (zeros included, never omitted); ``rows_before``/``rows_after`` and the
            file/snapshot fields bracket the whole sequence.

        Examples:
            >>> maint.full_maintenance('main', 'fact_table')
            >>> maint.full_maintenance('main', 'fact_table', older_than_days=7)
        """
        # Logging
        self.logger.info(
            f"Beginning the full DuckLake maintenance : schema={schema}, table={table}"
        )

        # Moment du début de la maintenance
        start_time = time.time()
        started_at = datetime.now()
        # Extraction des informations sur la base de données avant la maintenance
        rows_before = self._count_rows(schema, table)
        info_before = _table_info(
            self.conn, self.catalog_alias, schema, table, self.logger
        )
        files_before, bytes_before, _, _ = info_before or (0, 0, 0, 0)
        snapshot_before = _current_snapshot_id(
            self.conn, self.catalog_alias, self.logger
        )
        # Initialisation du rapport
        report = OperationReport(
            operation="maintenance",
            schema=schema,
            run_id=None,
            started_at=started_at,
            duration_seconds=0.0,
            rows_before=rows_before,
            files_before=files_before,
            bytes_before=bytes_before,
            snapshot_before=snapshot_before,
        )

        # Chaque étape est enveloppée dans un try/except pour garantir que
        # l'échec d'une étape ne bloque pas les étapes suivantes.

        # Étape 0 : écriture en Parquet des lignes inlinées dans le catalogue, avant
        # toute opération de compaction portant sur les fichiers
        try:
            flushed = self.flush_inlined_data(table)
            report.maintenance["flush_inlined_rows"] = sum(r[2] for r in flushed)
        except Exception as e:
            # Logging
            self.logger.warning(f"full_maintenance — flush_inlined_data failed : {e}")
            # Ajout au rapport
            report.warnings.append(f"flush_inlined_data failed: {e}")

        # Étape 1 : fusion des petits fichiers Parquet adjacents
        try:
            _, _, merge_processed, merge_created = self.merge_files(schema, table)
            report.maintenance["merge_files_processed"] = merge_processed
            report.maintenance["merge_files_created"] = merge_created
        except Exception as e:
            # Logging
            self.logger.warning(f"full_maintenance — merge_files failed : {e}")
            # Ajout au rapport
            report.warnings.append(f"merge_files failed: {e}")

        # Étape 2 : réécriture des fichiers contenant des suppressions
        try:
            _, _, rewrite_processed, rewrite_created = self.rewrite_data_files(
                schema, table
            )
            report.maintenance["rewrite_files_processed"] = rewrite_processed
            report.maintenance["rewrite_files_created"] = rewrite_created
        except Exception as e:
            # Logging
            self.logger.warning(f"full_maintenance — rewrite_data_files failed : {e}")
            # Ajout au rapport
            report.warnings.append(f"rewrite_data_files failed: {e}")

        # Étape 3 : expiration des anciens snapshots
        try:
            expired = self.expire_snapshots(schema, older_than_days=older_than_days)
            report.maintenance["expired_snapshots"] = len(expired)
        except Exception as e:
            # Logging
            self.logger.warning(f"full_maintenance — expire_snapshots failed : {e}")
            # Ajout au rapport
            report.warnings.append(f"expire_snapshots failed: {e}")

        # Étape 4 : suppression des fichiers Parquet orphelins
        try:
            cleaned = self.cleanup_files(schema)
            report.maintenance["cleaned_files"] = len(cleaned)
        except Exception as e:
            # Logging
            self.logger.warning(f"full_maintenance — cleanup_files failed : {e}")
            # Ajout au rapport
            report.warnings.append(f"cleanup_files failed: {e}")

        # Calcul d'informations sur la base de données après l'exécution de la maintenance
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
        self.logger.info(
            f"Maintenance of the DuckLake is finished : schema={schema}, table={table}"
        )
        self.logger.info(report.summary())
        return report

    # Méthode auxiliaire de comptage des lignes d'une table
    def _count_rows(self, schema: str, table: str) -> int:
        """Count the rows of a table, qualified by ``schema``.

        Args:
            schema: DuckLake schema name.
            table: Bare table name.

        Returns:
            int: Row count, or ``0`` if it cannot be read.
        """
        try:
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {quote_ident(schema)}.{quote_ident(table)}"
            ).fetchone()
            return int(row[0]) if row is not None else 0
        except Exception:
            return 0
