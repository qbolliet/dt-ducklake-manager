"""
Thin wrappers around the DuckLake maintenance procedures.

DuckLake files are immutable: every UPDATE/DELETE produces new Parquet delta files
and, for a partial deletion, a ``-delete.parquet`` tombstone file, while the old
file stays referenced by prior snapshots (time travel). Left alone, small files and
tombstones accumulate and degrade read performance. Per procedure — effect / when /
risk:

- ``rewrite_data_files(delete_threshold)`` — rewrites files whose deleted-row
  share exceeds the threshold; **without an explicit threshold it is a no-op**.
  After every update/delete (``delete_threshold`` 0.1-0.3). Risk: none (old files
  stay readable via time travel).
- ``merge_files(max_file_size)`` — merges adjacent files into files of at most
  ``max_file_size`` bytes (the catalog's ``target_file_size`` by default). After
  every write; after ``recluster``. Risk: none.
- ``flush_inlined_data`` — writes inlined catalog rows out to Parquet. Planned
  maintenance; before reading files directly. Risk: none.
- ``set_partitioned_by`` / ``repartition`` — change the partition keys (future
  writes only, or with a rewrite). Filter strategy change on a very low
  cardinality column. Risk: ``repartition`` is a full rewrite.
- ``expire_snapshots(older_than_days)`` — makes snapshots older than the cutoff
  unreachable. **Planned maintenance only**, explicit retention (days). Risk:
  **destroys time travel** beyond the retention.
- ``cleanup_files`` — deletes files no snapshot references anymore. After
  ``expire_snapshots``; the only step that actually frees space (measured).
  Risk: irreversible.
- ``delete_orphaned_files`` — deletes files under ``data_path`` unknown to the
  catalog. After an incident (interrupted transaction, manual copy); always
  ``dry_run`` first. Risk: irreversible.

The write operations (``DatabaseUpdater``/``DatabaseDeleter``) only run the safe
post-write steps through :meth:`DuckLakeProcedures.compact` (merge + rewrite) and
never call ``expire_snapshots``/``cleanup_files``/``delete_orphaned_files``. The
diagnostic of the storage state and the policy deciding which step is worth running
live in :mod:`dt_ducklake_manager.maintenance.policy`.

Every method takes the target table first (``'fact_table'`` by default) and the
schema as a keyword-only argument (the instance's ``schema`` by default); the
catalog-wide procedures (snapshots, files of the data path) take neither.
"""

# Importation des modules
# Modules de base
import os
from datetime import datetime, timedelta
from typing import Any

# DuckDB
import duckdb

# Rapport d'opération
from ..reporting import OperationReport

# Module d'initialisation du logger
from ..utils.logger import _init_logger
from ..utils.sql import qualify_table, quote_ident, quote_literal

# Taille cible (octets) d'un fichier de données, alignée sur l'option recommandée
# ``target_file_size = '100MB'`` (RECOMMENDED_DUCKLAKE_OPTIONS du connecteur,
# stockée en octets par DuckLake). Sert de repli lorsque l'option n'est pas lisible
# dans ``ducklake_options`` (option non positionnée sur le catalogue) et de seuil
# par défaut des « petits fichiers » du diagnostic de stockage.
DEFAULT_TARGET_FILE_SIZE_BYTES: int = 100_000_000

# Part de lignes supprimées au-delà de laquelle un fichier est réécrit après une
# écriture (borne basse de la plage 0,1-0,3 recommandée par la spécification)
DEFAULT_DELETE_THRESHOLD: float = 0.1


# Fonction d'agrégation des compteurs renvoyés par une procédure de fichiers
def _sum_file_counters(rows: list[tuple[Any, ...]]) -> tuple[int, int]:
    """Sum the ``files_processed``/``files_created`` counters of a procedure result.

    ``ducklake_merge_adjacent_files`` and ``ducklake_rewrite_data_files`` return
    ``(schema_name, table_name, files_processed, files_created)`` rows: none when
    there is nothing to do, a row of zeros in some versions, possibly one row per
    table.

    Args:
        rows: Rows returned by the procedure.

    Returns:
        tuple[int, int]: Total ``(files_processed, files_created)``.

    Examples:
        >>> _sum_file_counters([])
        (0, 0)
        >>> _sum_file_counters([("main", "fact_table", 6, 1)])
        (6, 1)
    """
    return (
        sum(int(row[2] or 0) for row in rows),
        sum(int(row[3] or 0) for row in rows),
    )


# Classe des procédures de maintenance d'un catalogue DuckLake
class DuckLakeProcedures:
    """
    Thin, non-fatal wrappers around the DuckLake maintenance procedures.

    Each method runs one ``ducklake_*`` procedure (or one partitioning DDL), logs
    its outcome — a zero is always spelled out — and returns what DuckLake
    reported. Failures of the procedures are logged as warnings and turned into
    an empty result, so that a failure in one step never prevents the following
    ones from running; only the partitioning DDL raises, since it is an explicit
    schema change requested by the caller.

    Attributes:
        conn (duckdb.DuckDBPyConnection): DuckDB connection with the DuckLake
            catalog already attached.
        catalog_alias (str): Alias used in the ``ATTACH`` statement.
        schema (str): Default DuckLake schema of the table-level procedures.
        logger (logging.Logger): Logger instance.

    Examples:
        >>> from dt_ducklake_manager.connection import DuckLakeConnector
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
        >>> procedures = DuckLakeProcedures(conn)
        >>> procedures.compact()
        {'merge_files_processed': 0, 'merge_files_created': 0, ...}
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
        Initialize the procedures wrapper.

        Args:
            connection (duckdb.DuckDBPyConnection): DuckDB connection with the
                DuckLake catalog attached, created via ``DuckLakeConnector.connect()``
                or equivalent.
            catalog_alias (str): Alias used in the ``ATTACH`` statement. Must match
                the alias passed to ``DuckLakeConnector``. Defaults to ``'db'``.
            schema (str): Default DuckLake schema of the table-level procedures.
                Defaults to ``'main'``.
            log_filename (os.PathLike | None): Path to the log file. Defaults to
                ``<cwd>/logs/ducklake_maintenance.log``.

        Examples:
            >>> procedures = DuckLakeProcedures(conn, catalog_alias='my_lake')
            >>> procedures = DuckLakeProcedures(conn, schema='predictions')
        """
        # Stockage de la connexion, de l'alias du catalogue et du schéma par défaut
        self.conn = connection
        self.catalog_alias = catalog_alias
        self.schema = schema

        # Initialisation du logger nommé.
        # Chemin par défaut centralisé dans utils.logger : <cwd>/logs/<name>.log.
        self.logger = _init_logger(filename=log_filename, name="ducklake_maintenance")

    # ---------------------------------------------------------------------------
    # Procédures portant sur une table
    # ---------------------------------------------------------------------------

    # Fusion des fichiers Parquet adjacents
    def merge_files(
        self,
        table: str = "fact_table",
        *,
        schema: str | None = None,
        max_file_size: int | None = None,
        min_file_size: int | None = None,
        max_compacted_files: int | None = None,
    ) -> tuple[str, str, int, int]:
        """
        Merge adjacent Parquet files into files of at most ``max_file_size`` bytes.

        Each INSERT/UPDATE in DuckLake produces its own Parquet file, so a table
        updated many times ends up made of many small files, which forces DuckDB to
        open many file handles during a scan. ``ducklake_merge_adjacent_files``
        combines adjacent files as long as the merged file stays under
        ``max_file_size``.

        ``min_file_size`` is a **lower** bound (measured): files *smaller* than it
        are excluded from the merge. It is therefore left unset by default, so that
        the small files — the very ones worth merging — are all candidates.

        Args:
            table (str): Table to compact. Defaults to ``'fact_table'``.
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).
            max_file_size (int | None): Upper bound, in **bytes** (DuckLake rejects a
                value with a unit such as ``'1KB'``), on the size of a merged file.
                Defaults to None: the catalog's ``target_file_size``.
            min_file_size (int | None): Files smaller than this many bytes are left
                out of the merge. Defaults to None (no lower bound).
            max_compacted_files (int | None): Maximum number of files combined into
                one output file. Defaults to None (DuckLake's default).

        Returns:
            tuple[str, str, int, int]: ``(schema_name, table_name, files_processed,
            files_created)`` as reported by DuckLake, or ``(schema, table, 0, 0)``
            when nothing was merged or the call failed (the latter logged as a
            warning).

        Examples:
            >>> procedures.merge_files()
            ('main', 'fact_table', 6, 1)
            >>> procedures.merge_files('fact_table', max_file_size=50_000_000)
        """
        # Résolution du schéma et de la borne haute
        schema = schema or self.schema
        if max_file_size is None:
            max_file_size = self._target_file_size()

        # Paramètres nommés de la procédure : seuls ceux fournis sont transmis, un
        # paramètre absent laissant DuckLake appliquer son propre défaut
        named = [f"max_file_size := {int(max_file_size)}"]
        if min_file_size is not None:
            named.append(f"min_file_size := {int(min_file_size)}")
        if max_compacted_files is not None:
            named.append(f"max_compacted_files := {int(max_compacted_files)}")
        named.append(f"schema := {quote_literal(schema)}")

        try:
            # Exécution de la fusion des fichiers. Le résultat est lu en entier :
            # la procédure DuckLake n'est validée qu'une fois son résultat
            # consommé, une lecture partielle (fetchone) la laisse sans effet
            # (mesuré).
            rows = self.conn.execute(
                "SELECT * FROM ducklake_merge_adjacent_files("
                f"{quote_literal(self.catalog_alias)}, {quote_literal(table)},"
                f" {', '.join(named)})"
            ).fetchall()
        except Exception as e:
            # Logging
            self.logger.warning(f"merge_files failed for {schema}.{table}: {e}")
            return schema, table, 0, 0

        # Agrégation des compteurs (résultat vide : aucun fichier à fusionner)
        files_processed, files_created = _sum_file_counters(rows)
        # Logging : un zéro est toujours explicité
        if not files_processed:
            self.logger.debug(
                f"merge_files {schema}.{table}: 0 file merged (no adjacent files"
                f" to combine under max_file_size={max_file_size})"
            )
        else:
            self.logger.debug(
                f"merge_files {schema}.{table}: {files_processed} file(s) processed,"
                f" {files_created} file(s) created (max_file_size={max_file_size})"
            )
        return schema, table, files_processed, files_created

    # Réécriture des fichiers contenant des suppressions
    def rewrite_data_files(
        self,
        table: str = "fact_table",
        *,
        schema: str | None = None,
        delete_threshold: float = DEFAULT_DELETE_THRESHOLD,
    ) -> tuple[str, str, int, int]:
        """
        Rewrite data files to physically remove their deleted rows.

        DuckLake represents DELETE and UPDATE operations as separate delete-tombstone
        files, applied as a filter on every read. This procedure rewrites the files
        whose deleted-row share exceeds ``delete_threshold``.

        **Without an explicit ``delete_threshold`` the DuckLake procedure is a true
        no-op** (measured: an empty result set, even at 25% deletions) — this is why
        it is always passed here rather than left to the engine default.

        Args:
            table (str): Table to rewrite. Defaults to ``'fact_table'``.
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).
            delete_threshold (float): Rewrite files whose deleted-row share exceeds
                this fraction (0-1). Defaults to 0.1.

        Returns:
            tuple[str, str, int, int]: ``(schema_name, table_name, files_processed,
            files_created)`` as reported by DuckLake, or ``(schema, table, 0, 0)``
            when no file crosses the threshold or the call failed (the latter
            logged as a warning).

        Examples:
            >>> procedures.rewrite_data_files()
            ('main', 'fact_table', 0, 0)
            >>> procedures.rewrite_data_files('fact_table', delete_threshold=0.3)
        """
        # Résolution du schéma cible
        schema = schema or self.schema
        try:
            # Exécution de la réécriture des fichiers, résultat lu en entier pour
            # que la procédure soit validée
            rows = self.conn.execute(
                "SELECT * FROM ducklake_rewrite_data_files("
                f"{quote_literal(self.catalog_alias)}, {quote_literal(table)},"
                f" delete_threshold := {float(delete_threshold)},"
                f" schema := {quote_literal(schema)})"
            ).fetchall()
        except Exception as e:
            # Logging
            self.logger.warning(f"rewrite_data_files failed for {schema}.{table}: {e}")
            return schema, table, 0, 0

        # Agrégation des compteurs : DuckLake renvoie soit un résultat vide, soit
        # une ligne de zéros lorsqu'aucun fichier ne dépasse le seuil
        files_processed, files_created = _sum_file_counters(rows)

        # Logging : un libellé unique pour le zéro, quelle que soit sa forme
        if not files_processed:
            self.logger.debug(
                f"rewrite_data_files {schema}.{table}: 0 file rewritten (no file"
                f" above delete_threshold={delete_threshold})"
            )
        else:
            self.logger.debug(
                f"rewrite_data_files {schema}.{table}: {files_processed} file(s)"
                f" processed, {files_created} file(s) created"
                f" (delete_threshold={delete_threshold})"
            )
        return schema, table, files_processed, files_created

    # Écriture en Parquet des lignes inlinées dans le catalogue
    def flush_inlined_data(
        self, table: str | None = None, *, schema: str | None = None
    ) -> list[tuple[str, str, int]]:
        """
        Write inlined catalog rows out to Parquet files.

        Data inlining is active by default: a small ``INSERT`` produces no Parquet
        file at all, the rows living in the catalog instead. This procedure flushes
        them out to Parquet — required before reading the data path's files
        directly, and recommended in planned maintenance.

        Args:
            table (str | None): Table to flush. Defaults to None, flushing every
                table of the whole catalog.
            schema (str | None): Schema of ``table``. Defaults to None (this
                instance's ``schema``). Ignored when ``table`` is None.

        Returns:
            list[tuple[str, str, int]]: ``(schema_name, table_name, rows_flushed)``
            rows as reported by DuckLake — empty when there was nothing inlined, or
            when the call failed (logged as a warning).

        Examples:
            >>> procedures.flush_inlined_data()
            >>> procedures.flush_inlined_data('fact_table')
        """
        # Construction de la requête : une table précise, ou tout le catalogue
        if table is not None:
            query = (
                "SELECT * FROM ducklake_flush_inlined_data("
                f"{quote_literal(self.catalog_alias)},"
                f" table_name := {quote_literal(table)},"
                f" schema_name := {quote_literal(schema or self.schema)})"
            )
        else:
            query = (
                "SELECT * FROM ducklake_flush_inlined_data("
                f"{quote_literal(self.catalog_alias)})"
            )
        target = table or "all tables"

        # Exécution de la requête
        try:
            rows = self.conn.execute(query).fetchall()
        except Exception as e:
            # Logging
            self.logger.warning(f"flush_inlined_data failed for {target}: {e}")
            return []

        # Logging : un zéro est toujours explicité
        if not rows:
            self.logger.debug(
                f"flush_inlined_data ({target}): 0 row flushed (no inlined row)"
            )
        else:
            total_rows = sum(int(r[2]) for r in rows)
            self.logger.debug(
                f"flush_inlined_data ({target}): {total_rows} row(s) flushed to Parquet"
            )
        return rows

    # ---------------------------------------------------------------------------
    # Procédures portant sur tout le catalogue
    # ---------------------------------------------------------------------------

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
        **irreversible**, catalog-wide operation, unrelated to snapshot expiration.

        Args:
            older_than (datetime | str | None): Only consider files older than this
                cutoff. Defaults to None (no age filter).
            dry_run (bool): When True (the default), list the files that would be
                deleted without deleting them.

        Returns:
            list[str]: Paths deleted (or that would be deleted, under ``dry_run``),
            or an empty list when the call failed (logged as a warning).

        Examples:
            >>> procedures.delete_orphaned_files()  # dry_run=True: safe review
            >>> procedures.delete_orphaned_files(dry_run=False)  # actually deletes
        """
        # older_than n'est ajouté que s'il est fourni : le passer explicitement à
        # NULL provoque une erreur interne DuckDB.
        named = [f"dry_run := {str(dry_run).lower()}"]
        # Définition du seuil au delà duquel les snapshots sont périmés
        if older_than is not None:
            timestamp = (
                older_than
                if isinstance(older_than, str)
                else older_than.strftime("%Y-%m-%d %H:%M:%S")
            )
            named.append(f"older_than := TIMESTAMPTZ {quote_literal(timestamp)}")
        # Suppression des entrées
        try:
            rows = self.conn.execute(
                "SELECT * FROM ducklake_delete_orphaned_files("
                f"{quote_literal(self.catalog_alias)}, {', '.join(named)})"
            ).fetchall()
        except Exception as e:
            # Logging
            self.logger.warning(f"delete_orphaned_files failed: {e}")
            return []

        paths = [r[0] for r in rows]
        # Logging
        mode = "would be deleted (dry_run)" if dry_run else "deleted"
        self.logger.info(f"delete_orphaned_files: {len(paths)} file(s) {mode}")
        return paths

    # Expiration des anciens snapshots du catalogue
    def expire_snapshots(
        self, *, older_than_days: int = 30, dry_run: bool = False
    ) -> list[tuple[Any, ...]]:
        """
        Expire the snapshots older than a retention, catalog-wide.

        DuckLake retains every committed snapshot indefinitely by default, enabling
        time travel but consuming catalog space. Expired snapshots can no longer be
        queried via ``AT (VERSION => n)`` or ``AT (TIMESTAMP => t)``.

        **Reserved for planned maintenance with an explicit retention** — never
        called after a normal write, since it destroys time travel beyond the
        retention.

        Args:
            older_than_days (int): Snapshots older than this many days are expired.
                Defaults to 30.
            dry_run (bool): When True, list the snapshots that would be expired
                without expiring them. Defaults to False.

        Returns:
            list[tuple]: The expired (or, under ``dry_run``, would-be-expired)
            snapshot rows, or an empty list when the call failed (logged as a
            warning).

        Examples:
            >>> procedures.expire_snapshots(older_than_days=30)
            >>> procedures.expire_snapshots(older_than_days=7, dry_run=True)
        """
        # Calcul de l'horodatage de coupure à partir du nombre de jours
        cutoff = (datetime.now() - timedelta(days=older_than_days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        try:
            rows = self.conn.execute(
                "SELECT * FROM ducklake_expire_snapshots("
                f"{quote_literal(self.catalog_alias)},"
                f" older_than := TIMESTAMPTZ {quote_literal(cutoff)},"
                f" dry_run := {str(dry_run).lower()})"
            ).fetchall()
        except Exception as e:
            # Logging
            self.logger.warning(f"expire_snapshots failed: {e}")
            return []

        # Logging
        mode = "would be expired (dry_run)" if dry_run else "expired"
        self.logger.info(
            f"expire_snapshots (cutoff {cutoff}): {len(rows)} snapshot(s) {mode}"
        )
        return rows

    # Nettoyage des fichiers Parquet que plus aucun snapshot ne référence
    def cleanup_files(self, *, dry_run: bool = False) -> list[str]:
        """
        Remove the files no longer referenced by any live snapshot, catalog-wide.

        After expiring snapshots, the Parquet files they referenced remain on disk
        until this procedure is called — the only step that actually frees disk
        space (measured). **Reserved for planned maintenance**, after
        ``expire_snapshots``.

        Args:
            dry_run (bool): When True, list the files that would be deleted without
                deleting them. Defaults to False.

        Returns:
            list[str]: Paths deleted (or that would be deleted, under ``dry_run``),
            or an empty list when the call failed (logged as a warning).

        Examples:
            >>> procedures.cleanup_files()
            >>> procedures.cleanup_files(dry_run=True)
        """
        try:
            # Exécution de la requête de suppression des fichiers expirés
            rows = self.conn.execute(
                "SELECT * FROM ducklake_cleanup_old_files("
                f"{quote_literal(self.catalog_alias)},"
                f" dry_run := {str(dry_run).lower()})"
            ).fetchall()
        except Exception as e:
            # Logging
            self.logger.warning(f"cleanup_files failed: {e}")
            return []

        paths = [r[0] for r in rows]
        # Logging
        mode = "would be deleted (dry_run)" if dry_run else "deleted"
        self.logger.info(f"cleanup_files: {len(paths)} file(s) {mode}")
        return paths

    # ---------------------------------------------------------------------------
    # Gestion du partitionnement
    # ---------------------------------------------------------------------------

    # Définition ou remplacement du partitionnement d'une table
    def set_partitioned_by(
        self,
        table: str = "fact_table",
        *,
        partition_by: list[str],
        schema: str | None = None,
    ) -> None:
        """
        Set or replace the partition keys of an existing DuckLake table.

        Executes ``ALTER TABLE … SET PARTITIONED BY``. Only data written *after*
        this call follows the new layout; existing files are unaffected (see
        :meth:`repartition` to also rewrite them). Partitioning complements the
        physical sort on ``cluster_by`` and is reserved for columns of very low
        cardinality, filtered on by most queries.

        Supported partition expressions (plain strings, used as is):

        - column name — ``'country'``
        - time transforms — ``'year(ts)'``, ``'month(ts)'``, ``'day(ts)'``,
          ``'hour(ts)'``
        - hash distribution — ``'bucket(8, user_id)'``

        Args:
            table (str): Table to partition. Defaults to ``'fact_table'``.
            partition_by (list[str]): Non-empty list of partition expressions.
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).

        Raises:
            ValueError: If ``partition_by`` is empty.
            duckdb.Error: If DuckLake rejects the partition expressions.

        Examples:
            >>> procedures.set_partitioned_by(partition_by=['country'])
            >>> procedures.set_partitioned_by(partition_by=['year(date_col)'])
        """
        # Validation : au moins une clé de partitionnement est requise
        if not partition_by:
            raise ValueError(
                "partition_by must not be empty; use reset_partitioned_by() to"
                " remove the partitioning"
            )

        # Construction et exécution du DDL sur la table qualifiée par le catalogue
        schema = schema or self.schema
        keys = ", ".join(partition_by)
        self.conn.execute(
            f"ALTER TABLE {qualify_table(table, schema, self.catalog_alias)}"
            f" SET PARTITIONED BY ({keys})"
        )

        # Logging
        self.logger.info(f"Partitioning of {schema}.{table} set to ({keys})")

    # Suppression du partitionnement d'une table
    def reset_partitioned_by(
        self, table: str = "fact_table", *, schema: str | None = None
    ) -> None:
        """
        Remove all partition keys from an existing DuckLake table.

        Executes ``ALTER TABLE … RESET PARTITIONED BY``. Subsequent writes produce
        unpartitioned files; existing files are unaffected.

        Args:
            table (str): Table to unpartition. Defaults to ``'fact_table'``.
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).

        Raises:
            duckdb.Error: If the table does not exist.

        Examples:
            >>> procedures.reset_partitioned_by()
        """
        # Suppression de la définition de partitionnement
        schema = schema or self.schema
        self.conn.execute(
            f"ALTER TABLE {qualify_table(table, schema, self.catalog_alias)}"
            " RESET PARTITIONED BY"
        )

        # Logging
        self.logger.info(f"Partitioning of {schema}.{table} removed")

    # Réinitialisation et application d'un nouveau partitionnement
    def repartition(
        self,
        table: str = "fact_table",
        *,
        partition_by: list[str] | None = None,
        schema: str | None = None,
        run_maintenance: bool = True,
    ) -> None:
        """
        Reset the current partitioning and optionally apply a new one.

        Orchestrates three steps in order:

        1. :meth:`reset_partitioned_by` — clears the existing partition definition.
        2. :meth:`set_partitioned_by` — applies the new keys (skipped when
           ``partition_by`` is None).
        3. :meth:`merge_files` + :meth:`rewrite_data_files` — rewrite the existing
           Parquet files (only when ``run_maintenance`` is True).

        Args:
            table (str): Table to repartition. Defaults to ``'fact_table'``.
            partition_by (list[str] | None): New partition keys. None removes the
                partitioning without defining a replacement.
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).
            run_maintenance (bool): Whether to merge and rewrite the existing files
                after changing the partition definition. Defaults to True.

        Raises:
            ValueError: If ``partition_by`` is an empty list.
            duckdb.Error: If the table does not exist or DuckLake rejects the
                partition expressions.

        Examples:
            >>> procedures.repartition(partition_by=['year(date_col)', 'country'])
            >>> procedures.repartition(partition_by=None)  # removes partitioning
            >>> procedures.repartition(partition_by=['country'], run_maintenance=False)
        """
        # Résolution du schéma cible
        schema = schema or self.schema
        self.logger.info(
            f"repartition {schema}.{table}: new partition keys {partition_by}"
        )

        # Étape 1 : suppression du partitionnement courant
        self.reset_partitioned_by(table, schema=schema)

        # Étape 2 : application du nouveau partitionnement, s'il est fourni
        if partition_by is not None:
            self.set_partitioned_by(table, partition_by=partition_by, schema=schema)

        # Étape 3 : réécriture des fichiers existants, afin que les données déjà
        # présentes adoptent elles aussi le nouveau découpage
        if run_maintenance:
            self.merge_files(table, schema=schema)
            self.rewrite_data_files(table, schema=schema)

    # ---------------------------------------------------------------------------
    # Compaction post-écriture
    # ---------------------------------------------------------------------------

    # Compaction légère après une écriture (fusion puis réécriture)
    def compact(
        self,
        table: str = "fact_table",
        *,
        schema: str | None = None,
        delete_threshold: float = DEFAULT_DELETE_THRESHOLD,
        report: OperationReport | None = None,
    ) -> dict[str, int]:
        """
        Run the lightweight post-write compaction: merge then rewrite.

        Merges adjacent small Parquet files up to the catalog's ``target_file_size``
        and rewrites the files whose deleted-row share exceeds ``delete_threshold``.
        Called by the write operations right after their commit. Never raises:
        :meth:`merge_files` and :meth:`rewrite_data_files` log and swallow their own
        failures.

        Never calls ``expire_snapshots``, ``cleanup_files`` or
        ``delete_orphaned_files``: those destroy time travel or are irreversible and
        belong to planned maintenance (``DuckLakeMaintenance.maintain`` with an
        explicit ``MaintenancePolicy``).

        Args:
            table (str): Table to compact. Defaults to ``'fact_table'``.
            schema (str | None): DuckLake schema. Defaults to None (this instance's
                ``schema``).
            delete_threshold (float): Rewrite files whose deleted-row share exceeds
                this fraction (0-1). Defaults to 0.1.
            report (OperationReport | None): When given, its ``maintenance`` dict
                receives the four counters (zeros included).

        Returns:
            dict[str, int]: ``merge_files_processed``, ``merge_files_created``,
            ``rewrite_files_processed`` and ``rewrite_files_created``.

        Examples:
            >>> procedures.compact()
            >>> procedures.compact('fact_table', schema='predictions',
            ...     delete_threshold=0.05)
        """
        # Résolution du schéma cible
        schema = schema or self.schema
        # Fusion des petits fichiers puis réécriture des fichiers trop supprimés
        _, _, merge_processed, merge_created = self.merge_files(table, schema=schema)
        _, _, rewrite_processed, rewrite_created = self.rewrite_data_files(
            table, schema=schema, delete_threshold=delete_threshold
        )
        counters = {
            "merge_files_processed": merge_processed,
            "merge_files_created": merge_created,
            "rewrite_files_processed": rewrite_processed,
            "rewrite_files_created": rewrite_created,
        }
        # Logging
        self.logger.debug(
            f"compact {schema}.{table}: merge {merge_processed} -> {merge_created}"
            f" file(s), rewrite {rewrite_processed} -> {rewrite_created} file(s)"
            f" (delete_threshold={delete_threshold})"
        )
        # Ajout au rapport
        if report is not None:
            report.maintenance.update(counters)
        return counters

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

    # Taille cible des fichiers du catalogue
    def _target_file_size(self) -> int:
        """Read the catalog's ``target_file_size`` option, in bytes.

        Returns:
            int: The option value (stored in bytes by DuckLake), or
            ``DEFAULT_TARGET_FILE_SIZE_BYTES`` when the option is not set on the
            catalog or cannot be read.
        """
        try:
            row = self.conn.execute(
                "SELECT value FROM ducklake_options(?)"
                " WHERE option_name = 'target_file_size'",
                [self.catalog_alias],
            ).fetchone()
        except Exception as e:
            self.logger.debug(f"target_file_size unreadable: {e}")
            return DEFAULT_TARGET_FILE_SIZE_BYTES
        if row is None or row[0] is None:
            return DEFAULT_TARGET_FILE_SIZE_BYTES
        return int(row[0])

    # Comptage des lignes d'une table
    def _count_rows(self, table: str, schema: str) -> int:
        """Count the rows of a table of the attached catalog.

        Args:
            table: Bare table name.
            schema: DuckLake schema name.

        Returns:
            int: Row count, or ``0`` if the table cannot be read.
        """
        qualified = qualify_table(table, schema, self.catalog_alias)
        try:
            row = self.conn.execute(f"SELECT COUNT(*) FROM {qualified}").fetchone()
        except Exception:
            return 0
        return int(row[0]) if row is not None else 0
