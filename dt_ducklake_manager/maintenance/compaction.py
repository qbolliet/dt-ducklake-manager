# Importation des modules
# Modules de base
import os
from datetime import datetime, timedelta
from pathlib import Path

# DuckDB
import duckdb

# Module d'initialisation du logger
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe de maintenance d'un catalogue DuckLake
class DuckLakeMaintenance:
    """
    Runs maintenance operations on a DuckLake catalog to keep read performance optimal.

    DuckLake writes small Parquet delta files and delete tombstone files for every
    UPDATE/DELETE/MERGE operation. Over time, many small files accumulate and degrade
    sequential read performance. This class wraps the four DuckLake maintenance
    procedures that compact those files back into larger, efficient Parquet files.

    All methods are non-fatal: errors are logged as warnings and execution continues,
    so a failure in one step does not prevent the remaining steps from running.

    Attributes:
        conn (duckdb.DuckDBPyConnection): DuckDB connection with the DuckLake catalog
            already attached.
        catalog_alias (str): Alias used in the ``ATTACH`` statement (default ``'db'``).
        logger (logging.Logger): Logger instance.

    Examples:
        >>> from dt_ducklake_manager.connection import DuckLakeConnector
        >>> from dt_ducklake_manager.maintenance import DuckLakeMaintenance
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
        >>> maint = DuckLakeMaintenance(conn)
        >>> maint.full_maintenance('main', 'fact_table')
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        catalog_alias: str = "db",
        log_filename: str | os.PathLike[str] | None = os.path.join(
            FILE_PATH.parents[2], "logs/ducklake_maintenance.log"
        ),
    ) -> None:
        """
        Initialize the DuckLakeMaintenance manager.

        Args:
            connection (duckdb.DuckDBPyConnection): DuckDB connection with the
                DuckLake catalog attached. The connection must have been created
                via ``DuckLakeConnector.connect()`` or equivalent.
            catalog_alias (str): Alias used in the ``ATTACH`` statement. Must match
                the alias passed to ``DuckLakeConnector``. Defaults to ``'db'``.
            log_filename (Optional[os.PathLike]): Path to the log file.

        Examples:
            >>> maint = DuckLakeMaintenance(conn)
            >>> maint = DuckLakeMaintenance(conn, catalog_alias='my_lake')
        """
        # Stockage de la connexion et de l'alias du catalogue
        self.conn = connection
        self.catalog_alias = catalog_alias

        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(
                FILE_PATH.parents[2], "logs/ducklake_maintenance.log"
            )
        self.logger = _init_logger(filename=log_filename)

    # ---------------------------------------------------------------------------
    # Méthodes de maintenance individuelles
    # ---------------------------------------------------------------------------

    # Fusion des petits fichiers Parquet adjacents
    def merge_files(self, schema: str, table: str) -> None:
        """
        Merge small adjacent Parquet files into larger files.

        Each INSERT/UPDATE/MERGE in DuckLake produces a small Parquet file.
        Over time, a table may consist of hundreds of tiny files, which forces
        DuckDB to open many file handles during a sequential scan. This procedure
        merges adjacent files into larger chunks, reducing scan overhead.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).
            table (str): Table name to compact (e.g. ``'fact_table'``).

        Examples:
            >>> maint.merge_files('main', 'fact_table')
        """
        try:
            # Exécution de la fusion des fichiers
            # Note : les table functions DuckLake vivent dans le catalogue mémoire.
            self.conn.execute(
                f"CALL ducklake_merge_adjacent_files('{self.catalog_alias}', '{table}',"
                f" schema := '{schema}')"
            )
            # Logging
            self.logger.info(
                f"The merge of the Parquet files is finished : {schema}.{table}"
            )
        except Exception as e:
            # Logging
            self.logger.warning(f"merge_files failed for {schema}.{table} : {e}")

    # Réécriture des fichiers contenant des suppressions
    def rewrite_data_files(self, schema: str, table: str) -> None:
        """
        Rewrite data files to remove deleted rows from Parquet files.

        DuckLake represents DELETE and UPDATE operations as separate delete-tombstone
        files. These tombstones accumulate and must be applied as a filter on every
        read. This procedure rewrites the underlying Parquet files to physically
        remove deleted rows, eliminating the tombstone overhead.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).
            table (str): Table name to rewrite (e.g. ``'fact_table'``).

        Examples:
            >>> maint.rewrite_data_files('main', 'fact_table')
        """
        try:
            # Exécution de la réécriture des fichiers
            self.conn.execute(
                f"CALL ducklake_rewrite_data_files('{self.catalog_alias}', '{table}',"
                f"schema := '{schema}')"
            )
            # Logging
            self.logger.info(
                f"The rewriting of the file deletion is finished : {schema}.{table}"
            )
        except Exception as e:
            # Loggin
            self.logger.warning(f"rewrite_data_files failed for {schema}.{table} : {e}")

    # Expiration des anciens snapshots du catalogue
    def expire_snapshots(self, schema: str, older_than_days: int = 30) -> None:
        """
        Expire old snapshots to free catalog space.

        DuckLake retains every committed snapshot indefinitely by default, enabling
        time travel but consuming catalog space. This procedure marks snapshots older
        than ``older_than_days`` as expired. Expired snapshots can no longer be
        queried via ``AT (VERSION => n)`` or ``AT (TIMESTAMP => t)``.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).
            older_than_days (int): Snapshots older than this many days will be expired.
                Defaults to 30.

        Examples:
            >>> maint.expire_snapshots('main', older_than_days=30)
            >>> maint.expire_snapshots('main', older_than_days=7)
        """
        # Calcul du timestamp de coupure à partir du nombre de jours
        cutoff: datetime = datetime.now() - timedelta(days=older_than_days)
        cutoff_str: str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        try:
            # Exécution de la requête
            self.conn.execute(
                f"CALL ducklake_expire_snapshots('{self.catalog_alias}', "
                f"older_than := TIMESTAMPTZ '{cutoff_str}')"
            )
            # Logging
            self.logger.info(
                f"Snapshots before the cutoff date {cutoff_str} expiration is finished"
                f": {schema}"
            )
        except Exception as e:
            # Logging
            self.logger.warning(f"expire_snapshots failed for {schema} : {e}")

    # Nettoyage des fichiers Parquet orphelins
    def cleanup_files(self, schema: str) -> None:
        """
        Remove orphaned Parquet files no longer referenced by any snapshot.

        After expiring snapshots, the Parquet data files they referenced remain on
        disk until this procedure is called. It scans the catalog and deletes any
        data files that are not referenced by a live snapshot.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).

        Examples:
            >>> maint.cleanup_files('main')
        """
        try:
            # Exécution de la suppression des fichiers orphelins
            self.conn.execute(
                f"CALL ducklake_cleanup_old_files('{self.catalog_alias}')"
            )
            # Logging
            self.logger.info(
                f"the cleaning of the orphaned files is finished : {schema}"
            )
        except Exception as e:
            # Logging
            self.logger.warning(f"cleanup_files failed for {schema} : {e}")

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
        self.conn.execute(
            f"ALTER TABLE {schema}.{table} SET PARTITIONED BY ({cols})"
        )

        # Logging
        self.logger.info(
            f"Partitionning defined on {schema}.{table} : ({cols})"
        )

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
        self.logger.info(
            f"Partitionning removed on {schema}.{table}"
        )

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
        self.logger.info(
            f"The reparttition of {schema}.{table} is completed"
        )

    # ---------------------------------------------------------------------------
    # Méthode de maintenance complète
    # ---------------------------------------------------------------------------

    # Exécution de l'ensemble des opérations de maintenance dans l'ordre recommandé
    def full_maintenance(
        self,
        schema: str,
        table: str,
        older_than_days: int = 30,
    ) -> None:
        """
        Run all maintenance operations in the recommended order.

        Executes in sequence: ``merge_files`` → ``rewrite_data_files`` →
        ``expire_snapshots`` → ``cleanup_files``. Each step is wrapped in a
        ``try/except`` so a failure in one step does not block the others.

        Args:
            schema (str): DuckLake schema name (e.g. ``'main'``).
            table (str): Table name to compact (passed to ``merge_files`` and
                ``rewrite_data_files``).
            older_than_days (int): Passed to ``expire_snapshots``. Defaults to 30.

        Examples:
            >>> maint.full_maintenance('main', 'fact_table')
            >>> maint.full_maintenance('main', 'fact_table', older_than_days=7)
        """
        # Logging
        self.logger.info(
            f"Beginning the full DuckLake maintenance : schema={schema}, table={table}"
        )

        # Chaque étape est enveloppée dans un try/except pour garantir que
        # l'échec d'une étape ne bloque pas les étapes suivantes.

        # Étape 1 : fusion des petits fichiers Parquet adjacents
        try:
            self.merge_files(schema, table)
        except Exception as e:
            self.logger.warning(f"full_maintenance — merge_files failed : {e}")

        # Étape 2 : réécriture des fichiers contenant des suppressions
        try:
            self.rewrite_data_files(schema, table)
        except Exception as e:
            self.logger.warning(f"full_maintenance — rewrite_data_files failed : {e}")

        # Étape 3 : expiration des anciens snapshots
        try:
            self.expire_snapshots(schema, older_than_days=older_than_days)
        except Exception as e:
            self.logger.warning(f"full_maintenance — expire_snapshots failed : {e}")

        # Étape 4 : suppression des fichiers Parquet orphelins
        try:
            self.cleanup_files(schema)
        except Exception as e:
            self.logger.warning(f"full_maintenance — cleanup_files failed : {e}")

        # Logging
        self.logger.info(
            f"Maintenance of the DuckLake is finished : schema={schema}, table={table}"
        )
