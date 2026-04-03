# Importation des modules
# Modules de base
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

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
        >>> from dashboard_template_database.builders import DuckLakeConnector
        >>> from dashboard_template_database.operations import DuckLakeMaintenance
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
        >>> maint = DuckLakeMaintenance(conn)
        >>> maint.full_maintenance('main', 'fact_table')
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        catalog_alias: str = 'db',
        log_filename: Optional[os.PathLike] = os.path.join(
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
            self.conn.execute(
                f"CALL {self.catalog_alias}.ducklake_merge_adjacent_files('{schema}', '{table}')"
            )
            # Logging
            self.logger.info(f"The merge of the Parquet files is finished : {schema}.{table}")
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
                f"CALL {self.catalog_alias}.ducklake_rewrite_data_files('{schema}', '{table}')"
            )
            # Logging
            self.logger.info(f"The rewriting of the file deletion is finished : {schema}.{table}")
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
        cutoff_str: str = cutoff.strftime('%Y-%m-%d %H:%M:%S')
        try:
            # Exécution de la requête
            self.conn.execute(
                f"CALL {self.catalog_alias}.ducklake_expire_snapshots("
                f"'{schema}', TIMESTAMP '{cutoff_str}')"
            )
            # Logging
            self.logger.info(
                f"Snapshots before the cutoff date {cutoff_str} expiration is finished : {schema}"
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
                f"CALL {self.catalog_alias}.ducklake_cleanup_old_files('{schema}')"
            )
            # Logging
            self.logger.info(f"the cleaning of the orphaned files is finished : {schema}")
        except Exception as e:
            # Logging
            self.logger.warning(f"cleanup_files failed for {schema} : {e}")

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
