# Importation des modules
# Modules de base
import os
from pathlib import Path
from typing import List, Optional
# DuckDB
import duckdb

# Module d'initialisation du logger
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe d'exportation d'un schéma DuckDB vers des fichiers Parquet
class ParquetExporter:
    """
    Exports DuckDB tables to Parquet files, with optional partitioning on the fact table.

    This class  operates on an existing
    DuckDB connection and is independent of ``DuckdbTablesBuilder``, so it can be
    used to export any DuckDB database without re-building the schema.

    DuckDB's ``COPY TO`` statement natively supports ``s3://`` paths, making this
    class S3-compatible without additional code.

    Only the fact table is partitioned. Metadata and dimension tables are always
    exported as flat Parquet files because they are small and partitioning would
    only add complexity without any performance benefit.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Active DuckDB connection.
        logger (logging.Logger): Logger instance for tracking operations.

    Examples:
        >>> import duckdb
        >>> conn = duckdb.connect(':memory:')
        >>> conn.execute("CREATE TABLE fact_table AS SELECT 1 AS id, 'FR' AS country, 42.0 AS value")
        >>> exporter = ParquetExporter(connection=conn)
        >>> exporter.export_fact_table('fact_table', '/tmp/output/fact.parquet')
    """

    # Initialisation
    def __init__(
        self,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
        path: Optional[os.PathLike] = None,
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[2], "logs/parquet_exporter.log"
        ),
    ) -> None:
        """
        Initialize the ParquetExporter.

        Args:
            connection (Optional[duckdb.DuckDBPyConnection]): Existing DuckDB connection.
                If None and path is also None, an in-memory connection is created.
            path (Optional[os.PathLike]): Path to a DuckDB database file. Used only
                when connection is None.
            log_filename (Optional[os.PathLike]): Path to the log file.

        Examples:
            >>> exporter = ParquetExporter()                          # in-memory
            >>> exporter = ParquetExporter(path='my_db.duckdb')      # file-based
            >>> exporter = ParquetExporter(connection=existing_conn)  # reuse connection
        """
        # Initialisation de la connexion DuckDB
        if (connection is None) and (path is None):
            self.conn = duckdb.connect(':memory:')
        elif connection is None:
            self.conn = duckdb.connect(path)
        else:
            self.conn = connection

        # Initialisation du logger
        self.logger = _init_logger(filename=log_filename)

    # Méthode d'exportation d'une table vers un fichier Parquet
    def export_table(
        self,
        table_name: str,
        output_path: str,
        partition_keys: Optional[List[str]] = None,
    ) -> None:
        """
        Export a single DuckDB table to one or more Parquet files.

        Without ``partition_keys``, the entire table is written to a single
        ``output_path`` file.  With ``partition_keys``, DuckDB creates a
        directory tree under ``output_path`` following the Hive partitioning
        convention: ``output_path/col=val/data_0.parquet``.

        Args:
            table_name (str): Name of the DuckDB table to export.
            output_path (str): Destination path. For partitioned exports this is
                the root directory; for flat exports it must end with ``.parquet``.
                Accepts local paths or ``s3://`` URIs.
            partition_keys (Optional[List[str]]): Column names to partition by.
                Each key must exist in ``table_name``. Defaults to None (no
                partitioning).

        Raises:
            ValueError: If ``table_name`` does not exist in the database.
            ValueError: If any element of ``partition_keys`` is not a column of
                ``table_name``.
            duckdb.Error: If the DuckDB ``COPY TO`` statement fails.

        Examples:
            >>> exporter.export_fact_table('fact_table', '/tmp/fact.parquet')
            >>> exporter.export_fact_table(
            ...     'fact_table', '/tmp/fact_partitioned/',
            ...     partition_keys=['country']
            ... )
        """
        try:
            # Validation de l'existence de la table dans la base de données
            existing_tables = [
                row[0] for row in self.conn.execute("SHOW TABLES").fetchall()
            ]
            if table_name not in existing_tables:
                raise ValueError(
                    f"The table '{table_name}' does not exist in the database. "
                    f"Available tables : {existing_tables}"
                )

            # Validation des clés de partitionnement si fournies
            if partition_keys:
                # Récupération des colonnes de la table via DESCRIBE
                existing_columns = [
                    row[0]
                    for row in self.conn.execute(f"DESCRIBE {table_name}").fetchall()
                ]
                # Extraction des clés qui ne sont pas dans les colonnes
                invalid_keys = [k for k in partition_keys if k not in existing_columns]
                if invalid_keys:
                    raise ValueError(
                        f"The partition keys are not in the columns "
                        f"of the table '{table_name}' : {invalid_keys}. "
                        f"Available columns : {existing_columns}"
                    )

            # Construction et exécution de la requête COPY TO
            if partition_keys:
                # Export avec partitionnement Hive (crée automatiquement les sous-répertoires)
                keys_str = ", ".join(partition_keys)
                query = (
                    f"COPY {table_name} TO '{output_path}' "
                    f"(FORMAT PARQUET, PARTITION_BY ({keys_str}))"
                )
                self.logger.info(
                    f"Export of '{table_name}' to '{output_path}' "
                    f"with partition on ({keys_str})"
                )
            else:
                # Export à plat vers un fichier unique
                query = f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET)"
                self.logger.info(
                    f"Export of '{table_name}' to '{output_path}' (without partition)"
                )
            # Exécution de la requête
            self.conn.execute(query)
            # Logging
            self.logger.info(f"Export de '{table_name}' terminé avec succès")

        except (ValueError, duckdb.Error):
            # Propagation des erreurs de validation et DuckDB sans les masquer
            raise
        except Exception as e:
            # Logging
            self.logger.error(f"Échec de l'export de '{table_name}' : {e}")
            raise

    # Méthode d'exportation complète du schéma (fact table + metadata + dimensions)
    def export_schema(
        self,
        fact_table: str,
        output_dir: str,
        partition_keys: Optional[List[str]] = None,
        include_metadata: bool = True,
        include_dimensions: bool = True,
        metadata_table: str = 'metadata',
        dim_table_prefix: str = 'dim_',
    ) -> None:
        """
        Export the full database schema (fact table, metadata, dimension tables) to Parquet.

        The fact table is exported with optional partitioning. The metadata and
        dimension tables are always exported flat (one file each) because they are
        small and partitioning would not be beneficial.

        Output layout::

            output_dir/
            ├── fact_table/           # partitioned if partition_keys provided
            │   └── country=FR/
            │       └── data_0.parquet
            ├── metadata.parquet      # flat (include_metadata=True)
            └── dim_country.parquet   # flat (include_dimensions=True)

        Args:
            fact_table (str): Name of the fact table in DuckDB.
            output_dir (str): Root output directory. Accepts local paths or
                ``s3://`` URIs.
            partition_keys (Optional[List[str]]): Columns to partition the fact
                table by. Defaults to None (no partitioning).
            include_metadata (bool): Whether to export the metadata table.
                Defaults to True.
            include_dimensions (bool): Whether to export dimension tables.
                Defaults to True.
            metadata_table (str): Name of the metadata table. Defaults to
                ``'metadata'``.
            dim_table_prefix (str): Prefix used to identify dimension tables.
                Defaults to ``'dim_'``.

        Raises:
            ValueError: If ``fact_table`` does not exist in the database.
            duckdb.Error: If any ``COPY TO`` statement fails.

        Examples:
            >>> exporter.export_schema('fact_table', '/tmp/export/')
            >>> exporter.export_schema(
            ...     'fact_table', '/tmp/export/',
            ...     partition_keys=['country'],
            ...     include_metadata=True,
            ...     include_dimensions=True,
            ... )
        """
        # Normalisation du chemin de sortie (suppression du slash final pour cohérence)
        output_dir = output_dir.rstrip('/').rstrip('\\')

        # Export de la table des faits (avec partitionnement optionnel)
        fact_output = f"{output_dir}/{fact_table}"
        self.export_table(fact_table, fact_output, partition_keys=partition_keys)

        # Export de la table de métadonnées à plat
        if include_metadata:
            existing_tables = [
                row[0] for row in self.conn.execute("SHOW TABLES").fetchall()
            ]
            if metadata_table in existing_tables:
                # Nom de la table
                metadata_output = f"{output_dir}/{metadata_table}.parquet"
                # Export de la table des méta-données
                self.export_table(metadata_table, metadata_output, partition_keys=None)
            else:
                # Logging
                self.logger.warning(
                    f"Metadata table cannot be found '{metadata_table}'. The export of this table is ignored"
                )

        # Export des tables de dimension à plat
        if include_dimensions:
            # Identification des tables de dimension par leur préfixe
            all_tables = [
                row[0] for row in self.conn.execute("SHOW TABLES").fetchall()
            ]
            dim_tables = [t for t in all_tables if t.startswith(dim_table_prefix)]

            # Cas où aucune table de dimensions est trouvée
            if not dim_tables:
                self.logger.warning(
                    f"No dimension table found with the prefix '{dim_table_prefix}'"
                )

            # Exportation des tables de dimension
            for dim_table in dim_tables:
                # Chemin d'export
                dim_output = f"{output_dir}/{dim_table}.parquet"
                # Export sans partition
                self.export_table(dim_table, dim_output, partition_keys=None)
