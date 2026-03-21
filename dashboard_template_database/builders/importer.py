# Importation des modules
# Modules de base
import os
import glob as glob_module
from pathlib import Path
from typing import List, Optional, Tuple

# DuckDB
import duckdb

# Module d'initialisation du logger
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe de chargement d'un export Parquet vers un schéma DuckDB
class ParquetImporter:
    """
    Reconstructs a DuckDB schema from a Parquet export directory.

    This class is the symmetric counterpart of ``ParquetExporter``: it reads
    the directory layout produced by ``ParquetExporter.export_schema()`` and
    recreates the corresponding DuckDB tables (metadata, dimension tables, and
    fact table) in an existing or new DuckDB connection.

    All Parquet reading is performed through DuckDB's native ``read_parquet()``
    table function, which is vectorised and zero-copy for local files, and
    natively supports ``s3://`` URIs via the ``httpfs`` extension — no
    pandas round-trip is needed.

    After calling ``load_schema()``, the exposed ``conn`` attribute can be
    passed directly to ``DatabaseUpdaterV2`` for in-place updates, followed
    by a new ``ParquetExporter.export_schema()`` call to persist the changes.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Active DuckDB connection holding the
            reconstructed schema.
        input_dir (str): Root directory of the Parquet export (local or S3).
        logger (logging.Logger): Logger instance for tracking operations.

    Examples:
        >>> importer = ParquetImporter('/tmp/export/')
        >>> importer.load_schema()
        >>> # La connexion est prête pour DatabaseUpdaterV2
        >>> from dashboard_template_database.operations import DatabaseUpdaterV2
        >>> updater = DatabaseUpdaterV2(connection=importer.conn)
    """

    # Initialisation
    def __init__(
        self,
        input_dir: str,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
        path: Optional[os.PathLike] = None,
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[2], "logs/parquet_importer.log"
        ),
    ) -> None:
        """
        Initialize the ParquetImporter.

        Args:
            input_dir (str): Root directory of the Parquet export produced by
                ``ParquetExporter.export_schema()``. Accepts local paths or
                ``s3://`` URIs.
            connection (Optional[duckdb.DuckDBPyConnection]): Existing DuckDB
                connection to load tables into. If None and path is also None,
                an in-memory connection is created.
            path (Optional[os.PathLike]): Path to a DuckDB database file. Used
                only when connection is None.
            log_filename (Optional[os.PathLike]): Path to the log file.

        Examples:
            >>> importer = ParquetImporter('/tmp/export/')           # in-memory
            >>> importer = ParquetImporter('/tmp/export/', path='restored.duckdb')
            >>> importer = ParquetImporter('/tmp/export/', connection=existing_conn)
        """
        # Stockage du répertoire d'entrée (suppression du slash final pour cohérence)
        self.input_dir = str(input_dir).rstrip('/').rstrip('\\')

        # Initialisation de la connexion DuckDB
        if (connection is None) and (path is None):
            self.conn = duckdb.connect(':memory:')
        elif connection is None:
            self.conn = duckdb.connect(path)
        else:
            self.conn = connection

        # Initialisation du logger
        self.logger = _init_logger(filename=log_filename)

    # ---------------------------------------------------------------------------
    # Méthodes privées de détection et validation
    # ---------------------------------------------------------------------------

    # Vérification que le chemin est un URI S3
    def _is_s3_path(self, path: str) -> bool:
        """Return True if path is an S3 URI (starts with 's3://')."""
        return str(path).startswith('s3://')

    # Validation de l'accessibilité du répertoire d'entrée
    def _validate_input_dir(self) -> None:
        """
        Validate that input_dir is accessible.

        For S3 paths, loads the httpfs extension so that subsequent
        read_parquet() calls can resolve the URI. For local paths,
        checks that the directory exists.

        Raises:
            FileNotFoundError: If input_dir does not exist (local paths only).
        """
        if self._is_s3_path(self.input_dir):
            # Chargement de l'extension httpfs nécessaire aux accès S3
            self.conn.execute("INSTALL httpfs; LOAD httpfs;")
            self.logger.info("Extension httpfs chargée pour accès S3")
        else:
            if not os.path.isdir(self.input_dir):
                raise FileNotFoundError(
                    f"Le répertoire d'entrée '{self.input_dir}' n'existe pas."
                )

    # Détection du chemin et du mode de partitionnement de la fact table
    def _detect_fact_table_path(self, fact_table: str) -> Tuple[str, bool]:
        """
        Detect the location and partition status of the fact table.

        ``ParquetExporter.export_schema()`` writes the fact table either as:
        - A directory ``<input_dir>/<fact_table>/`` (Hive-partitioned).
        - A flat file ``<input_dir>/<fact_table>`` without a ``.parquet``
          extension (produced by DuckDB's ``COPY TO`` with ``FORMAT PARQUET``
          when no ``PARTITION_BY`` clause is given).

        A plain ``.parquet`` extension variant is also supported for
        user-defined paths.

        Args:
            fact_table (str): Name of the fact table to locate.

        Returns:
            Tuple[str, bool]: ``(path, is_partitioned)`` where ``path`` is the
            root path to pass to ``read_parquet()`` and ``is_partitioned``
            indicates whether Hive partitioning should be enabled on read.

        Raises:
            FileNotFoundError: If the fact table cannot be found under any
                expected path.
        """
        if self._is_s3_path(self.input_dir):
            return self._detect_fact_table_s3(fact_table)

        # Chemin candidat commun (répertoire ou fichier selon l'export)
        base_path = f"{self.input_dir}/{fact_table}"

        # Priorité 1 : répertoire Hive (export avec PARTITION_BY)
        if os.path.isdir(base_path):
            self.logger.info(
                f"Fact table '{fact_table}' détectée comme répertoire partitionné : '{base_path}'"
            )
            return base_path, True

        # Priorité 2 : fichier plat sans extension (produit par export_schema)
        if os.path.isfile(base_path):
            self.logger.info(
                f"Fact table '{fact_table}' détectée comme fichier plat : '{base_path}'"
            )
            return base_path, False

        # Priorité 3 : fichier plat avec extension .parquet (chemins personnalisés)
        parquet_path = f"{base_path}.parquet"
        if os.path.isfile(parquet_path):
            self.logger.info(
                f"Fact table '{fact_table}' détectée comme fichier plat : '{parquet_path}'"
            )
            return parquet_path, False

        raise FileNotFoundError(
            f"Aucune fact table trouvée pour '{fact_table}' dans '{self.input_dir}'. "
            f"Chemins vérifiés : répertoire '{base_path}/', "
            f"fichier '{base_path}', fichier '{parquet_path}'."
        )

    # Détection de la fact table sur S3 via DuckDB glob
    def _detect_fact_table_s3(self, fact_table: str) -> Tuple[str, bool]:
        """
        Detect the fact table path for S3 URIs.

        Uses DuckDB's ``glob()`` table function to list objects under the
        expected prefix without requiring boto3.

        Args:
            fact_table (str): Name of the fact table to locate.

        Returns:
            Tuple[str, bool]: ``(path, is_partitioned)``.
        """
        # Recherche de fichiers Parquet dans un sous-répertoire (structure partitionnée)
        partitioned_prefix = f"{self.input_dir}/{fact_table}"
        try:
            result = self.conn.execute(
                f"SELECT file FROM glob('{partitioned_prefix}/**/*.parquet')"
            ).fetchall()
            if result:
                self.logger.info(
                    f"Fact table '{fact_table}' détectée sur S3 comme répertoire partitionné"
                )
                return partitioned_prefix, True
        except Exception:
            pass

        # Repli sur le chemin plat (fichier sans extension ou avec .parquet)
        self.logger.info(
            f"Fact table '{fact_table}' supposée plate sur S3 : '{partitioned_prefix}'"
        )
        return partitioned_prefix, False

    # Détection des tables de dimension dans le répertoire d'entrée
    def _detect_dim_tables(self, dim_table_prefix: str) -> List[str]:
        """
        List dimension table names present in input_dir.

        Args:
            dim_table_prefix (str): Prefix used to identify dimension Parquet
                files (e.g. ``'dim_'``).

        Returns:
            List[str]: Table names without the ``.parquet`` extension
                (e.g. ``['dim_category', 'dim_status']``).
        """
        if self._is_s3_path(self.input_dir):
            # Utilisation de DuckDB glob pour éviter toute dépendance à boto3
            try:
                result = self.conn.execute(
                    f"SELECT file FROM glob('{self.input_dir}/{dim_table_prefix}*.parquet')"
                ).fetchall()
                return [
                    os.path.basename(row[0]).replace('.parquet', '')
                    for row in result
                ]
            except Exception:
                return []

        # Utilisation du module glob de la bibliothèque standard pour les chemins locaux
        pattern = os.path.join(self.input_dir, f"{dim_table_prefix}*.parquet")
        files = sorted(glob_module.glob(pattern))
        return [os.path.basename(f).replace('.parquet', '') for f in files]

    # Extraction des clés primaires depuis la table de métadonnées chargée
    def _get_primary_keys_from_metadata(self, metadata_table: str = 'metadata') -> List[str]:
        """
        Extract primary key column names from the loaded metadata table.

        Args:
            metadata_table (str): Name of the metadata table in the DuckDB
                connection. Defaults to ``'metadata'``.

        Returns:
            List[str]: Column names where ``is_primary_key`` is True. Returns
                an empty list if the metadata table is not available.
        """
        try:
            existing_tables = [
                row[0] for row in self.conn.execute("SHOW TABLES").fetchall()
            ]
            if metadata_table not in existing_tables:
                return []

            result = self.conn.execute(
                f"SELECT name FROM {metadata_table} WHERE is_primary_key = TRUE"
            ).fetchall()
            return [row[0] for row in result]
        except Exception:
            return []

    # Inférence du schéma de colonnes depuis un fichier Parquet
    def _get_schema_from_parquet(
        self,
        parquet_path: str,
        hive_partitioning: bool = False,
    ) -> List[Tuple[str, str]]:
        """
        Infer column names and DuckDB SQL types from a Parquet source.

        Args:
            parquet_path (str): Path to the Parquet file or directory root.
            hive_partitioning (bool): Enable Hive partitioning on read.
                Defaults to False.

        Returns:
            List[Tuple[str, str]]: List of ``(column_name, sql_type)`` pairs
                as reported by DuckDB's ``DESCRIBE`` statement.
        """
        if hive_partitioning:
            read_expr = (
                f"read_parquet('{parquet_path}/**/*.parquet', hive_partitioning=true)"
            )
        else:
            read_expr = f"read_parquet('{parquet_path}')"

        result = self.conn.execute(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()
        # Chaque ligne de DESCRIBE contient : (column_name, column_type, null, key, default, extra)
        return [(row[0], row[1]) for row in result]

    # ---------------------------------------------------------------------------
    # Méthodes publiques de chargement
    # ---------------------------------------------------------------------------

    # Chargement d'une table individuelle depuis un fichier Parquet
    def load_table(
        self,
        table_name: str,
        parquet_path: str,
        overwrite: bool = False,
        hive_partitioning: bool = False,
        primary_key_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Load a single table from a Parquet file or Hive-partitioned directory into DuckDB.

        Without ``primary_key_columns``, the table is created with a single
        ``CREATE TABLE AS SELECT * FROM read_parquet(...)`` statement.
        When ``primary_key_columns`` is provided, a two-step DDL is used
        (``CREATE TABLE`` with explicit constraint + ``INSERT INTO ... SELECT``)
        because DuckDB does not support ``PRIMARY KEY`` constraints in a
        ``CREATE TABLE AS SELECT`` statement.

        Args:
            table_name (str): Name of the table to create in DuckDB.
            parquet_path (str): Path to the ``.parquet`` file or partitioned
                directory root. Accepts local paths or ``s3://`` URIs.
            overwrite (bool): Drop and recreate the table if it already exists.
                Defaults to False.
            hive_partitioning (bool): Enable Hive partitioning when reading
                from a partitioned directory. Defaults to False.
            primary_key_columns (Optional[List[str]]): Column names to declare
                as ``PRIMARY KEY``. Defaults to None (no constraint).

        Raises:
            ValueError: If the table already exists and ``overwrite`` is False.
            duckdb.Error: If any SQL statement fails.

        Examples:
            >>> importer.load_table('metadata', '/tmp/export/metadata.parquet')
            >>> importer.load_table(
            ...     'fact_table', '/tmp/export/fact_table',
            ...     hive_partitioning=True, primary_key_columns=['id'],
            ... )
        """
        try:
            # Vérification de l'existence préalable de la table
            existing_tables = [
                row[0] for row in self.conn.execute("SHOW TABLES").fetchall()
            ]

            if table_name in existing_tables:
                if not overwrite:
                    raise ValueError(
                        f"La table '{table_name}' existe déjà dans la connexion DuckDB. "
                        f"Utilisez overwrite=True pour la remplacer."
                    )
                # Suppression de la table existante avant recréation
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                self.logger.info(f"Table '{table_name}' supprimée avant recréation (overwrite=True)")

            # Construction de l'expression de lecture Parquet
            if hive_partitioning:
                read_expr = (
                    f"read_parquet('{parquet_path}/**/*.parquet', hive_partitioning=true)"
                )
            else:
                read_expr = f"read_parquet('{parquet_path}')"

            if primary_key_columns:
                # Chemin deux étapes : inférence du schéma + DDL explicite avec contrainte PRIMARY KEY.
                # DuckDB ne supporte pas les contraintes de clé primaire dans un CREATE TABLE AS SELECT,
                # d'où l'obligation d'utiliser un CREATE TABLE suivi d'un INSERT INTO.
                schema = self._get_schema_from_parquet(parquet_path, hive_partitioning)
                col_defs = ", ".join(f"{col} {dtype}" for col, dtype in schema)
                pk_clause = f"PRIMARY KEY ({', '.join(primary_key_columns)})"

                self.conn.execute(f"""
                    CREATE TABLE {table_name} (
                        {col_defs},
                        {pk_clause}
                    )
                """)
                self.conn.execute(f"""
                    INSERT INTO {table_name}
                    SELECT * FROM {read_expr}
                """)
                self.logger.info(
                    f"Table '{table_name}' chargée avec PRIMARY KEY "
                    f"({', '.join(primary_key_columns)}) depuis '{parquet_path}'"
                )
            else:
                # Chemin simple : CREATE TABLE AS SELECT (plus performant, pas de DDL intermédiaire)
                self.conn.execute(f"""
                    CREATE TABLE {table_name} AS
                    SELECT * FROM {read_expr}
                """)
                self.logger.info(
                    f"Table '{table_name}' chargée depuis '{parquet_path}'"
                )

        except (ValueError, duckdb.Error):
            # Propagation des erreurs de validation et DuckDB sans les masquer
            raise
        except Exception as e:
            self.logger.error(f"Échec du chargement de '{table_name}' : {e}")
            raise

    # Chargement complet du schéma (metadata + dimensions + fact table)
    def load_schema(
        self,
        fact_table: str = 'fact_table',
        metadata_table: str = 'metadata',
        dim_table_prefix: str = 'dim_',
        overwrite: bool = False,
    ) -> None:
        """
        Reconstruct the full DuckDB schema from a Parquet export directory.

        Tables are loaded in dependency order: metadata first (so that primary
        key columns can be recovered before the fact table is created), then
        dimension tables, then the fact table.

        The metadata and dimension tables use explicit DDL to faithfully
        reproduce the constraints created by ``DuckdbTablesBuilder``, ensuring
        compatibility with ``DatabaseUpdaterV2`` and ``BaseSchemaManager``.

        Output layout expected in ``input_dir``::

            input_dir/
            ├── fact_table          # flat file (no extension) OR
            ├── fact_table/         # Hive-partitioned directory
            │   └── col=val/
            │       └── data_0.parquet
            ├── metadata.parquet
            └── dim_category.parquet

        Args:
            fact_table (str): Name of the fact table to create in DuckDB.
                Defaults to ``'fact_table'``.
            metadata_table (str): Name of the metadata table. Defaults to
                ``'metadata'``.
            dim_table_prefix (str): Prefix to identify dimension Parquet files.
                Defaults to ``'dim_'``.
            overwrite (bool): Drop and recreate tables that already exist in
                the connection. Defaults to False.

        Raises:
            FileNotFoundError: If ``input_dir`` does not exist or the fact
                table cannot be found.
            ValueError: If a table already exists and ``overwrite`` is False.
            duckdb.Error: If any SQL reconstruction step fails.

        Examples:
            >>> importer = ParquetImporter('/tmp/export/', path='restored.duckdb')
            >>> importer.load_schema()
            >>> from dashboard_template_database.operations import DatabaseUpdaterV2
            >>> updater = DatabaseUpdaterV2(connection=importer.conn)
            >>> importer.load_schema(
            ...     fact_table='my_fact',
            ...     metadata_table='my_metadata',
            ...     overwrite=True,
            ... )
        """
        # Validation de l'accessibilité du répertoire d'entrée
        self._validate_input_dir()

        # --- Étape 1 : chargement de la table de métadonnées ---
        # La metadata est chargée en premier car elle est utilisée pour reconstruire
        # les contraintes PRIMARY KEY lors du chargement de la fact table.
        # Un DDL explicite est utilisé (plutôt qu'un CTAS) pour garantir que la colonne
        # 'name' soit déclarée comme PRIMARY KEY, conformément à ce que crée DuckdbTablesBuilder.
        metadata_path = f"{self.input_dir}/{metadata_table}.parquet"
        metadata_accessible = (
            self._is_s3_path(self.input_dir) or os.path.isfile(metadata_path)
        )

        if metadata_accessible:
            try:
                # Vérification et suppression éventuelle de la table existante
                existing_tables = [
                    row[0] for row in self.conn.execute("SHOW TABLES").fetchall()
                ]
                if metadata_table in existing_tables:
                    if not overwrite:
                        raise ValueError(
                            f"La table '{metadata_table}' existe déjà. "
                            f"Utilisez overwrite=True pour la remplacer."
                        )
                    self.conn.execute(f"DROP TABLE IF EXISTS {metadata_table}")

                # Création avec DDL explicite pour garantir PRIMARY KEY sur 'name'
                self.conn.execute(f"""
                    CREATE TABLE {metadata_table} (
                        name        VARCHAR PRIMARY KEY,
                        label       VARCHAR,
                        python_type VARCHAR,
                        sql_type    VARCHAR,
                        is_categorical BOOLEAN,
                        is_primary_key BOOLEAN
                    )
                """)
                self.conn.execute(f"""
                    INSERT INTO {metadata_table}
                    SELECT * FROM read_parquet('{metadata_path}')
                """)
                self.logger.info(
                    f"Table de métadonnées '{metadata_table}' chargée depuis '{metadata_path}'"
                )
            except ValueError:
                raise
            except Exception as e:
                # Avertissement non bloquant : l'absence de metadata empêchera la reconstruction
                # des PRIMARY KEY mais ne bloque pas le chargement du reste du schéma.
                self.logger.warning(
                    f"Impossible de charger la table de métadonnées '{metadata_table}' : {e}. "
                    f"Le chargement continue sans reconstruction des clés primaires."
                )
        else:
            self.logger.warning(
                f"Fichier de métadonnées '{metadata_path}' introuvable. "
                f"Les contraintes PRIMARY KEY de la fact table ne seront pas reconstruites."
            )

        # --- Étape 2 : chargement des tables de dimension ---
        # Les tables de dimension utilisent un DDL explicite (value VARCHAR PRIMARY KEY, label VARCHAR)
        # pour reproduire fidèlement le schéma créé par DuckdbTablesBuilder.create_duckdb_dimension_tables().
        dim_tables = self._detect_dim_tables(dim_table_prefix)

        if not dim_tables:
            self.logger.warning(
                f"Aucune table de dimension trouvée avec le préfixe '{dim_table_prefix}' "
                f"dans '{self.input_dir}'."
            )

        for dim_name in dim_tables:
            dim_path = f"{self.input_dir}/{dim_name}.parquet"

            # Vérification de l'existence préalable
            existing_tables = [
                row[0] for row in self.conn.execute("SHOW TABLES").fetchall()
            ]
            if dim_name in existing_tables:
                if not overwrite:
                    raise ValueError(
                        f"La table '{dim_name}' existe déjà. "
                        f"Utilisez overwrite=True pour la remplacer."
                    )
                self.conn.execute(f"DROP TABLE IF EXISTS {dim_name}")

            # Création avec DDL explicite pour garantir PRIMARY KEY sur 'value'
            self.conn.execute(f"""
                CREATE TABLE {dim_name} (
                    value VARCHAR PRIMARY KEY,
                    label VARCHAR
                )
            """)
            self.conn.execute(f"""
                INSERT INTO {dim_name}
                SELECT * FROM read_parquet('{dim_path}')
            """)
            self.logger.info(f"Table de dimension '{dim_name}' chargée depuis '{dim_path}'")

        # --- Étape 3 : chargement de la fact table ---
        # La structure (répertoire Hive vs fichier plat) est détectée automatiquement.
        # Les clés primaires sont extraites depuis la table de métadonnées si disponible.
        fact_path, is_partitioned = self._detect_fact_table_path(fact_table)
        primary_keys = self._get_primary_keys_from_metadata(metadata_table)

        self.load_table(
            table_name=fact_table,
            parquet_path=fact_path,
            overwrite=overwrite,
            hive_partitioning=is_partitioned,
            primary_key_columns=primary_keys if primary_keys else None,
        )

        # Récapitulatif des tables chargées dans la connexion
        loaded_tables = [
            row[0] for row in self.conn.execute("SHOW TABLES").fetchall()
        ]
        self.logger.info(
            f"Chargement du schéma terminé. Tables disponibles : {loaded_tables}"
        )
