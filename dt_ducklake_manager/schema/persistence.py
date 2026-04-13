# Importation des modules
# Modules de base
import os
from pathlib import Path
from typing import Literal

# Duckdb
import duckdb
import narwhals as nw
from narwhals.typing import IntoDataFrame

# Utilitaires de traitement des données
from ..utils.sql import remove_dataframe_duplicates

# Modules ad hoc
from .inference import SchemaBuilder

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe créant les tables correspondant au schéma dans un catalogue DuckLake
class DuckLakeTablesBuilder:
    """
    Builds and writes the database schema (metadata, dimension, and fact tables)
    into a DuckLake catalog.

    This class uses a ``SchemaBuilder`` instance (composition) to infer the three
    table layers of the dashboard database schema, then writes them directly into
    the DuckLake catalog attached to the provided connection:

    - ``metadata`` : column descriptors (type, label, categorical flag, primary key).
    - ``dim_<col>`` : dimension tables for low-cardinality categorical columns.
    - ``fact_table`` : the main fact table, optionally Hive-partitioned.

    The connection must be obtained from ``DuckLakeConnector.connect()`` before
    instantiating this class.

    Attributes:
        schema_builder (SchemaBuilder): Instance used for schema inference.
        conn (duckdb.DuckDBPyConnection): DuckLake-attached DuckDB connection.
        logger (logging.Logger): Logger shared with the SchemaBuilder.

    Examples:
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
        >>> builder = DuckLakeTablesBuilder(df, categorical_threshold=10,
        connection=conn)
        >>> builder.build_schema()
    """

    # Initialisation
    def __init__(
        self,
        df: IntoDataFrame,
        categorical_threshold: int | None = None,
        primary_keys: list[str] | None = None,
        connection: duckdb.DuckDBPyConnection | None = None,
        log_filename: os.PathLike | None = os.path.join(
            FILE_PATH.parents[2], "logs/ducklake_tables_builder.log"
        ),
    ):
        """
        Initialize the DuckLakeTablesBuilder.

        Args:
            df: Input dataset (pandas, polars, or narwhals-compatible) used to infer the
            schema.
            categorical_threshold (Optional[int]): Maximum number of distinct values
                for a column to be treated as categorical. Defaults to None (uses
                ``SchemaBuilder`` default).
            primary_keys (Optional[List[str]]): Column names used as logical primary
                keys (enforced applicatively, not as DDL constraints). Defaults to None.
            connection (Optional[duckdb.DuckDBPyConnection]): DuckLake-attached DuckDB
                connection obtained from ``DuckLakeConnector.connect()``. If None, an
                in-memory DuckDB connection is used (for unit tests only).
            log_filename (Optional[os.PathLike]): Path to the log file.

        Examples:
            >>> import polars as pl
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> df = pl.DataFrame({'col': [1, 2, 3]})
            >>> builder = DuckLakeTablesBuilder(df, connection=conn)
        """
        # Délégation de l'inférence du schéma à un SchemaBuilder (composition).
        # DuckLakeTablesBuilder n'est pas un SchemaBuilder : il utilise un SchemaBuilder
        # pour produire les DataFrames intermédiaires, puis les écrit dans DuckDB.
        self.schema_builder = SchemaBuilder(
            df=df,
            categorical_threshold=categorical_threshold,
            primary_keys=primary_keys,
            log_filename=log_filename,
        )

        # Réutilisation du logger du SchemaBuilder pour centraliser les traces
        self.logger = self.schema_builder.logger

        # Initialisation de la connexion DuckLake.
        # Le fallback :memory: est réservé aux tests unitaires ; en production la
        # connexion doit toujours être fournie via DuckLakeConnector.connect().
        self.conn = connection if connection is not None else duckdb.connect(":memory:")

    # Méthode de création de la table des méta-données
    def create_duckdb_metadata_table(
        self,
        table_name: str | None = "metadata",
        column_labels: dict[str, str] | None = None,
    ) -> None:
        """
        Create a metadata table in DuckDB.

        Args:
            table_name (Optional[str]): Name of the metadata table in DuckDB. Defaults
            to 'metadata'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names
            to labels.

        """
        # Création de la table des méta-données si elle n'existe pas déjà
        if not hasattr(self.schema_builder, "df_metadata"):
            _ = self.schema_builder.create_metadata_table(column_labels)

        # Création de la table avec schéma explicite
        # L'unicité de 'name' est garantie applicativement par DuckdbTablesBuilder.
        # Conversion vers Arrow pour garantir la compatibilité DuckDB quel que soit le
        # backend narwhals.
        # Note : .select() est nécessaire car narwhals/pandas inclut l'index comme
        # colonne
        # supplémentaire dans pa.Table.from_pandas() lorsque l'index n'est pas
        # séquentiel
        # (ex. après un .sort()). On filtre explicitement sur les colonnes nommées du
        # DataFrame.
        df_meta = self.schema_builder.df_metadata
        self.conn.register(
            "temp_metadata", df_meta.to_arrow().select(list(df_meta.columns))
        )
        self.conn.execute(f"""
            CREATE TABLE {table_name} (
                name VARCHAR,
                label VARCHAR,
                python_type VARCHAR,
                sql_type VARCHAR,
                is_categorical BOOLEAN,
                is_primary_key BOOLEAN
            )
        """)

        # Insertion des données depuis la vue temporaire
        self.conn.execute(f"""
            INSERT INTO {table_name}
            SELECT * FROM temp_metadata
        """)
        self.conn.execute("DROP VIEW temp_metadata")

        # Logging
        self.logger.info("Successfully registered duckdb meta-data table")

    # Méthode de création des tables de dimensions
    def create_duckdb_dimension_tables(
        self,
        table_prefix: str | None = "dim_",
        column_labels: dict[str, str] | None = None,
    ) -> None:
        """
        Create dimension tables in DuckDB for categorical variables.

        Args:
            table_prefix (Optional[str]): Prefix for dimension table names. Defaults to
            'dim_'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names
            to labels.

        """
        # Création du dictionnaire des tables de dimensions si elles n'existent pas déjà
        if not hasattr(self.schema_builder, "dimension_tables"):
            _ = self.schema_builder.create_dimension_tables(column_labels)

        # Création de chaque table de dimension dans DuckDB
        for dim_name, dim_df in self.schema_builder.dimension_tables.items():
            # Initialisation du nom de la table
            table_name = f"{table_prefix}{dim_name}"
            # Conversion vers Arrow pour garantir la compatibilité DuckDB quel que soit
            # le backend narwhals.
            # Note : .select() filtre l'index pandas éventuel (cf.
            # create_duckdb_metadata_table).
            self.conn.register(
                "temp_dim", dim_df.to_arrow().select(list(dim_df.columns))
            )

            # Création d'une table avec schéma explicite.
            # Pas de contrainte PRIMARY KEY : DuckLake ne supporte pas les contraintes
            # DDL.
            # L'unicité de 'value' est garantie applicativement par DimensionManager.
            self.conn.execute(f"""
                CREATE TABLE {table_name} (
                    value VARCHAR,
                    label VARCHAR
                )
            """)

            # Insertion des données depuis la vue temporaire
            self.conn.execute(f"""
                INSERT INTO {table_name}
                SELECT value, label FROM temp_dim
            """)

            # Ajout de la vue correspondante
            self.conn.execute("DROP VIEW temp_dim")

            # Logging
            self.logger.info(
                f"Successfully registered duckdb dimension table for {dim_name}"
            )

    # Méthode de création de la table d'informations
    def create_duckdb_fact_table(
        self,
        table_name: str | None = "fact_table",
        table_prefix: str | None = "dim_",
        column_labels: dict[str, str] | None = None,
        partition_by: list[str] | None = None,
    ) -> None:
        """
        Create a fact table in DuckDB with optional Hive partitioning.

        Args:
            table_name (Optional[str]): Name of the fact table in DuckDB. Defaults to
            'fact_table'.
            table_prefix (Optional[str]): Prefix for dimension table names. Defaults to
            'dim_'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names
            to labels.
            partition_by (Optional[List[str]]): Column names to partition the table by
            using
                DuckLake's Hive-style partitioning. Defaults to None (no partitioning).

        Examples:
            >>> builder.create_duckdb_fact_table()
            >>> builder.create_duckdb_fact_table(partition_by=['country', 'year'])
        """
        # Création de la table d'informations si elle n'existe pas déjà
        if not hasattr(self.schema_builder, "df_fact"):
            _ = self.schema_builder.create_fact_table(column_labels)

        df_fact = self.schema_builder.df_fact
        df_metadata = self.schema_builder.df_metadata
        primary_keys = self.schema_builder.primary_keys

        # Conversion vers Arrow pour garantir la compatibilité DuckDB quel que soit le
        # backend narwhals.
        # Note : .select() filtre l'index pandas éventuel (cf.
        # create_duckdb_metadata_table).
        self.conn.register(
            "temp_fact", df_fact.to_arrow().select(list(df_fact.columns))
        )

        # Inférence du schéma de colonnes depuis la table de métadonnées.
        # Utilisée dans les deux chemins DDL explicites (avec primary_keys ou avec
        # partition_by).
        needs_explicit_ddl = (primary_keys and len(primary_keys) > 0) or partition_by

        if needs_explicit_ddl:
            # Récupération des types SQL pour chaque colonne depuis la table de
            # métadonnées
            column_definitions = []
            for col in df_fact.columns:
                # Filtrage de la ligne de métadonnée correspondant à la colonne
                matching_rows = df_metadata.filter(nw.col("name") == col)
                if len(matching_rows) > 0:
                    # Extraction du type SQL via accès positionnel à la Series
                    sql_type = matching_rows.get_column("sql_type")[0]
                    column_definitions.append(f"{col} {sql_type}")

            # Construction de la clause PARTITION BY si des colonnes de partition sont
            # spécifiées.
            # Pas de clause PRIMARY KEY : DuckLake ne supporte pas les contraintes DDL.
            partition_clause = (
                f" PARTITION BY ({', '.join(partition_by)})" if partition_by else ""
            )

            # Création de la table avec DDL explicite
            query = f"""
                CREATE TABLE {table_name} (
                    {", ".join(column_definitions)}
                ){partition_clause}
            """
            self.conn.execute(query)

            # Insertion des données depuis la vue temporaire
            self.conn.execute(f"""
                INSERT INTO {table_name}
                SELECT {", ".join(df_fact.columns)}
                FROM temp_fact
            """)

            # Logging des clés logiques (non contraintes DDL)
            if primary_keys and len(primary_keys) > 0:
                pk_columns = ", ".join(primary_keys)
                self.logger.info(
                    f"Logical primary keys registered in the metadata table for the"
                    f" fact_table : ({pk_columns})"
                )
        else:
            # Chemin CTAS (sans partition ni clés primaires) : plus performant, pas de
            # DDL intermédiaire
            query = f"""
                CREATE TABLE {table_name} AS
                SELECT {", ".join(df_fact.columns)}
                FROM temp_fact
            """
            self.conn.execute(query)

        # Suppression de la vue temporaire
        self.conn.execute("DROP VIEW temp_fact")

        # Logging des clés étrangères créées (pour information)
        for dim_name in self.schema_builder.dimension_tables.keys():
            self.logger.info(
                f"Foreign key created for dimension '{dim_name}' - can be joined on"
                f" {table_name}.{dim_name} = {table_prefix}{dim_name}.value"
            )

        # Logging
        self.logger.info("Successfully registered duckdb fact table")

    # Méthode de construction du schéma
    def build_schema(
        self,
        metadata_table: str | None = "metadata",
        fact_table: str | None = "fact_table",
        dim_table_prefix: str | None = "dim_",
        column_labels: dict[str, str] | None = None,
        check_duplicates: bool = True,
        keep: Literal["any", "none", "first", "last"] = "none",
        partition_by: list[str] | None = None,
    ) -> None:
        """
        Build the entire schema in DuckDB, including metadata, dimension, and fact
        tables.

        Args:
            metadata_table (Optional[str]): Name of the metadata table. Defaults to
            'metadata'.
            fact_table (Optional[str]): Name of the fact table. Defaults to
            'fact_table'.
            dim_table_prefix (Optional[str]): Prefix for dimension tables. Defaults to
            'dim_'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names
            to labels.
            check_duplicates (bool): Whether to check and remove duplicates. Defaults to
            True.
            keep (Literal['any', 'none', 'first', 'last']): Which duplicates to keep.
            Defaults to 'none'.
            partition_by (Optional[List[str]]): Column names to partition the fact table
            by.
                Passed through to ``create_duckdb_fact_table()``. Defaults to None.

        Examples:
            >>> builder.build_schema()
            >>> builder.build_schema(partition_by=['country'])
        """
        # Vérification et suppression des doublons sur le DataFrame du SchemaBuilder.
        # Les clés primaires sont transmises à la fonction de déduplication afin que
        # la détection des doublons porte uniquement sur ces colonnes identifiantes,
        # en ignorant les colonnes de valeurs (value, lower_bound, upper_bound, etc.).
        primary_keys = self.schema_builder.primary_keys
        if check_duplicates:
            self.schema_builder.df = remove_dataframe_duplicates(
                self.schema_builder.df,
                keep,
                self.logger,
                "Main DataFrame",
                primary_keys=primary_keys if primary_keys else None,
            )

        # Vérification des doublons basée sur les clés primaires
        if len(primary_keys) > 0:
            # Réduction aux colonnes clés avant détection des doublons
            duplicates_mask = self.schema_builder.df.select(
                primary_keys
            ).is_duplicated()
            n_duplicates = duplicates_mask.sum()
            if n_duplicates > 0:
                raise ValueError(
                    f"Found {n_duplicates} duplicated rows based on the primary key"
                    f" columns {primary_keys}. "
                    f"Primary keys must be unique."
                )

        # Création de la table des méta-données
        self.create_duckdb_metadata_table(
            table_name=metadata_table, column_labels=column_labels
        )

        # Création de la table de dimensions
        self.create_duckdb_dimension_tables(
            table_prefix=dim_table_prefix, column_labels=column_labels
        )

        # Création de la table d'informations avec partitionnement optionnel
        self.create_duckdb_fact_table(
            table_name=fact_table,
            table_prefix=dim_table_prefix,
            column_labels=column_labels,
            partition_by=partition_by,
        )

    # Méthode d'affichage du schéma
    def display_schema(self) -> None:
        """
        Display the structure of all tables in the DuckDB schema.
        """
        # Extraction des tables
        tables = self.conn.execute("SHOW TABLES").fetchall()
        # Logging
        self.logger.info("\n Created Tables:")
        # Parcours des tables
        for table in tables:
            # Affichage de la structure
            self.logger.info(f"\n {table[0]} Structure:")
            # Extraction des informations relatives à la table
            table_info = self.conn.execute(f"DESCRIBE {table[0]}").fetchall()
            # Affichage de chaque information
            for col in table_info:
                self.logger.info(f"  {col[0]}: {col[1]}")
