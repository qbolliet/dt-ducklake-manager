# Importation des modules
# Modules de base
import json
import os
import time
from datetime import datetime
from typing import Any, Literal

# Duckdb
import duckdb
import narwhals as nw
from narwhals.typing import IntoDataFrame

# Rapport d'opération et traçabilité des runs
from ..reporting import (
    OperationReport,
    _current_snapshot_id,
    _set_commit_message,
    _table_changes_counts,
    _table_info,
)

# Utilitaires de traitement des données
from ..utils.sql import (
    SchemaScoped,
    quote_ident,
    remove_dataframe_duplicates,
    resolve_catalog,
)

# Modules ad hoc
from ..utils.types import METADATA_COLUMNS, metadata_table_ddl
from ..utils.value_labels import check_value_label_dependency
from .inference import SchemaBuilder

# Version du schéma de base de données écrite dans dataset_metadata.
# Ce champ n'existe que pour permettre une évolution future.
SCHEMA_VERSION: int = 1


# Classe créant les tables correspondant au schéma dans un catalogue DuckLake
class DuckLakeTablesBuilder(SchemaScoped):
    """
    Builds and writes the database schema into a DuckLake catalog.

    This class uses a ``SchemaBuilder`` instance (composition) to infer the schema,
    then writes exactly three tables into the DuckLake catalog attached to the
    provided connection:

    - ``fact_table`` : the observations, optionally Hive-partitioned. Categorical
      columns hold their **original labels** — there is no dimension table and no
      synthetic code anywhere in the schema.
    - ``metadata`` : one row per fact table column (label, SQL type, categorical
      and primary-key flags).
    - ``dataset_metadata`` : exactly one row describing the result set itself.

    The connection must be obtained from ``DuckLakeConnector.connect()`` before
    instantiating this class.

    A single catalog can hold several schemas (one per result set). The ``schema``
    argument selects the target schema; ``build_schema`` creates it if needed, so
    several result sets (e.g. ``predictions`` and ``shapley``) can be built on the
    same connection.

    Attributes:
        schema_builder (SchemaBuilder): Instance used for schema inference.
        conn (duckdb.DuckDBPyConnection): DuckLake-attached DuckDB connection.
        schema (str): DuckLake schema into which the tables are written.
        catalog_alias (str): Alias of the attached DuckLake catalog, carried
            alongside ``schema``.
        dataset_label (str | None): Title of the result set, written to
            ``dataset_metadata``.
        dataset_description (str | None): Description of the result set.
        dataset_source (str | None): Provenance of the result set (model,
            pipeline).
        logger (logging.Logger): Logger shared with the SchemaBuilder.

    Examples:
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
        >>> builder = DuckLakeTablesBuilder(df, categorical_threshold=10,
        connection=conn)
        >>> builder.build_schema()
        >>> # Build a second result set as a separate schema of the same catalog
        >>> DuckLakeTablesBuilder(df2, connection=conn,
        schema='shapley').build_schema()
    """

    # Initialisation
    def __init__(
        self,
        df: IntoDataFrame,
        categorical_threshold: int | None = None,
        primary_keys: list[str] | None = None,
        categorical_overrides: dict[str, bool] | None = None,
        hierarchies: dict[str, str] | None = None,
        value_labels: dict[str, str] | None = None,
        connection: duckdb.DuckDBPyConnection | None = None,
        schema: str = "main",
        catalog_alias: str = "db",
        dataset_label: str | None = None,
        dataset_description: str | None = None,
        dataset_source: str | None = None,
        log_filename: str | os.PathLike[str] | None = None,
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
            categorical_overrides (Optional[Dict[str, bool]]): Per-column forcing of
                the categorical status, independent of the threshold. The status is
                inferred once and never re-evaluated by a later update. Defaults to
                None.
            hierarchies (Optional[Dict[str, str]]): Column hierarchy declared as a
                mapping of child column name to parent column name, written to
                ``metadata.parent_name``. Must agree with
                any ``parent_name`` also supplied through ``column_metadata``.
                Defaults to None.
            value_labels (Optional[Dict[str, str]]): Code/label column pairs,
                declared as a mapping of label column name to code column name,
                written to ``metadata.label_for``, carried by the label column. Must
                agree with any ``label_for`` also supplied through
                ``column_metadata``. Defaults to None.
            connection (Optional[duckdb.DuckDBPyConnection]): DuckLake-attached DuckDB
                connection obtained from ``DuckLakeConnector.connect()``. If None, an
                in-memory DuckDB connection is used (for unit tests only).
            schema (str): DuckLake schema into which the three tables are written. A
                single catalog can host several schemas. Defaults to ``'main'``.
            catalog_alias (str): Alias of the attached DuckLake catalog, matching
                the one passed to ``DuckLakeConnector``. Carried alongside
                ``schema`` so table references can be qualified by the catalog.
                Defaults to ``'db'``.
            dataset_label (Optional[str]): Title of the result set, written to
                ``dataset_metadata.label``. Defaults to None.
            dataset_description (Optional[str]): Description of the result set.
                Defaults to None.
            dataset_source (Optional[str]): Provenance of the result set (model,
                pipeline). Defaults to None.
            log_filename (Optional[os.PathLike]): Path to the log file.

        Examples:
            >>> import polars as pl
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> df = pl.DataFrame({'col': [1, 2, 3]})
            >>> builder = DuckLakeTablesBuilder(df, connection=conn)
            >>> builder = DuckLakeTablesBuilder(df, connection=conn,
            schema='predictions')
        """
        # Délégation de l'inférence du schéma à un SchemaBuilder (composition).
        # DuckLakeTablesBuilder n'est pas un SchemaBuilder : il utilise un SchemaBuilder
        # pour produire les DataFrames intermédiaires, puis les écrit dans DuckDB.
        self.schema_builder = SchemaBuilder(
            df=df,
            categorical_threshold=categorical_threshold,
            primary_keys=primary_keys,
            categorical_overrides=categorical_overrides,
            hierarchies=hierarchies,
            value_labels=value_labels,
            log_filename=log_filename,
        )

        # Réutilisation du logger du SchemaBuilder pour centraliser les traces
        self.logger = self.schema_builder.logger

        # Initialisation de la connexion DuckLake.
        # Le fallback :memory: est réservé aux tests unitaires ; en production la
        # connexion doit toujours être fournie via DuckLakeConnector.connect().
        self.conn = connection if connection is not None else duckdb.connect(":memory:")

        # Schéma DuckLake cible : les tables sont qualifiées par ce schéma, permettant
        # à plusieurs jeux de résultats de coexister dans un même catalogue.
        self.schema = schema

        # Alias du catalogue DuckLake attaché : conservé au même titre que le schéma,
        # afin de pouvoir qualifier les tables par le catalogue.
        self.catalog_alias = catalog_alias

        # Alias de catalogue effectif : utilisé pour la qualification uniquement s'il
        # correspond à une base réellement attachée (None pour les connexions
        # in-memory des tests).
        self._catalog = resolve_catalog(self.conn, self.catalog_alias)

        # Métadonnées descriptives du jeu de résultats, écrites dans
        # dataset_metadata à la construction du schéma
        self.dataset_label = dataset_label
        self.dataset_description = dataset_description
        self.dataset_source = dataset_source

    # Méthode de création de la table des méta-données
    def create_duckdb_metadata_table(
        self,
        table_name: str | None = "metadata",
        column_labels: dict[str, str] | None = None,
        column_metadata: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """
        Create the metadata table in DuckDB, one row per fact table column.

        Args:
            table_name (Optional[str]): Name of the metadata table in DuckDB. Defaults
                to 'metadata'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names
                to labels.
            column_metadata (Optional[Dict[str, Dict[str, str]]]): Optional per-column
                UI metadata (``label``, ``parent_name``, ``unit``, ``display_format``,
                ``family``, ``description``, ``default_aggregation``). Forwarded to
                ``SchemaBuilder.create_metadata_table``; ``column_metadata`` wins over
                ``column_labels`` when both carry a label for the same column.

        Examples:
            >>> builder.create_duckdb_metadata_table()
            >>> builder.create_duckdb_metadata_table(table_name='column_metadata')
        """
        # Création de la table des méta-données si elle n'existe pas déjà
        if not hasattr(self.schema_builder, "df_metadata"):
            _ = self.schema_builder.create_metadata_table(
                column_labels, column_metadata
            )

        # Création de la table avec schéma explicite
        # L'unicité de 'name' est garantie applicativement par DuckdbTablesBuilder.
        # Conversion vers Arrow quel que soit le backend narwhals. Le .select()
        # écarte l'index qu'un backend pandas ajoute comme colonne lorsqu'il n'est
        # pas séquentiel (ex. après un tri).
        df_meta = self.schema_builder.df_metadata
        # Nom qualifié par le schéma (et le catalogue) cible
        qualified_name = self._qualified(table_name or "metadata")
        self.conn.register(
            "temp_metadata", df_meta.to_arrow().select(list(df_meta.columns))
        )
        self.conn.execute(metadata_table_ddl(qualified_name))

        # Insertion des données depuis la vue temporaire.
        # Liste de colonnes explicite : l'ordre du DataFrame inféré ne doit pas avoir
        # à coïncider avec celui du DDL.
        column_list = ", ".join(METADATA_COLUMNS)
        self.conn.execute(
            f"INSERT INTO {qualified_name} ({column_list})"
            f" SELECT {column_list} FROM temp_metadata"
        )
        self.conn.execute("DROP VIEW temp_metadata")

        # Logging
        self.logger.info("Successfully registered duckdb meta-data table")

    # Méthode de création de la table d'informations
    def create_duckdb_fact_table(
        self,
        table_name: str | None = "fact_table",
        column_labels: dict[str, str] | None = None,
        partition_by: list[str] | None = None,
        cluster_by: list[str] | None = None,
    ) -> None:
        """
        Create a fact table in DuckDB with optional Hive partitioning.

        Args:
            table_name (Optional[str]): Name of the fact table in DuckDB. Defaults to
                'fact_table'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names
                to labels.
            partition_by (Optional[List[str]]): Column names to partition the table by
                using DuckLake's Hive-style partitioning. Defaults to None (no
                partitioning).
            cluster_by (Optional[List[str]]): Column names the fact table is
                physically sorted by at write time (``ORDER BY`` on the initial
                ``INSERT``/CTAS). Enables DuckLake's per-file min/max pruning
                (``ducklake_file_column_stats``). Defaults to None (no sort).

        Examples:
            >>> builder.create_duckdb_fact_table()
            >>> builder.create_duckdb_fact_table(partition_by=['country', 'year'])
            >>> builder.create_duckdb_fact_table(cluster_by=['date', 'region'])
        """
        # Création de la table d'informations si elle n'existe pas déjà
        if not hasattr(self.schema_builder, "df_fact"):
            _ = self.schema_builder.create_fact_table(column_labels)

        df_fact = self.schema_builder.df_fact
        df_metadata = self.schema_builder.df_metadata
        primary_keys = self.schema_builder.primary_keys

        # Nom qualifié de la table des faits par le schéma (et le catalogue) cible
        qualified_name = self._qualified(table_name or "fact_table")

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

        # Clause ORDER BY commune aux deux chemins d'écriture (CTAS et DDL explicite) :
        # tri physique à l'écriture sur les colonnes de cluster_by, condition du
        # pruning par fichier.
        order_clause = (
            f"ORDER BY {', '.join(quote_ident(c) for c in cluster_by)}"
            if cluster_by
            else ""
        )

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
                    column_definitions.append(f"{quote_ident(col)} {sql_type}")

            # Création de la table avec DDL explicite
            query = f"""
                CREATE TABLE {qualified_name} (
                    {", ".join(column_definitions)}
                )
            """
            self.conn.execute(query)

            # Définition du partitionnement Hive via ALTER TABLE.
            # Cette instruction doit précéder l'INSERT pour que
            # les données soient écrites dans des fichiers partitionnés.
            if partition_by:
                self.conn.execute(
                    f"ALTER TABLE {qualified_name} SET PARTITIONED BY"
                    f" ({', '.join(partition_by)})"
                )

                # Logging
                self.logger.info(
                    f"The fact_table is successfully partitionned on"
                    f" the following keys : ({partition_by})"
                )

            # Insertion des données depuis la vue temporaire.
            # Liste de colonnes issue des données : identifiants entre guillemets.
            fact_columns_sql = ", ".join(quote_ident(c) for c in df_fact.columns)
            self.conn.execute(f"""
                INSERT INTO {qualified_name}
                SELECT {fact_columns_sql}
                FROM temp_fact
                {order_clause}
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
            fact_columns_sql = ", ".join(quote_ident(c) for c in df_fact.columns)
            query = f"""
                CREATE TABLE {qualified_name} AS
                SELECT {fact_columns_sql}
                FROM temp_fact
                {order_clause}
            """
            self.conn.execute(query)

        # Suppression de la vue temporaire
        self.conn.execute("DROP VIEW temp_fact")

        # Logging
        self.logger.info("Successfully registered duckdb fact table")

    # Méthode de création de la table des méta-données du jeu de résultats
    def create_duckdb_dataset_metadata_table(
        self,
        table_name: str | None = "dataset_metadata",
        cluster_by: list[str] | None = None,
    ) -> None:
        """
        Create the single-row ``dataset_metadata`` table describing the result set.

        ``updated_at`` and ``schema_version`` are always filled in; ``label``,
        ``description`` and ``source`` come from the optional builder arguments.
        ``cluster_by`` is left NULL unless explicitly provided: it is written as a
        JSON list of column names, kept in sync with the physical sort order applied
        by ``create_duckdb_fact_table``.

        Args:
            table_name (Optional[str]): Name of the table in DuckDB. Defaults to
                'dataset_metadata'.
            cluster_by (Optional[List[str]]): Physical sort key of the fact table, as
                a plain list of column names. Written to ``cluster_by`` as a JSON
                list; ``None`` writes ``NULL``. Defaults to None.

        Examples:
            >>> builder.create_duckdb_dataset_metadata_table()
            >>> builder.create_duckdb_dataset_metadata_table(cluster_by=['id'])
        """
        # Nom qualifié par le schéma (et le catalogue) cible
        qualified_name = self._qualified(table_name or "dataset_metadata")

        # Création de la table : une seule ligne par schéma
        self.conn.execute(f"""
            CREATE TABLE {qualified_name} (
                label VARCHAR,
                description VARCHAR,
                source VARCHAR,
                updated_at TIMESTAMP,
                schema_version INTEGER,
                cluster_by VARCHAR
            )
        """)

        # Insertion de l'unique ligne descriptive.
        # Horodatage lié en Python plutôt que via now() : la colonne est un TIMESTAMP
        # sans fuseau, là où now() renvoie un TIMESTAMP WITH TIME ZONE.
        self.conn.execute(
            f"""
            INSERT INTO {qualified_name}
                (label, description, source, updated_at, schema_version, cluster_by)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                self.dataset_label,
                self.dataset_description,
                self.dataset_source,
                datetime.now(),
                SCHEMA_VERSION,
                json.dumps(cluster_by) if cluster_by else None,
            ],
        )

        # Logging
        self.logger.info(
            f"Successfully registered duckdb dataset meta-data table"
            f" (schema_version={SCHEMA_VERSION})"
        )

    # Méthode de contrôle de la dépendance fonctionnelle des colonnes de libellés
    def _check_value_labels_dependency(self) -> None:
        """
        Check the code -> label functional dependency of every declared pair
        before writing.

        Runs on ``self.schema_builder.df`` — the deduplicated source DataFrame,
        already resolved by ``create_duckdb_metadata_table`` into
        ``self.schema_builder.value_labels_resolved`` — via a temporary view, so the
        check happens before the fact table itself is ever created. A no-op when no
        code/label pair is declared.

        Raises:
            ValueError: If the functional dependency code -> label is violated for
                any declared pair.
        """
        value_labels_resolved = self.schema_builder.value_labels_resolved
        if not value_labels_resolved:
            return

        df = self.schema_builder.df
        view_name = "_value_labels_check_src"
        self.conn.register(view_name, df.to_arrow().select(list(df.columns)))
        try:
            for label_col, code_col in value_labels_resolved.items():
                check_value_label_dependency(self.conn, view_name, code_col, label_col)
        finally:
            self.conn.unregister(view_name)

    # Méthode de construction du schéma
    def build_schema(
        self,
        metadata_table: str | None = "metadata",
        fact_table: str | None = "fact_table",
        dataset_metadata_table: str | None = "dataset_metadata",
        column_labels: dict[str, str] | None = None,
        column_metadata: dict[str, dict[str, str]] | None = None,
        check_duplicates: bool = True,
        keep: Literal["any", "none", "first", "last"] = "none",
        partition_by: list[str] | None = None,
        cluster_by: list[str] | None = None,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> OperationReport:
        """
        Build the entire schema in DuckDB: metadata, fact and dataset_metadata
        tables.

        Runs as a single DuckDB transaction (``BEGIN``/``COMMIT``, ``ROLLBACK`` on
        exception): unlike the rest of the build (duplicate validation,
        ``cluster_by`` resolution), which happens before any DDL and simply
        raises, the three ``CREATE``/``INSERT`` steps either all land or none do.

        Args:
            metadata_table (Optional[str]): Name of the metadata table. Defaults to
                'metadata'.
            fact_table (Optional[str]): Name of the fact table. Defaults to
                'fact_table'.
            dataset_metadata_table (Optional[str]): Name of the dataset metadata
                table. Defaults to 'dataset_metadata'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names
                to labels.
            column_metadata (Optional[Dict[str, Dict[str, str]]]): Optional per-column
                UI metadata (``label``, ``parent_name``, ``label_for``, ``unit``,
                ``display_format``, ``family``, ``description``,
                ``default_aggregation``), written into the metadata table. Defaults
                to None.
            check_duplicates (bool): Whether to check and remove duplicates. Defaults to
                True.
            keep (Literal['any', 'none', 'first', 'last']): Which duplicates to keep.
                Defaults to 'none'.
            partition_by (Optional[List[str]]): Column names to partition the fact table
                by.
                Passed through to ``create_duckdb_fact_table()``. Defaults to None.
            cluster_by (Optional[List[str]]): Column names the fact table is
                physically sorted by at write time. Defaults to the primary keys, in
                their declared order, when primary keys are set; otherwise no sort is
                applied. Every column must exist in the source DataFrame. Persisted
                to ``dataset_metadata.cluster_by`` as a JSON list.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``). Ignored (skipped with a DEBUG
                log) on a connection with no real DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``.

        Raises:
            ValueError: If ``cluster_by`` references a column absent from the source
                DataFrame, in addition to the existing primary-key duplicate check, or
                if a declared code/label pair (``value_labels``/``label_for``)
                violates the functional dependency code -> label on the deduplicated
                DataFrame (nothing is written, including the ``metadata`` rows).

        Returns:
            OperationReport: report describing the tables just built
            (``columns_added`` lists the fact table's columns).

        Examples:
            >>> builder.build_schema()
            >>> builder.build_schema(partition_by=['country'])
            >>> builder.build_schema(cluster_by=['date', 'region'])
        """
        # Création du schéma cible s'il n'existe pas encore.
        # Utile lorsque la connexion n'a pas été préparée par DuckLakeConnector
        # (ex. connexion in-memory de test) ou pour ajouter un nouveau schéma à un
        # catalogue existant. CREATE SCHEMA IF NOT EXISTS est idempotent.
        schema_ref = quote_ident(self.schema)
        if self._catalog is not None:
            schema_ref = f"{quote_ident(self._catalog)}.{schema_ref}"
        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_ref}")

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

        # Résolution de cluster_by : par défaut les clés primaires dans leur ordre de
        # déclaration (colonnes les plus sélectives filtrées en premier) ; sans clé
        # primaire, aucun tri par défaut. Une valeur explicite doit référencer des
        # colonnes existantes du DataFrame source.
        if cluster_by is None:
            cluster_by = list(primary_keys) if primary_keys else None
        else:
            unknown_cluster_by = [
                c for c in cluster_by if c not in self.schema_builder.df.columns
            ]
            if unknown_cluster_by:
                raise ValueError(
                    f"cluster_by columns {unknown_cluster_by} do not exist in the"
                    f" DataFrame"
                )

        # Avant-état : la table n'existe pas encore (rows_before/files_before/
        # bytes_before restent à 0), mais le catalogue peut déjà porter d'autres
        # schémas, d'où un snapshot_before potentiellement non nul.
        start_time = time.time()
        started_at = datetime.now()
        snapshot_before = _current_snapshot_id(self.conn, self._catalog, self.logger)
        report = OperationReport(
            operation="build",
            schema=self.schema,
            run_id=run_id,
            started_at=started_at,
            duration_seconds=0.0,
            snapshot_before=snapshot_before,
        )

        # Transaction DuckDB unique : les trois tables sont créées ensemble ou pas
        # du tout, et le message de commit DuckLake s'applique au batch entier.
        self.conn.begin()
        try:
            # Création de la table des méta-données
            self.create_duckdb_metadata_table(
                table_name=metadata_table,
                column_labels=column_labels,
                column_metadata=column_metadata,
            )

            # Contrôle de la dépendance fonctionnelle code -> libellé, sur le DataFrame
            # dédupliqué, avant l'écriture de la fact_table : une violation annule
            # tout, y compris les lignes de metadata déjà insérées ci-dessus (même
            # transaction).
            self._check_value_labels_dependency()

            # Création de la table d'informations avec partitionnement et tri
            # optionnels
            self.create_duckdb_fact_table(
                table_name=fact_table,
                column_labels=column_labels,
                partition_by=partition_by,
                cluster_by=cluster_by,
            )

            # Création de la table des méta-données du jeu de résultats
            self.create_duckdb_dataset_metadata_table(
                table_name=dataset_metadata_table,
                cluster_by=cluster_by,
            )

            report.columns_added = list(self.schema_builder.df.columns)

            # Message de commit DuckLake (traçabilité du run), avant COMMIT.
            _set_commit_message(
                self.conn,
                self._catalog,
                run_id,
                commit_message,
                {
                    "operation": "build",
                    "schema": self.schema,
                    "columns_added": report.columns_added,
                    **(commit_info or {}),
                },
                self.logger,
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            report.duration_seconds = time.time() - start_time
            # Logging
            self.logger.error(
                f"build {self.schema} FAILED after {report.duration_seconds:.2f}s"
                f" (run_id={run_id}): {e}"
            )
            raise

        # Après-état
        # Comptage du nombre de lignes
        report.rows_after = self._count_fact_table_rows(fact_table or "fact_table")
        # Extraction des informations associées à la table des faits
        info = _table_info(
            self.conn,
            self._catalog,
            self.schema,
            fact_table or "fact_table",
            self.logger,
        )
        # Déstructuration des informations de la table des faits dans le rapport
        if info is not None:
            report.files_after, report.bytes_after, _, _ = info
        # Snapshot
        report.snapshot_after = _current_snapshot_id(
            self.conn, self._catalog, self.logger
        )
        # Changements dans la table des faits
        changes = _table_changes_counts(
            self.conn,
            self._catalog,
            self.schema,
            fact_table or "fact_table",
            snapshot_before,
            report.snapshot_after,
            self.logger,
        )
        report.rows_inserted = changes.get("insert", 0)
        report.duration_seconds = time.time() - start_time

        # Logging
        self.logger.info(report.summary())
        return report

    # Méthode auxiliaire de comptage des lignes de la table des faits
    def _count_fact_table_rows(self, table: str) -> int:
        """Count the rows of the just-built fact table.

        Args:
            table: Bare table name (e.g. ``'fact_table'``).

        Returns:
            int: Row count, or ``0`` if it cannot be read.
        """
        try:
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._qualified(table)}"
            ).fetchone()
            return int(row[0]) if row is not None else 0
        except Exception:
            return 0

    # Méthode d'affichage du schéma
    def display_schema(self) -> None:
        """
        Display the structure of all tables in the DuckDB schema.
        """
        # Extraction des tables du schéma cible.
        # Filtrage par schéma : SHOW TABLES ne liste que le schéma actif de la
        # connexion, qui n'est pas nécessairement le schéma de ce builder.
        # On exlcut les tables internes de ducklake
        tables = self.conn.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = ? AND table_name NOT LIKE 'ducklake_%'",
            [self.schema],
        ).fetchall()

        print(tables)

        # Logging
        self.logger.info("\n Created Tables:")
        # Parcours des tables
        for table in tables:
            # Affichage de la structure
            self.logger.info(f"\n {table[0]} Structure:")
            # Extraction des informations relatives à la table (qualifiée)
            table_info = self.conn.execute(
                f"DESCRIBE {self._qualified(table[0])}"
            ).fetchall()
            # Affichage de chaque information
            for col in table_info:
                self.logger.info(f"  {col[0]}: {col[1]}")
