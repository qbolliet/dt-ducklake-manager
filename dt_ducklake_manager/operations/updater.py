# Importation des modules
# Modules de base
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# DuckDB
import duckdb
import narwhals as nw
from narwhals.typing import IntoDataFrame

from ..maintenance.auditor import DatabaseAuditor, IssueSeverity, ValidationLevel
from ..reporting import OperationReport

# Import des utilitaires
from ..utils.sql import (
    build_database_duplicate_removal_query,
    quote_ident,
    remove_dataframe_duplicates,
)
from ..utils.types import map_python_to_sql_type, validate_column_metadata
from ..utils.value_labels import check_value_label_dependency, get_value_label_columns

# Import des gestionnaires
from ._base import BaseSchemaManager
from ._data import DataManager

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe de mise à jour d'une base de données
class DatabaseUpdater(BaseSchemaManager):
    """
    Refactored database updater using the new modular architecture.

    Every public write operation runs as a single DuckDB transaction (``BEGIN`` /
    ``COMMIT``, ``ROLLBACK`` on exception) opened by
    :meth:`BaseSchemaManager._transaction`; post-write maintenance runs after the
    commit. Recovery beyond a failed operation relies on DuckLake time travel
    (``DatabaseRecoveryManager.list_ducklake_snapshots`` and
    ``DuckLakeConnector(..., snapshot_version=N)``), not on application backups.

    Attributes:
        data_mgr (DataManager): Manages fact table operations
        auditor (DatabaseAuditor): Validates database state and operations
        max_workers (int): Maximum number of parallel workers
        batch_size (int): Size of batches for processing large datasets
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        categorical_threshold: int | None = 50,
        log_filename: str | os.PathLike[str] | None = None,
        max_workers: int = 4,
        batch_size: int = 10000,
        enable_validation: bool = True,
        catalog_alias: str = "db",
        schema: str = "main",
    ):
        """
        Initialize the refactored database updater.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            categorical_threshold: Maximum number of distinct non-null values for a
                textual column to be flagged categorical when it is created (by
                ``allow_new_columns`` or ``add_columns``). Never re-evaluated.
            log_filename: Path to log file.
            max_workers: Maximum number of parallel workers.
            batch_size: Size of batches for processing.
            enable_validation: Whether to enable pre/post operation validation.
            catalog_alias: Alias used in the DuckLake ATTACH statement.
                Passed to maintenance and snapshot calls. Defaults to ``'db'``.
            schema: DuckLake schema to update. A single catalog can host several
                schemas; all tables are qualified by this one, and maintenance and
                snapshot calls target it. Defaults to ``'main'``.

        Examples:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> updater = DatabaseUpdater(conn, max_workers=8, enable_validation=True)
            >>> updater = DatabaseUpdater(conn, schema='predictions')
        """
        # Initialisation du parent
        super().__init__(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
            schema=schema,
            catalog_alias=catalog_alias,
        )

        # Initialisation des gestionnaires spécialisés
        self.data_mgr = DataManager(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
            batch_size=batch_size,
            schema=schema,
            catalog_alias=catalog_alias,
        )

        self.auditor = (
            DatabaseAuditor(
                connection=connection,
                log_filename=log_filename,
                schema=schema,
                catalog_alias=catalog_alias,
            )
            if enable_validation
            else None
        )

        # Configuration
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_validation = enable_validation

        # Configuration DuckLake pour les appels de maintenance (compaction) :
        # l'alias du catalogue (self.catalog_alias) et le schéma (self.schema) sont
        # tous deux portés par la classe de base BaseSchemaManager.

    # Méthode de validation d'une opération
    def validate_operation(self, operation_type: str, **kwargs: Any) -> bool:
        """
        Validate update operations before execution.

        Args:
            operation_type: Type of operation to validate
            **kwargs: Operation-specific parameters

        Returns:
            True if operation is valid
        """
        # Absence de validation si un auditeur n'est pas spécifié ou si elle n'est pas
        # permise
        if not self.enable_validation or not self.auditor:
            return True

        # Validation par l'auditeur
        validation_report = self.auditor.validate_operation_preconditions(
            operation_type, **kwargs
        )

        # Vérification des problèmes critiques
        if validation_report.get_critical_issues_count() > 0:
            # Logging
            self.logger.error(
                f"Critical validation issues found for {operation_type} operation:"
            )
            # Logging des erreurs critiques
            for issue in validation_report.get_issues_by_severity(
                IssueSeverity.CRITICAL
            ):
                self.logger.error(f"  - {issue.description}")
            return False

        # Avertissements pour les problèmes de priorité haute
        high_issues = validation_report.get_issues_by_severity(IssueSeverity.HIGH)
        if high_issues:
            # Logging
            self.logger.warning(
                f"High priority validation issues found for {operation_type} operation:"
            )
            # Logging des problèmes de priorité haute
            for issue in high_issues:
                self.logger.warning(f"  - {issue.description}")

        return True

    # Méthode principale de mise à jour
    def update_database(
        self,
        update_df: IntoDataFrame,
        check_duplicates_db: bool = True,
        check_duplicates_update: bool = True,
        keep: Literal["any", "none", "first", "last"] = "none",
        use_batch_processing: bool = True,
        use_transaction: bool = True,
        compact_after_update: bool = True,
        allow_new_columns: bool = False,
        column_metadata: dict[str, dict[str, str]] | None = None,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update the entire database with new data using atomic operations.

        Args:
            update_df: DataFrame containing update data
            check_duplicates_db: Whether to check duplicates in existing database
            check_duplicates_update: Whether to check duplicates in update DataFrame
            keep: Which duplicates to keep when removing duplicates
            use_batch_processing: Whether to use batch processing for large datasets
            use_transaction: Whether to run every step inside a single DuckDB
                transaction, so that a failure mid-update leaves the database
                exactly as it was. Defaults to True. When False the steps run in
                autocommit mode and a failure leaves partial state behind.
            compact_after_update: Whether to run DuckLake compaction (merge small delta
                files and rewrite delete files) immediately after a successful update.
                Adds write latency but keeps read performance optimal. Defaults to True.
            allow_new_columns: Whether a column of ``update_df`` absent from the
                fact table may be added. Defaults to False: an unknown column then
                raises ``ValueError`` instead of being silently added. When True,
                each new column is added (SQL type via ``map_python_to_sql_type``),
                given a ``metadata`` row (``is_primary_key=False``, ``is_categorical``
                inferred), and its UI fields are taken from ``column_metadata``.
            column_metadata: Per-new-column UI fields (``label``, ``unit``,
                ``display_format``, ``family``, ``description``,
                ``default_aggregation``, ``parent_name``), applied only when
                ``allow_new_columns`` is True. Ignored for columns that already
                exist in the fact table.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``), and on ``self.last_report``.
                Ignored (skipped with a DEBUG log) on a connection with no real
                DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info`` (e.g. ``{'model_version': '1.3'}``).

        Returns:
            True if update was successful, False otherwise. The
            :class:`~dt_ducklake_manager.reporting.OperationReport` describing what
            actually happened is exposed via ``self.last_report`` regardless of
            success (a partial report is attached on failure too).

        Raises:
            ValueError: If ``update_df`` carries a column absent from the fact
                table and ``allow_new_columns`` is False.

        Examples:
            >>> success = updater.update_database(new_data_df)
            >>> success = updater.update_database(new_data_df,
            compact_after_update=True)
            >>> success = updater.update_database(
            ...     new_data_df, allow_new_columns=True,
            ...     column_metadata={'score': {'unit': '%'}})
        """
        # Conversion vers narwhals dès le point d'entrée public
        update_df = nw.from_native(update_df, eager_only=True)

        # Validation préalable
        if not self.validate_operation("update", df=update_df):
            # Logging
            self.logger.error("Pre-update validation failed")
            return False

        # Vérification de l'existence d'une clé primaire sur la fact_table :
        # sans clé primaire, il est impossible de distinguer les lignes à insérer
        # de celles à mettre à jour, rendant l'opération non déterministe.
        primary_keys = self._get_primary_key_columns()
        if not primary_keys:
            self.logger.error(
                "No primary key is defined on the fact_table. "
                "The update procedure requires a primary key."
            )
            return False

        # Colonnes du DataFrame absentes de la fact_table : refusées par défaut.
        # L'ajout implicite silencieux (DataManager._ensure_columns_exist)
        # n'est plus jamais atteint depuis cette méthode publique puisque les
        # colonnes manquantes sont ajoutées explicitement ci-dessous avant que
        # l'insertion/upsert ne s'exécute.
        existing_columns = set(self._get_fact_table_columns())
        new_columns = [c for c in update_df.columns if c not in existing_columns]
        if new_columns:
            if not allow_new_columns:
                raise ValueError(
                    f"Unknown column(s) in update_df: {sorted(new_columns)}. Pass "
                    "allow_new_columns=True to add them explicitly."
                )
            self._add_new_columns_from_update(update_df, new_columns, column_metadata)

        # Logging
        self.logger.info(
            f"Starting database update with {len(update_df)} rows (transaction:"
            f"{use_transaction})"
        )

        # Bloc transactionnel unique : toutes les étapes ou aucune.
        try:
            with self._transaction(
                "update",
                use_transaction=use_transaction,
                run_id=run_id,
                commit_message=commit_message,
                commit_info=commit_info,
            ) as report:
                report.columns_added = list(new_columns)
                self._run_update_steps(
                    update_df,
                    check_duplicates_db,
                    check_duplicates_update,
                    keep,
                    use_batch_processing,
                    report,
                )
        except Exception as e:
            # Logging
            self.logger.error(f"Error during database update: {e}")
            # Avertissement sur les colonnes ajoutées hors transaction : ajoutées
            # avant l'upsert (les étapes suivantes en dépendent), elles subsistent.
            if new_columns:
                self.logger.warning(
                    f"Update failed after adding column(s) {sorted(new_columns)};"
                    " those columns remain in the fact table"
                )
            return False

        # Logging
        self.logger.info("Database update completed successfully")
        # Invalidation du cache des méta-données
        self._invalidate_metadata_cache()
        # Horodatage de la dernière écriture réussie.
        # Placé après le commit : un échec d'horodatage, purement descriptif,
        # ne doit jamais annuler une écriture de données.
        self._touch_dataset_metadata()
        # Compaction DuckLake optionnelle après le commit (fusion des petits fichiers
        # delta) : la maintenance ne fait jamais partie de la transaction.
        final_report = self.last_report
        if compact_after_update:
            self.maintenance.compact(schema=self.schema, report=final_report)
        if final_report is not None:
            self._finalize_report_after_write(final_report)
            self.logger.info(final_report.summary())
        return True

    # Méthode auxiliaire d'ajout explicite des colonnes inconnues d'un update_df
    def _add_new_columns_from_update(
        self,
        update_df: nw.DataFrame[Any],
        new_columns: list[str],
        column_metadata: dict[str, dict[str, str]] | None,
    ) -> None:
        """Add columns of ``update_df`` absent from the fact table, with metadata.

        Called by ``update_database`` when ``allow_new_columns=True``. Each column
        is added via ``ALTER TABLE ... ADD COLUMN`` (type from
        ``map_python_to_sql_type``), given a ``metadata`` row
        (``is_primary_key=False``, ``is_categorical`` inferred from the batch), and
        its UI fields (if any, in ``column_metadata``) are applied. Runs before the
        insert/upsert step, so the columns already exist by the time it executes.

        Args:
            update_df: The full update DataFrame (narwhals).
            new_columns: Columns of ``update_df`` absent from the fact table.
            column_metadata: Per-column UI fields, validated against
                ``new_columns``.

        Raises:
            ValueError: If ``column_metadata`` references a column outside
                ``new_columns`` or carries an unknown/invalid field.
        """
        # Validation du dictionnaire de métadonnées d'UI, restreint aux colonnes
        # effectivement nouvelles
        normalized_metadata = validate_column_metadata(column_metadata, new_columns)

        # Parcours des colonnes
        for column in new_columns:
            # Ajout de la colonne à la fact table
            sql_type = map_python_to_sql_type(update_df.schema[column])
            self.conn.execute(
                f"ALTER TABLE {self._qualified('fact_table')} ADD COLUMN"
                f" {quote_ident(column)} {sql_type} DEFAULT NULL"
            )
            # Ligne de méta-données (is_primary_key=FALSE, is_categorical inféré)
            self._add_column_to_metadata(column, update_df)
            # Champs d'UI éventuels
            fields = normalized_metadata.get(column)
            if fields:
                self.update_column_metadata(column, **fields)

        # Logging
        self.logger.info(f"Added new column(s) from update_df: {new_columns}")

    # Méthode de contrôle de la dépendance fonctionnelle des colonnes de libellés
    # après l'écriture d'un lot (update_database ou add_columns)
    def _check_value_label_dependencies(
        self,
        df: nw.DataFrame[Any],
        primary_keys: list[str],
        touched_columns: set[str],
    ) -> None:
        """
        Check label -> code association for every declared pair touched by a batch.

        Restricted to the codes of the rows the batch touched, not a full-table
        scan: for each declared code column whose code column or at least one of
        its label columns is in ``touched_columns``, builds the *current*
        (post-write) distinct code values of the fact table rows matching ``df``'s
        primary keys, then checks the dependency against that restriction. This one
        restriction view correctly covers every scenario of a refused update: a new
        label for an existing code, a batch without the label column inserting a
        code already labeled elsewhere, and a non-primary-key code changed without
        its label.

        Args:
            df: The batch just written (``update_df`` or ``add_columns``' ``df``),
                carrying every primary key column.
            primary_keys: Primary key columns of the fact table, used to join ``df``
                back to it.
            touched_columns: Columns of ``df`` that were actually written.

        Raises:
            ValueError: If the functional dependency is violated for any relevant
                pair, naming the faulty codes and pointing to
                ``DatabaseUpdater.update_value_labels`` for a deliberate relabeling.
        """
        # Paires code -> colonnes de libellés concernées par ce lot
        pairs = get_value_label_columns(
            self.conn, schema=self.schema, catalog_alias=self.catalog_alias
        )
        relevant = {
            code_col: label_cols
            for code_col, label_cols in pairs.items()
            if code_col in touched_columns
            or any(label_col in touched_columns for label_col in label_cols)
        }
        if not relevant:
            return

        # Extraction de la table des faits
        fact_table = self._qualified("fact_table")
        batch_view = "_value_label_check_batch"
        codes_view = "_value_label_check_codes"
        # Enregistrement de la vue
        self.conn.register(batch_view, nw.to_native(df))
        try:
            # Condition de jointure sur les clés primaires
            join_condition = " AND ".join(
                f"f.{quote_ident(k)} = t.{quote_ident(k)}" for k in primary_keys
            )
            # Parcours des colonnes de code/labels
            for code_col, label_cols in relevant.items():
                # Extraction des codes sur la vue
                quoted_code = quote_ident(code_col)
                self.conn.execute(f"""
                    CREATE OR REPLACE TEMP VIEW {codes_view} AS
                    SELECT DISTINCT f.{quoted_code} AS {quoted_code}
                    FROM {fact_table} f
                    JOIN {batch_view} t ON {join_condition}
                """)
                try:
                    # Vérification de l'association label -> code
                    for label_col in label_cols:
                        check_value_label_dependency(
                            self.conn,
                            fact_table,
                            code_col,
                            label_col,
                            restrict_to=codes_view,
                        )
                finally:
                    self.conn.execute(f"DROP VIEW IF EXISTS {codes_view}")
        finally:
            # Suppression de la vue
            self.conn.unregister(batch_view)

    # Méthode d'exécution ordonnée des étapes d'une mise à jour
    def _run_update_steps(
        self,
        update_df: nw.DataFrame[Any],
        check_duplicates_db: bool,
        check_duplicates_update: bool,
        keep: Literal["any", "none", "first", "last"],
        use_batch_processing: bool,
        report: OperationReport,
    ) -> None:
        """Run the ordered steps of a database update.

        Steps, in order: update-data deduplication, database deduplication,
        metadata update, fact table upsert, value-label functional dependency check
        ), post-update validation. Called from
        inside the transaction opened by ``update_database``: every failure raises,
        so the whole update is rolled back and the database returns to its
        pre-update state.

        Args:
            update_df: DataFrame containing the update data.
            check_duplicates_db: Whether to remove duplicates from the fact table.
            check_duplicates_update: Whether to remove duplicates from ``update_df``.
            keep: Duplicate handling strategy ('any', 'none', 'first', 'last').
            use_batch_processing: Whether to use batch processing for large datasets.
            report: In-progress report of the enclosing transaction, appended to
                (``metadata_changes``) as categorical flags flip.

        Raises:
            RuntimeError: If any step fails, naming the step reached. The exception
                message is what the caller logs as the step at which the update
                stopped.
            ValueError: If the update violates the functional dependency of a
                code/label column pair declared on a column of ``update_df``.
        """
        # Étape 1 : suppression des doublons dans les données de mise à jour
        if check_duplicates_update:
            update_df = self._get_cleaned_update_data(update_df, keep)

        # Étape 2 : suppression des doublons dans la base de données
        if check_duplicates_db and not self._remove_database_duplicates(keep):
            raise RuntimeError("database duplicate removal failed")

        # Étape 3 : mise à jour des métadonnées
        if not self._update_metadata_safe(update_df, report):
            raise RuntimeError("metadata update failed")

        # Étape 4 : mise à jour de la table de faits
        if use_batch_processing and len(update_df) > self.batch_size:
            if not self._update_fact_table_batch(update_df, report):
                raise RuntimeError("fact table update failed (batch processing)")
        else:
            if not self._update_fact_table_direct(update_df, report):
                raise RuntimeError("fact table update failed (direct)")

        # Étape 4b : dépendance fonctionnelle des colonnes de libellés, sur
        # l'état post-upsert, restreinte aux codes du lot. Une violation lève
        # directement une ValueError (pas de RuntimeError intermédiaire) : le
        # message pointe déjà vers update_value_labels pour un changement de
        # libellé délibéré.
        self._check_value_label_dependencies(
            update_df, self._get_primary_key_columns(), set(update_df.columns)
        )

        # Étape 5 : validation post-update
        if self.enable_validation and self.auditor:
            validation_report = self.auditor.validate_database(ValidationLevel.STANDARD)
            # Problèmes critiques : annulation de l'ensemble de la mise à jour
            if validation_report.get_critical_issues_count() > 0:
                raise RuntimeError(
                    "post-update validation found"
                    f" {validation_report.get_critical_issues_count()} critical"
                    " issue(s)"
                )

    # Méthodes de mise à jour sécurisées
    # Méthode auxiliaire de mise à jour des méta-données
    def _update_metadata_safe(
        self, update_df: nw.DataFrame[Any], report: OperationReport | None = None
    ) -> bool:
        """Safely update metadata table with type conflict resolution.

        Args:
            update_df: DataFrame whose columns may require metadata updates.
            report: When given, each resolved type conflict is appended to
                ``report.metadata_changes``.

        Returns:
            True if metadata updated successfully, False on error.
        """
        try:
            # Chargement des métadonnées actuelles
            current_metadata = self._load_current_metadata()
            current_columns = (
                set(current_metadata["name"].to_list())
                if len(current_metadata) > 0
                else set()
            )
            new_columns = set(update_df.columns)

            # Vérification des conflits de types pour les colonnes existantes
            for col in current_columns.intersection(new_columns):
                self._resolve_type_conflicts(col, update_df, current_metadata, report)

            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error updating metadata: {e}")
            return False

    # Méthode auxiliaire de mise à jour directe de la table des faits
    def _update_fact_table_direct(
        self, update_df: nw.DataFrame[Any], report: OperationReport
    ) -> bool:
        """Update fact table directly without batch processing.

        Uses primary keys from metadata to determine INSERT vs UPSERT strategy:
        - No primary keys defined: Always INSERT
        - Primary keys defined and values exist: UPSERT
        - Primary keys defined but no existing values: INSERT

        Args:
            update_df: DataFrame containing the data to upsert.
            report: In-progress report of the enclosing transaction. Exact,
                Python-computed insert/update counts are deposited here as a
                fallback, overwritten by the DuckLake-measured values
                (``ducklake_table_changes``) when a real catalog is attached.

        Returns:
            True if fact table updated successfully, False on error.
        """
        try:
            # Préparation des données pour la fact table

            # Récupération des clés primaires depuis les métadonnées
            primary_keys = self._get_primary_key_columns()

            # Vérification que les clés primaires sont présentes dans le DataFrame
            missing_keys = [key for key in primary_keys if key not in update_df.columns]
            if missing_keys:
                self.logger.error(f"Primary keys missing in DataFrame: {missing_keys}")
                return False

            # Séparation ligne par ligne : nouvelles observations (INSERT) vs
            # observations
            # existantes dont les valeurs doivent être mises à jour (UPSERT).
            rows_to_insert, rows_to_update = self._split_dataframe_by_pk_existence(
                update_df, primary_keys
            )

            # Initialisation des nombres de données insérées et mises à jour
            total_inserted, total_updated = 0, 0

            # Insertion des données
            if len(rows_to_insert) > 0:
                total_inserted = self.data_mgr.insert_data(
                    rows_to_insert, use_batch=False
                )

            # Mise à jour des données
            if len(rows_to_update) > 0:
                _, total_updated = self.data_mgr.upsert_data(
                    rows_to_update, primary_keys, use_batch=False
                )

            # Valeurs exactes, calculées en Python : servent de repli tant que
            # _transaction n'a pas pu obtenir la mesure DuckLake réelle
            # (table_changes), qui les remplacera si elle est disponible.
            report.rows_inserted = total_inserted
            report.rows_updated = total_updated

            # Logging
            self.logger.info(
                f"Fact table (direct): {total_inserted} inserted, {total_updated}"
                f" updated"
            )
            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error updating fact table: {e}")
            return False

    # Méthode auxiliaire de la mise à jour par batch de la table des faits
    def _update_fact_table_batch(
        self, update_df: nw.DataFrame[Any], report: OperationReport
    ) -> bool:
        """Update fact table using batch processing for large datasets.

        Uses primary keys from metadata to determine INSERT vs UPSERT strategy:
        - No primary keys defined: Always INSERT
        - Primary keys defined and values exist: UPSERT
        - Primary keys defined but no existing values: INSERT

        Args:
            update_df: DataFrame containing the data to upsert in batches.
            report: In-progress report of the enclosing transaction. Exact,
                Python-computed insert/update counts are deposited here as a
                fallback, overwritten by the DuckLake-measured values
                (``ducklake_table_changes``) when a real catalog is attached.

        Returns:
            True if fact table updated successfully, False on error.
        """
        try:
            # Préparation des données pour la fact table

            # Récupération des clés primaires depuis les métadonnées
            primary_keys = self._get_primary_key_columns()

            # Vérification que les clés primaires sont présentes dans le DataFrame
            missing_keys = [key for key in primary_keys if key not in update_df.columns]
            if missing_keys:
                self.logger.error(f"Primary keys missing in DataFrame: {missing_keys}")
                return False

            # Séparation ligne par ligne : nouvelles observations (INSERT) vs
            # observations
            # existantes dont les valeurs doivent être mises à jour (UPSERT).
            rows_to_insert, rows_to_update = self._split_dataframe_by_pk_existence(
                update_df, primary_keys
            )

            # Initialisation du nombre total de données mises à jour et insérées
            total_inserted, total_updated = 0, 0

            # Insertion des nouvelles données
            if len(rows_to_insert) > 0:
                total_inserted = self.data_mgr.insert_data(
                    rows_to_insert, use_batch=True
                )

            # Mise à jour des données
            if len(rows_to_update) > 0:
                _, total_updated = self.data_mgr.upsert_data(
                    rows_to_update, primary_keys, use_batch=True
                )

            # Valeurs exactes, calculées en Python : servent de repli tant que
            # _transaction n'a pas pu obtenir la mesure DuckLake réelle
            # (table_changes), qui les remplacera si elle est disponible.
            report.rows_inserted = total_inserted
            report.rows_updated = total_updated

            # Logging
            self.logger.info(
                f"Fact table (batch): {total_inserted} inserted, {total_updated}"
                f" updated"
            )
            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error updating fact table in batch: {e}")
            return False

    # Méthode auxiliaire de séparation des lignes selon l'existence de leur clé primaire
    def _split_dataframe_by_pk_existence(
        self, df: nw.DataFrame[Any], primary_keys: list[str]
    ) -> tuple[nw.DataFrame[Any], nw.DataFrame[Any]]:
        """Split a DataFrame into rows to INSERT and rows to UPDATE
        based on primary key existence.

        For each row in the DataFrame, checks whether its primary key combination
        already
        exists in the fact_table. Rows whose PKs are absent from the fact_table should
        be
        INSERTed; rows whose PKs are already present should be UPSERTed (updated).

        Args:
            df: DataFrame containing data with primary key columns.
            primary_keys: List of primary key column names.

        Returns:
            Tuple of (rows_to_insert, rows_to_update) where:
            - rows_to_insert: DataFrame rows whose PKs do not exist in fact_table.
            - rows_to_update: DataFrame rows whose PKs already exist in fact_table.

        Example:
            >>> df = pd.DataFrame({'id': [1, 2, 3], 'value': ['a', 'b', 'c']})
            >>> to_insert, to_update = updater._split_dataframe_by_pk_existence(df,
            ['id'])
            >>> # to_update contient les lignes dont l'id existe déjà en base
        """
        # DataFrame vide : aucune ligne à traiter
        if len(df) == 0:
            return df.clone(), df.clone()

        try:
            # Vérification de l'existence de la fact_table
            if not self._table_exists("fact_table"):
                # Aucune fact_table : toutes les lignes sont à insérer
                return df.clone(), df.head(0)

            # Nom qualifié de la table des faits
            fact_table = self._qualified("fact_table")

            # Vérification que la fact_table n'est pas vide
            _rc = self.conn.execute(f"SELECT COUNT(*) FROM {fact_table}").fetchone()
            count = _rc[0] if _rc is not None else 0
            if count == 0:
                return df.clone(), df.head(0)

            # Construction de la condition de jointure corrélée sur les clés primaires
            # composites :
            # chaque colonne du DataFrame (alias upd) est comparée à la fact_table
            # (alias f).
            conditions = " AND ".join(
                f"f.{quote_ident(key)} = upd.{quote_ident(key)}" for key in primary_keys
            )

            # Enregistrement dans DuckDB
            self.conn.register("_upd_split", df.to_arrow().select(df.columns))

            # Lignes dont les clés primaires existent déjà en base → à mettre à jour
            rows_to_update = nw.from_native(
                self.conn.execute(f"""
                SELECT upd.* FROM _upd_split upd
                WHERE EXISTS (
                    SELECT 1 FROM {fact_table} f
                    WHERE {conditions}
                )
            """).to_arrow_table(),
                eager_only=True,
            )

            # Lignes dont les clés primaires sont absentes de la base → à insérer
            rows_to_insert = nw.from_native(
                self.conn.execute(f"""
                SELECT upd.* FROM _upd_split upd
                WHERE NOT EXISTS (
                    SELECT 1 FROM {fact_table} f
                    WHERE {conditions}
                )
            """).to_arrow_table(),
                eager_only=True,
            )

            # Nettoyage de l'enregistrement temporaire
            self.conn.unregister("_upd_split")

            return rows_to_insert, rows_to_update

        except Exception as e:
            self.logger.error(
                f"Error splitting DataFrame by primary key existence: {e}"
            )
            # En cas d'erreur, on traite toutes les lignes comme des insertions
            return df.clone(), df.head(0)

    # Méthodes utilitaires
    # Méthode auxiliaire de nettoyage des données mises à jour
    def _get_cleaned_update_data(
        self,
        update_df: nw.DataFrame[Any],
        keep: Literal["any", "none", "first", "last"],
    ) -> nw.DataFrame[Any]:
        """Get deduplicated update data.

        Args:
            update_df: Original DataFrame with potential duplicates.
            keep: Strategy for keeping duplicates ('first', 'last', or False).

        Returns:
            DataFrame with duplicates removed according to keep strategy.
        """
        return remove_dataframe_duplicates(update_df, keep, self.logger, "update")

    # Méthode auxiliaire de suppression des doublons dans la base de données
    def _remove_database_duplicates(
        self, keep: Literal["any", "none", "first", "last"]
    ) -> bool:
        """Remove duplicate rows from the database fact table.

        Args:
            keep: Strategy for keeping duplicates ('first', 'last', or False).

        Returns:
            True if deduplication succeeded, False on error.
        """
        try:
            # Nom qualifié de la table des faits
            fact_table = self._qualified("fact_table")

            # Récupération du nombre initial de lignes
            _ri = self.conn.execute(f"SELECT COUNT(*) FROM {fact_table}").fetchone()
            initial_count = _ri[0] if _ri is not None else 0

            # Récupération des colonnes
            all_columns = [
                col[0] for col in self.conn.execute(f"DESCRIBE {fact_table}").fetchall()
            ]
            columns_to_check = [col for col in all_columns if col != "value"]

            if not columns_to_check:
                return True

            # Construction et exécution de la requête de suppression des doublons
            delete_query = build_database_duplicate_removal_query(
                columns_to_check, keep, fact_table
            )
            self.conn.execute(delete_query)

            # Calcul des lignes supprimées
            _final_row = self.conn.execute(
                f"SELECT COUNT(*) FROM {fact_table}"
            ).fetchone()
            final_count = _final_row[0] if _final_row is not None else 0
            removed_count = initial_count - final_count

            if removed_count > 0:
                # Logging
                self.logger.info(
                    f"Database duplicate removal: {removed_count} rows removed"
                )

            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error removing database duplicates: {e}")
            return False

    # ---------------------------------------------------------------------------
    # Gestion explicite des colonnes : ajout de colonnes de valeurs
    # ---------------------------------------------------------------------------

    # Méthode d'ajout explicite de colonnes de valeurs à partir d'un DataFrame
    def add_columns(
        self,
        df: IntoDataFrame,
        column_metadata: dict[str, dict[str, str]] | None = None,
        overwrite: bool = False,
        compact_after_update: bool = True,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> OperationReport:
        """
        Add value column(s) to the fact table from a DataFrame keyed by the
        primary keys, merging rows on those keys (outer merge).

        ``df`` must carry every primary key column plus one or more other columns,
        the values to add. Rows are matched on the primary keys:

        - key combination present in both: the row receives ``df``'s values;
        - key combination only in ``df``: a new row is **inserted** with the primary
          keys and ``df``'s columns, every other value column of the fact table
          left NULL;
        - key combination only in the fact table: the row keeps NULL in the new
          column(s) (and its previous values in overwritten ones).

        Implemented as ``ALTER TABLE ... ADD COLUMN`` for every genuinely new
        column, followed by a **single** ``UPDATE fact_table ... FROM <df> WHERE
        <primary keys match>`` covering all of them, then a single ``INSERT ...
        SELECT`` of the unmatched combinations sorted by ``cluster_by``, inside one
        DuckDB transaction: on any failure, neither the column(s), their
        ``metadata`` row(s) nor the inserted rows survive. An ``UPDATE`` that
        touches every row of the fact table is a complete copy-on-write rewrite;
        the row count about to be touched is logged before it runs, and
        ``rewrite_data_files`` is called afterwards with a low ``delete_threshold``
        to clear the resulting delete-tombstones.

        Args:
            df: DataFrame carrying every primary key column plus the value
                column(s) to add. Must be unique on the primary keys, with no null
                primary key.
            column_metadata: Per-added-column UI fields (``label``, ``unit``,
                ``display_format``, ``family``, ``description``,
                ``default_aggregation``, ``parent_name``, ``label_for``). Applies to
                newly added columns as well as to overwritten existing ones. A new
                column can be declared a label column this way, e.g.
                ``{'nc8_libelle': {'label_for': 'nc8'}}``.
            overwrite: Whether a column of ``df`` that already exists in the fact
                table may have its values replaced. Defaults to False: an existing
                column then raises ``ValueError`` instead.
            compact_after_update: Whether to run DuckLake compaction
                (``rewrite_data_files`` with a low ``delete_threshold``) after a
                successful commit. Defaults to True.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``). Ignored (skipped with a DEBUG
                log) on a connection with no real DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``.

        Returns:
            OperationReport: report describing what was actually added, updated
            (``rows_updated``) and inserted (``rows_inserted``).

        Raises:
            ValueError: If no primary key is defined on the fact table, if ``df``
                is missing a primary key column, if a primary key of ``df`` holds a
                null, if ``df`` is not unique on the primary keys, if ``df``
                carries no value column, if a value column
                already exists and ``overwrite`` is False, if ``column_metadata``
                is malformed, or if writing a code or label column violates the
                functional dependency of a declared code/label pair.

        Examples:
            >>> updater.add_columns(df_with_score)
            >>> updater.add_columns(df_with_score, overwrite=True)
            >>> # Diffusion explicite d'une valeur portée par une clé partielle :
            >>> keys = updater.get_key_combinations(['region', 'produit'])
            >>> df_partial = keys.to_polars().join(df_score, on=['region', 'produit'])
            >>> updater.add_columns(df_partial)
        """
        # Conversion vers narwhals dès le point d'entrée public
        df_nw = nw.from_native(df, eager_only=True)

        # Une clé primaire est requise pour cibler les lignes existantes
        primary_keys = self._get_primary_key_columns()
        if not primary_keys:
            raise ValueError(
                "No primary key is defined on the fact_table; add_columns requires"
                " one to match rows."
            )

        # df doit porter toutes les clés primaires
        missing_keys = [k for k in primary_keys if k not in df_nw.columns]
        if missing_keys:
            raise ValueError(
                f"df is missing primary key column(s): {sorted(missing_keys)}"
            )

        # Les clés primaires de df ne peuvent être nulles : elles créent des lignes
        null_keys = [k for k in primary_keys if df_nw[k].null_count() > 0]
        if null_keys:
            raise ValueError(f"df has null value(s) in primary key(s) {null_keys}")

        # df doit être unique sur les clés primaires : sinon la valeur affectée à
        # une même ligne de la fact table serait indéterminée (dernière ligne du
        # lot gagnante, silencieusement)
        if len(df_nw) != len(df_nw.unique(subset=primary_keys, keep="any")):
            raise ValueError(f"df must be unique on primary key(s) {primary_keys}")

        # Colonnes à ajouter : tout ce qui n'est pas une clé primaire
        new_columns = [c for c in df_nw.columns if c not in primary_keys]
        if not new_columns:
            raise ValueError(
                "df carries no value column to add besides the primary keys"
            )

        # Colonnes déjà existantes dans la fact table : erreur sauf overwrite=True
        existing_columns = set(self._get_fact_table_columns())
        already_existing = [c for c in new_columns if c in existing_columns]
        if already_existing and not overwrite:
            raise ValueError(
                f"Column(s) already exist in fact_table: {sorted(already_existing)}."
                " Pass overwrite=True to update their values instead."
            )
        columns_to_add = [c for c in new_columns if c not in already_existing]

        # Validation des métadonnées d'UI, restreintes aux colonnes concernées
        normalized_metadata = validate_column_metadata(column_metadata, new_columns)

        fact_table = self._qualified("fact_table")
        view_name = "_add_columns_src"

        # Transaction DuckDB unique : sur échec, ni les colonnes ni les lignes
        # metadata ne subsistent (ALTER TABLE est transactionnel dans DuckDB).
        try:
            with self._transaction(
                "add_columns",
                run_id=run_id,
                commit_message=commit_message,
                commit_info=commit_info,
            ) as report:
                report.columns_added = list(columns_to_add)
                # ALTER TABLE ... ADD COLUMN pour chaque colonne réellement nouvelle
                for column in columns_to_add:
                    sql_type = map_python_to_sql_type(df_nw.schema[column])
                    self.conn.execute(
                        f"ALTER TABLE {fact_table} ADD COLUMN {quote_ident(column)}"
                        f" {sql_type} DEFAULT NULL"
                    )
                    # Ligne de méta-données (is_primary_key=FALSE, is_categorical
                    # inféré du DataFrame)
                    self._add_column_to_metadata(column, df_nw)

                # Champs d'UI (nouvelles colonnes et colonnes overwrite confondues)
                for column in new_columns:
                    fields = normalized_metadata.get(column)
                    if fields:
                        self.update_column_metadata(column, **fields)

                # Enregistrement d'une vue temporaire pour la jointure
                self.conn.register(view_name, nw.to_native(df_nw))

                # Condition de jointure sur les clés primaires (identifiants qualifiés
                # et quotés des deux côtés)
                join_condition = " AND ".join(
                    f"f.{quote_ident(k)} = t.{quote_ident(k)}" for k in primary_keys
                )

                # Comptage des combinaisons de df sans correspondance en base, à
                # insérer après l'UPDATE
                _unmatched_row = self.conn.execute(f"""
                    SELECT COUNT(*) FROM {view_name} t
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {fact_table} f WHERE {join_condition}
                    )
                """).fetchone()
                unmatched_count = _unmatched_row[0] if _unmatched_row is not None else 0

                # Volume avant écriture : une UPDATE touchant toutes les lignes est une
                # réécriture complète de la table (copy-on-write).
                _total_row = self.conn.execute(
                    f"SELECT COUNT(*) FROM {fact_table}"
                ).fetchone()
                total_rows = _total_row[0] if _total_row is not None else 0
                rows_updated = len(df_nw) - unmatched_count
                # Valeur exacte, calculée en Python : sert de repli tant que
                # _transaction n'a pas pu obtenir la mesure DuckLake réelle
                # (table_changes), qui la remplacera si elle est disponible.
                report.rows_updated = rows_updated
                report.rows_inserted = unmatched_count
                self.logger.info(
                    f"add_columns: about to UPDATE {rows_updated} of {total_rows}"
                    f" fact_table row(s) (copy-on-write rewrite of touched files)"
                )

                # Un seul UPDATE ... FROM pour toutes les colonnes concernées (noms non
                # qualifiés côté gauche du SET : DuckDB rejette les qualificateurs de
                # table dans la clause SET d'un UPDATE ... FROM)
                set_clause = ", ".join(
                    f"{quote_ident(c)} = t.{quote_ident(c)}" for c in new_columns
                )
                self.conn.execute(f"""
                    UPDATE {fact_table} f
                    SET {set_clause}
                    FROM {view_name} t
                    WHERE {join_condition}
                """)

                # Insertion des combinaisons absentes de la base (fusion externe) :
                # les autres colonnes de valeur prennent NULL. Tri par cluster_by
                # pour préserver l'élagage par fichier.
                if unmatched_count > 0:
                    insert_columns = ", ".join(
                        quote_ident(c) for c in [*primary_keys, *new_columns]
                    )
                    order_clause = self.data_mgr._cluster_by_order_clause(
                        list(df_nw.columns)
                    )
                    self.conn.execute(f"""
                        INSERT INTO {fact_table} ({insert_columns})
                        SELECT {insert_columns} FROM {view_name} t
                        WHERE NOT EXISTS (
                            SELECT 1 FROM {fact_table} f WHERE {join_condition}
                        )
                        {order_clause}
                    """)
                    self.logger.info(
                        f"add_columns: inserted {unmatched_count} new key"
                        " combination(s) absent from fact_table"
                    )

                self.conn.execute(f"DROP VIEW {view_name}")

                # Dépendance fonctionnelle des colonnes de libellés, sur
                # l'état post-écriture, restreinte aux codes du lot. Couvre aussi le
                # cas d'une nouvelle colonne de libellés déclarée via
                # column_metadata={'x_libelle': {'label_for': 'x'}} : sa validation
                # structurelle a déjà eu lieu dans l'appel à update_column_metadata.
                self._check_value_label_dependencies(
                    df_nw, primary_keys, set(new_columns)
                )

                # dataset_metadata.updated_at
                self._touch_dataset_metadata()
        except Exception:
            # Nettoyage de la vue temporaire : elle survit au ROLLBACK, qui ne
            # porte que sur le catalogue.
            try:
                self.conn.execute(f"DROP VIEW IF EXISTS {view_name}")
            except Exception:
                pass
            raise

        # Nombre de lignes préexistantes de la base restées NULL : celles qu'aucune
        # ligne de df n'est venue mettre à jour
        rows_left_null = total_rows - rows_updated
        self.logger.info(
            f"add_columns: {rows_updated} row(s) updated, {rows_left_null}"
            f" fact_table row(s) left NULL (no match in df)"
        )

        # Compaction DuckLake optionnelle : un UPDATE massif laisse des fichiers de
        # suppression (tombstones) sur les anciennes versions des lignes touchées ;
        # delete_threshold bas car le taux de suppression peut approcher 100%.
        final_report = self.last_report
        assert final_report is not None  # posé par _transaction sur tout succès
        if compact_after_update:
            self.maintenance.compact(
                schema=self.schema, delete_threshold=0.05, report=final_report
            )

        self._invalidate_metadata_cache()
        self._finalize_report_after_write(final_report)
        self.logger.info(final_report.summary())
        return final_report

    # ---------------------------------------------------------------------------
    # Gestion explicite des colonnes de libellés
    # ---------------------------------------------------------------------------

    # Méthode de remplacement explicite des libellés de certains codes
    def update_value_labels(
        self,
        label_column: str,
        labels: IntoDataFrame,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
        compact_after_update: bool = True,
    ) -> OperationReport:
        """
        Replace the label of one or more codes of a code/label column pair .

        The one legitimate way to relabel a code: an upsert cannot do it, since the
        rows of that code already written under the old label would violate the
        functional dependency (``update_database``'s value-label check refuses
        exactly this and points here). Runs a single ``UPDATE fact_table SET
        <label_column> = t.<label_column> FROM <labels> t WHERE fact_table.<code> =
        t.<code>``, rewriting (copy-on-write) every row of the codes given, inside a
        single transaction: a change of *some* labels either all lands or none does.
        The displayed label is thus always the **current** one; earlier labels
        remain readable through DuckLake time travel.

        Args:
            label_column: Name of the label column to update. Must already have a
                ``label_for`` declared (via the build, ``add_columns`` or
                ``update_column_metadata``).
            labels: Narwhals-compatible DataFrame carrying exactly two columns: the
                code (under the code column's own name) and the new label (under
                ``label_column``'s name). Must be unique on the code column.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``). Ignored (skipped with a DEBUG
                log) on a connection with no real DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``.
            compact_after_update: Whether to run DuckLake compaction
                (``rewrite_data_files``) after a successful commit. Defaults to True.

        Returns:
            OperationReport: report describing the rows rewritten
            (``rows_updated``); codes of ``labels`` absent from the fact table are
            listed (sample) in ``report.warnings``, never inserted.

        Raises:
            ValueError: If ``label_column`` has no ``label_for`` declared, if
                ``labels`` does not carry exactly the code and label columns, if
                ``labels`` is not unique on the code column, or if the update would
                (still) violate the functional dependency code -> label.

        Examples:
            >>> import polars as pl
            >>> new_labels = pl.DataFrame({
            ...     'nc8': ['01012100'], 'nc8_libelle': ['Chevaux reproducteurs']})
            >>> report = updater.update_value_labels('nc8_libelle', new_labels)
        """
        # Conversion vers narwhals dès le point d'entrée public
        labels_nw = nw.from_native(labels, eager_only=True)

        # Résolution de la colonne de code associée à la colonne de libellés
        metadata_table = self._qualified("metadata")
        _row = self.conn.execute(
            f"SELECT label_for FROM {metadata_table} WHERE name = ?", [label_column]
        ).fetchone()
        code_column = _row[0] if _row is not None else None
        if code_column is None:
            raise ValueError(
                f"Column {label_column!r} has no label_for; it is not a value-label"
                " column"
            )

        # labels doit porter exactement le code et le libellé
        expected_columns = {code_column, label_column}
        if set(labels_nw.columns) != expected_columns:
            raise ValueError(
                f"labels must carry exactly the columns {sorted(expected_columns)},"
                f" got {sorted(labels_nw.columns)}"
            )

        # Unicité sur le code : sinon le libellé affecté à un même code serait
        # indéterminé (dernière ligne du lot gagnante, silencieusement)
        if len(labels_nw) != len(labels_nw.unique(subset=[code_column], keep="any")):
            raise ValueError(f"labels must be unique on {code_column!r}")

        fact_table = self._qualified("fact_table")
        view_name = "_update_value_labels_src"
        quoted_code = quote_ident(code_column)
        quoted_label = quote_ident(label_column)

        # Transaction DuckDB unique : sur échec, aucun libellé n'est modifié.
        try:
            with self._transaction(
                "update_value_labels",
                run_id=run_id,
                commit_message=commit_message,
                commit_info=commit_info,
            ) as report:
                # Enregistrement d'une vue temporaire pour la jointure
                self.conn.register(view_name, nw.to_native(labels_nw))

                # Journalisation, avant d'agir, du nombre de lignes sur le point
                # d'être réécrites (réécriture copy-on-write des fichiers concernés)
                _cnt_row = self.conn.execute(f"""
                    SELECT COUNT(*) FROM {fact_table}
                    WHERE {quoted_code} IN (SELECT {quoted_code} FROM {view_name})
                """).fetchone()
                rows_to_rewrite = _cnt_row[0] if _cnt_row is not None else 0
                self.logger.info(
                    f"update_value_labels: about to rewrite {rows_to_rewrite}"
                    f" fact_table row(s) for {len(labels_nw)} code(s)"
                )

                # Codes absents de la base : jamais insérés, seulement signalés
                absent_rows = self.conn.execute(f"""
                    SELECT DISTINCT {quoted_code} FROM {view_name}
                    WHERE {quoted_code} NOT IN (
                        SELECT {quoted_code} FROM {fact_table}
                    )
                    LIMIT 10
                """).fetchall()
                if absent_rows:
                    sample = [row[0] for row in absent_rows]
                    report.warnings.append(
                        f"{len(sample)} code(s) of labels absent from the fact"
                        f" table (showing up to 10, never inserted): {sample}"
                    )

                # UPDATE unique portant sur tous les codes du lot
                self.conn.execute(f"""
                    UPDATE {fact_table} f SET {quoted_label} = t.{quoted_label}
                    FROM {view_name} t
                    WHERE f.{quoted_code} = t.{quoted_code}
                """)
                # Valeur exacte, calculée en Python : sert de repli tant que
                # _transaction n'a pas pu obtenir la mesure DuckLake réelle
                # (table_changes), qui la remplacera si elle est disponible.
                report.rows_updated = rows_to_rewrite

                # Contrôle de dépendance après l'UPDATE (garde-fou : l'unicité sur le
                # code et la restriction aux codes du lot devraient déjà le garantir)
                check_value_label_dependency(
                    self.conn,
                    fact_table,
                    code_column,
                    label_column,
                    restrict_to=view_name,
                )

                self.conn.execute(f"DROP VIEW {view_name}")

                # dataset_metadata.updated_at
                self._touch_dataset_metadata()
        except Exception:
            # Nettoyage de la vue temporaire : elle survit au ROLLBACK, qui ne porte
            # que sur le catalogue.
            try:
                self.conn.execute(f"DROP VIEW IF EXISTS {view_name}")
            except Exception:
                pass
            raise

        # Compaction DuckLake optionnelle : l'UPDATE réécrit (copy-on-write) toutes
        # les lignes des codes concernés.
        final_report = self.last_report
        assert final_report is not None  # posé par _transaction sur tout succès
        if compact_after_update:
            self.maintenance.compact(schema=self.schema, report=final_report)

        self._invalidate_metadata_cache()
        self._finalize_report_after_write(final_report)
        self.logger.info(final_report.summary())
        return final_report

    # Méthode d'extraction des combinaisons de clés existantes en base
    def get_key_combinations(
        self, columns: list[str] | None = None
    ) -> nw.DataFrame[Any]:
        """
        Get distinct existing combinations of key column(s) from the fact table.

        A value carried by a
        partial key (e.g. ``(region, produit)``) is not automatically spread over
        a fuller key (e.g. ``(date, region, produit)``) by ``add_columns`` — that
        would be a different result set, denormalized. To broadcast deliberately,
        join the caller's partial DataFrame against the full key combinations
        returned here, then pass the joined DataFrame to ``add_columns``.

        Args:
            columns: Columns to project. Defaults to every primary key column.

        Returns:
            nw.DataFrame: Distinct combinations of ``columns`` present in the fact
            table, one row per combination (pyarrow backend: convert it with
            ``to_polars()``/``to_pandas()`` before joining a native DataFrame).

        Raises:
            ValueError: If no primary key is defined on the fact table and
                ``columns`` is None, or if ``columns`` references a column absent
                from the fact table.

        Examples:
            >>> keys = updater.get_key_combinations(['region', 'produit'])
            >>> df_partial = keys.to_polars().join(df_score, on=['region', 'produit'])
            >>> updater.add_columns(df_partial)
        """
        # Colonnes par défaut : toutes les clés primaires
        if columns is None:
            columns = self._get_primary_key_columns()
            if not columns:
                raise ValueError(
                    "No primary key is defined on the fact_table; pass columns"
                    " explicitly."
                )

        # Validation de l'existence des colonnes demandées
        existing_columns = set(self._get_fact_table_columns())
        unknown = [c for c in columns if c not in existing_columns]
        if unknown:
            raise ValueError(f"Unknown column(s) in fact_table: {sorted(unknown)}")

        # Création de la requête
        column_list = ", ".join(quote_ident(c) for c in columns)
        result = self.conn.execute(
            f"SELECT DISTINCT {column_list} FROM {self._qualified('fact_table')}"
        ).to_arrow_table()
        combinations: nw.DataFrame[Any] = nw.from_native(result, eager_only=True)
        return combinations

    # Méthodes publiques additionnelles
    # Méthode d'extraction du statut de la base de données
    def get_update_status(self) -> dict[str, Any]:
        """
        Get the status of the database update system.

        Returns:
            Dictionary containing system status information

        Example:
            >>> status = updater.get_update_status()
            >>> print(f"System health: {status['health_status']}")
        """
        try:
            status = {
                "timestamp": datetime.now(),
                "health_status": "unknown",
                # Toujours 0 : les transactions sont portées par DuckDB
                # (BEGIN/COMMIT par opération) et ne font plus l'objet d'un suivi
                # applicatif. Clé conservée pour la stabilité du dictionnaire.
                "active_transactions": 0,
                "validation_enabled": self.enable_validation,
                "batch_size": self.batch_size,
                "max_workers": self.max_workers,
            }

            # Vérification de la santé de la base de données
            if self.auditor:
                health_check = self.auditor.get_quick_health_check()
                status["health_status"] = health_check.get("status", "unknown")
                status["database_info"] = health_check

            # Statistiques de la fact table
            if self._table_exists("fact_table"):
                table_stats = self.data_mgr.get_table_stats()
                status["fact_table_stats"] = table_stats

            return status

        except Exception as e:
            # Logging
            self.logger.error(f"Error getting update status: {e}")
            return {"error": str(e), "timestamp": datetime.now()}

    # Méthode d'optimisation de la base de données
    def optimize_database(self) -> bool:
        """
        Perform database optimization operations.

        Returns:
            True if optimization was successful

        Example:
            >>> success = updater.optimize_database()
            >>> if success:
            ...     print("Database optimized successfully")
        """
        try:
            # Optimisation de la fact table
            if not self.data_mgr.optimize_table():
                self.logger.warning("Failed to optimize fact table")

            # Suppression des colonnes entièrement nulles (références comprises)
            self._cleanup_null_only_columns()

            # Logging
            self.logger.info("Database optimization completed")
            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error during database optimization: {e}")
            return False
