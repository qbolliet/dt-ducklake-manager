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

# Import des gestionnaires
from .._internal.managers.base import BaseSchemaManager
from .._internal.managers.data import DataManager
from .._internal.managers.dimension import DimensionManager
from .._internal.managers.transaction import TransactionManager, TransactionOperation
from ..maintenance.auditor import DatabaseAuditor, IssueSeverity, ValidationLevel

# Import des utilitaires
from ..utils.sql import (
    build_database_duplicate_removal_query,
    remove_dataframe_duplicates,
)

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe de mise à jour d'une base de données
class DatabaseUpdater(BaseSchemaManager):
    """
    Refactored database updater using the new modular architecture.

    Provides atomic, transactional updates to DuckDB databases with proper
    validation, error recovery, and state consistency. Uses specialized managers
    for different aspects of database operations.

    Attributes:
        dimension_mgr (DimensionManager): Manages dimension table operations
        data_mgr (DataManager): Manages fact table operations
        transaction_mgr (TransactionManager): Manages transactions and rollback
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
        ducklake_catalog_alias: str = "db",
        schema: str = "main",
    ):
        """
        Initialize the refactored database updater.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            categorical_threshold: Threshold for determining categorical variables.
            log_filename: Path to log file.
            max_workers: Maximum number of parallel workers.
            batch_size: Size of batches for processing.
            enable_validation: Whether to enable pre/post operation validation.
            ducklake_catalog_alias: Alias used in the DuckLake ATTACH statement.
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
        )

        # Initialisation des gestionnaires spécialisés
        self.dimension_mgr = DimensionManager(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
            max_workers=max_workers,
            schema=schema,
        )

        self.data_mgr = DataManager(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
            batch_size=batch_size,
            schema=schema,
        )

        self.transaction_mgr = TransactionManager(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
            ducklake_catalog_alias=ducklake_catalog_alias,
            schema=schema,
        )

        self.auditor = (
            DatabaseAuditor(
                connection=connection,
                categorical_threshold=categorical_threshold,
                log_filename=log_filename,
                schema=schema,
            )
            if enable_validation
            else None
        )

        # Configuration
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_validation = enable_validation

        # Configuration DuckLake pour les appels de maintenance (compaction).
        # L'alias du catalogue ; le schéma est porté par self.schema (classe de base).
        self.ducklake_catalog_alias = ducklake_catalog_alias

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
    ) -> bool:
        """
        Update the entire database with new data using atomic operations.

        Args:
            update_df: DataFrame containing update data
            check_duplicates_db: Whether to check duplicates in existing database
            check_duplicates_update: Whether to check duplicates in update DataFrame
            keep: Which duplicates to keep when removing duplicates
            use_batch_processing: Whether to use batch processing for large datasets
            use_transaction: Whether to use database transactions
            compact_after_update: Whether to run DuckLake compaction (merge small delta
                files and rewrite delete files) immediately after a successful update.
                Adds write latency but keeps read performance optimal. Defaults to True.

        Returns:
            True if update was successful, False otherwise

        Examples:
            >>> success = updater.update_database(new_data_df)
            >>> success = updater.update_database(new_data_df,
            compact_after_update=True)
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

        # Logging
        self.logger.info(
            f"Starting database update with {len(update_df)} rows (transaction:"
            f"{use_transaction})"
        )

        # Utilisation de la transaction pour la mise à jour de la base de données
        if use_transaction:
            return self._update_database_transactional(
                update_df,
                check_duplicates_db,
                check_duplicates_update,
                keep,
                use_batch_processing,
                compact_after_update,
            )
        # Sinon mise à jour directe
        else:
            return self._update_database_direct(
                update_df,
                check_duplicates_db,
                check_duplicates_update,
                keep,
                use_batch_processing,
                compact_after_update,
            )

    # Méthode de mise à jour de la base de données de manière transactionnelle
    def _update_database_transactional(
        self,
        update_df: nw.DataFrame[Any],
        check_duplicates_db: bool,
        check_duplicates_update: bool,
        keep: Literal["any", "none", "first", "last"],
        use_batch_processing: bool,
        compact_after_update: bool,
    ) -> bool:
        """Perform transactional database update with validation and rollback.

        Executes the update within a transaction, allowing rollback on failure.
        Steps: preprocess → duplicate removal → metadata → fact table → dimensions.

        Args:
            update_df: DataFrame containing the update data.
            check_duplicates_db: Whether to check and remove duplicates in database.
            check_duplicates_update: Whether to check and remove duplicates in update
            data.
            keep: Duplicate handling strategy ('first', 'last', or False).
            use_batch_processing: Whether to use batch processing for large datasets.
            compact_after_update: Whether to run DuckLake compaction after commit.

        Returns:
            True if update succeeded and committed, False otherwise.
        """

        # Début de la transaction
        tx_id = self.transaction_mgr.begin_transaction(
            "Database update with validation and rollback"
        )

        try:
            # Étape 1: Suppression des doublons dans les données de mise à jour
            if check_duplicates_update:
                # Initialisation de l'opération de transaction
                operation = TransactionOperation(
                    operation_type="preprocess",
                    operation_func=self._remove_update_duplicates,
                    operation_args=(update_df, keep),
                    description="Remove duplicates from update data",
                )

                # Annulation de la transaction si l'opération ne peut être ajoutée
                if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False

                # Annulation de la transaction si l'opération ne peut être exécutée
                if not self.transaction_mgr.execute_operation(tx_id):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False

                # Récupération des données nettoyées
                update_df = self._get_cleaned_update_data(update_df, keep)

            # Étape 2: Suppression des doublons dans la base de données
            if check_duplicates_db:
                # Initialisation de l'opération de transacti
                operation = TransactionOperation(
                    operation_type="cleanup",
                    operation_func=self._remove_database_duplicates,
                    operation_args=(keep,),
                    rollback_func=self._restore_database_state,
                    description="Remove duplicates from existing database",
                )

                # Annulation de la transaction si l'opération ne peut être ajoutée
                if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False

                # Annulation de la transaction si l'opération ne peut être exécutée
                if not self.transaction_mgr.execute_operation(tx_id):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False

            # Création d'un savepoint avant les modifications majeures
            self.transaction_mgr.create_savepoint(tx_id, "before_major_updates")

            # Étape 3: Mise à jour des métadonnées
            # Initialisation de l'opération de transaction
            operation = TransactionOperation(
                operation_type="metadata_update",
                operation_func=self._update_metadata_safe,
                operation_args=(update_df,),
                rollback_func=self._rollback_metadata_changes,
                description="Update metadata table",
            )

            # Annulation de la transaction si l'opération ne peut être ajoutée
            if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Annulation de la transaction si l'opération ne peut être exécutée
            if not self.transaction_mgr.execute_operation(tx_id):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Étape 4: Mise à jour de la table de faits (avant les dimensions pour
            # refléter l'état actuel)
            # Mise à jour en batch si spécifié
            if use_batch_processing and len(update_df) > self.batch_size:
                # Initialisation de l'opération de transaction
                operation = TransactionOperation(
                    operation_type="fact_update_batch",
                    operation_func=self._update_fact_table_batch,
                    operation_args=(update_df,),
                    rollback_func=self._rollback_fact_changes,
                    description="Update fact table (batch processing)",
                )
            else:
                # Initialisation de l'opération de transaction
                operation = TransactionOperation(
                    operation_type="fact_update_direct",
                    operation_func=self._update_fact_table_direct,
                    operation_args=(update_df,),
                    rollback_func=self._rollback_fact_changes,
                    description="Update fact table (direct)",
                )

            # Annulation de la transaction si l'opération ne peut être ajoutée
            if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Annulation de la transaction si l'opération ne peut être exécutée
            if not self.transaction_mgr.execute_operation(tx_id):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Étape 5: Mise à jour des tables de dimensions (après fact table pour
            # refléter les données actuelles)
            # Initialisation de l'opération de transaction
            operation = TransactionOperation(
                operation_type="dimension_update",
                operation_func=self._update_dimensions_safe,
                operation_args=(update_df,),
                rollback_func=self._rollback_dimension_changes,
                description="Update dimension tables",
            )

            # Annulation de la transaction si l'opération ne peut être ajoutée
            if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Annulation de la transaction si l'opération ne peut être exécutée
            if not self.transaction_mgr.execute_operation(tx_id):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Étape 5b: Nettoyage des entrées orphelines dans les tables de dimension
            cleanup_operation = TransactionOperation(
                operation_type="dimension_cleanup",
                operation_func=self.dimension_mgr.cleanup_orphaned_dimension_entries,
                operation_args=(),
                description="Clean orphaned dimension entries",
            )

            if not self.transaction_mgr.add_operation(
                tx_id, **cleanup_operation.__dict__
            ):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            if not self.transaction_mgr.execute_operation(tx_id):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Validation post-update
            if self.enable_validation and self.auditor:
                # Validation de la base de données
                validation_report = self.auditor.validate_database(
                    ValidationLevel.STANDARD
                )

                # Vérification des problèmes critiques
                if validation_report.get_critical_issues_count() > 0:
                    # Logging
                    self.logger.error(
                        "Critical issues found after update, rolling back"
                    )
                    # Annulation de la transaction
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False

            # Commit de la transaction
            if self.transaction_mgr.commit_transaction(tx_id):
                # Logging
                self.logger.info("Database update completed successfully")
                # Invalidation du cache
                self._invalidate_metadata_cache()
                # Compaction DuckLake optionnelle après commit (fusion des petits
                # fichiers delta)
                if compact_after_update:
                    self._run_ducklake_compaction()
                return True
            else:
                # Logging
                self.logger.error("Failed to commit transaction")
                return False

        except Exception as e:
            # Logging
            self.logger.error(f"Error during transactional update: {e}")
            # Annulation de la transaction
            self.transaction_mgr.rollback_transaction(tx_id)
            return False

    # Méthode auxiliaire de mise à jour directe de la base de données
    def _update_database_direct(
        self,
        update_df: nw.DataFrame[Any],
        check_duplicates_db: bool,
        check_duplicates_update: bool,
        keep: Literal["any", "none", "first", "last"],
        use_batch_processing: bool,
        compact_after_update: bool,
    ) -> bool:
        """Perform direct database update without transaction wrapping.

        Suitable for simple updates where rollback capability is not needed.
        Faster but no automatic rollback on partial failure.

        Args:
            update_df: DataFrame containing the update data.
            check_duplicates_db: Whether to check and remove duplicates in database.
            check_duplicates_update: Whether to check and remove duplicates in update
            data.
            keep: Duplicate handling strategy ('first', 'last', or False).
            index_config: Optional index configuration dictionary.
            use_batch_processing: Whether to use batch processing for large datasets.

        Returns:
            True if update completed successfully, False otherwise.
        """
        try:
            # Suppression des doublons dans les données de mise à jour
            if check_duplicates_update:
                update_df = remove_dataframe_duplicates(
                    update_df, keep, self.logger, "update"
                )

            # Suppression des doublons dans la base de données
            if check_duplicates_db:
                self._remove_database_duplicates(keep)

            # Mise à jour des métadonnées
            if not self._update_metadata_safe(update_df):
                return False

            # Mise à jour de la table de faits (avant les dimensions pour refléter
            # l'état actuel)
            # Mise à jour par batch si spécifié
            if use_batch_processing and len(update_df) > self.batch_size:
                if not self._update_fact_table_batch(update_df):
                    return False
            else:
                if not self._update_fact_table_direct(update_df):
                    return False

            # Mise à jour des tables de dimensions (après fact table pour refléter les
            # données actuelles)
            if not self._update_dimensions_safe(update_df):
                return False

            # Nettoyage des entrées orphelines dans les tables de dimension
            self.dimension_mgr.cleanup_orphaned_dimension_entries()

            # Nettoyage final
            self._cleanup_orphaned_data()

            # Logging
            self.logger.info("Database update completed successfully (direct mode)")
            # Invalidation du cache des méta-données
            self._invalidate_metadata_cache()
            # Compaction DuckLake optionnelle (fusion des petits fichiers delta)
            if compact_after_update:
                self._run_ducklake_compaction()
            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error during direct update: {e}")
            return False

    # Méthodes de mise à jour sécurisées
    # Méthode auxiliaire de mise à jour des méta-données
    def _update_metadata_safe(self, update_df: nw.DataFrame[Any]) -> bool:
        """Safely update metadata table with type conflict resolution.

        Args:
            update_df: DataFrame whose columns may require metadata updates.

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
                self._resolve_type_conflicts(col, update_df, current_metadata)

            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error updating metadata: {e}")
            return False

    # Méthode auxiliaire de mise à jour des tables de dimension
    def _update_dimensions_safe(self, update_df: nw.DataFrame[Any]) -> bool:
        """Safely update dimension tables with categorical threshold checks.

        Handles conversion between categorical and non-categorical status
        based on unique value counts relative to the threshold.

        Args:
            update_df: DataFrame containing potential dimension updates.

        Returns:
            True if dimensions updated successfully, False on error.
        """
        try:
            # Chargement des métadonnées
            current_metadata = self._load_current_metadata()

            # Classification des colonnes par statut catégoriel via filtrage narwhals
            cat_names = current_metadata.filter(nw.col("is_categorical"))[
                "name"
            ].to_list()
            # Colonnes non-catégorielles de type String (ex-'object' pandas) :
            # candidates à conversion
            non_cat_string_names = current_metadata.filter(
                (~nw.col("is_categorical")) & (nw.col("python_type") == "String")
            )["name"].to_list()

            categorical_columns = {
                col: update_df[col] for col in cat_names if col in update_df.columns
            }
            non_categorical_columns = {
                col: update_df[col]
                for col in non_cat_string_names
                if col in update_df.columns
            }

            # Mise à jour des dimensions existantes
            if categorical_columns:
                results = self.dimension_mgr.batch_update_dimensions(
                    categorical_columns,
                    use_parallel=(
                        len(categorical_columns) > 1 and self.max_workers > 1
                    ),
                )

                # Vérification des résultats
                for col_name, added_count in results.items():
                    if added_count < 0:  # Erreur
                        # Logging
                        self.logger.error(f"Failed to update dimension for {col_name}")
                        return False

            # Vérification des conversions vers catégoriel pour les colonnes
            # non-catégorielles
            # La vérification porte sur l'état GLOBAL de fact_table après l'upsert, et
            # non
            # sur le seul update_df : un petit update_df pourrait avoir < seuil valeurs
            # uniques
            # pour une colonne qui en compte > seuil dans la base entière.
            for col_name in non_categorical_columns.keys():
                # Récupération des valeurs distinctes dans fact_table (état post-upsert)
                db_values_series = nw.from_native(
                    self.conn.execute(
                        f"SELECT DISTINCT {col_name} FROM"
                        f" {self._qualified('fact_table')} WHERE {col_name}"
                        f" IS NOT NULL"
                    ).pl()[col_name],
                    series_only=True,
                )
                # Vérification du seuil sur l'ensemble complet des valeurs
                if self._check_categorical_threshold(db_values_series):
                    # Conversion en catégoriel (les valeurs DB sont passées pour créer
                    # la dim table)
                    if not self.dimension_mgr.convert_to_categorical(
                        col_name, db_values_series
                    ):
                        # Logging
                        self.logger.warning(
                            f"Failed to convert {col_name} to categorical"
                        )

            # Vérification des conversions vers non-catégoriel pour les colonnes
            # catégorielles
            # La vérification porte sur le nombre d'entrées dans la table de dimension
            # (état
            # post-batch_update_dimensions), et non sur les seules valeurs de update_df.
            for col_name in categorical_columns.keys():
                # Comptage des entrées dans la table de dimension après
                # batch_update_dimensions
                _rd = self.conn.execute(
                    f"SELECT COUNT(*) FROM {self._qualified(f'dim_{col_name}')}"
                ).fetchone()
                n_dim_entries = _rd[0] if _rd is not None else 0
                # Vérification du seuil (garde contre un seuil non défini)
                if (
                    self.categorical_threshold is not None
                    and n_dim_entries > self.categorical_threshold
                ):
                    # Conversion en non-catégoriel
                    if not self.dimension_mgr.convert_to_non_categorical(col_name):
                        # Logging
                        self.logger.warning(
                            f"Failed to convert {col_name} to non-categorical"
                        )

            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error updating dimensions: {e}")
            return False

    # Méthode auxiliaire de mise à jour directe de la table des faits
    def _update_fact_table_direct(self, update_df: nw.DataFrame[Any]) -> bool:
        """Update fact table directly without batch processing.

        Uses primary keys from metadata to determine INSERT vs UPSERT strategy:
        - No primary keys defined: Always INSERT
        - Primary keys defined and values exist: UPSERT
        - Primary keys defined but no existing values: INSERT

        Args:
            update_df: DataFrame containing the data to upsert.

        Returns:
            True if fact table updated successfully, False on error.
        """
        try:
            # Préparation des données pour la fact table
            prepared_df = self._prepare_dataframe_for_fact_table(update_df)

            # Récupération des clés primaires depuis les métadonnées
            primary_keys = self._get_primary_key_columns()

            # Vérification que les clés primaires sont présentes dans le DataFrame
            missing_keys = [
                key for key in primary_keys if key not in prepared_df.columns
            ]
            if missing_keys:
                self.logger.error(f"Primary keys missing in DataFrame: {missing_keys}")
                return False

            # Séparation ligne par ligne : nouvelles observations (INSERT) vs
            # observations
            # existantes dont les valeurs doivent être mises à jour (UPSERT).
            rows_to_insert, rows_to_update = self._split_dataframe_by_pk_existence(
                prepared_df, primary_keys
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
    def _update_fact_table_batch(self, update_df: nw.DataFrame[Any]) -> bool:
        """Update fact table using batch processing for large datasets.

        Uses primary keys from metadata to determine INSERT vs UPSERT strategy:
        - No primary keys defined: Always INSERT
        - Primary keys defined and values exist: UPSERT
        - Primary keys defined but no existing values: INSERT

        Args:
            update_df: DataFrame containing the data to upsert in batches.

        Returns:
            True if fact table updated successfully, False on error.
        """
        try:
            # Préparation des données pour la fact table
            prepared_df = self._prepare_dataframe_for_fact_table(update_df)

            # Récupération des clés primaires depuis les métadonnées
            primary_keys = self._get_primary_key_columns()

            # Vérification que les clés primaires sont présentes dans le DataFrame
            missing_keys = [
                key for key in primary_keys if key not in prepared_df.columns
            ]
            if missing_keys:
                self.logger.error(f"Primary keys missing in DataFrame: {missing_keys}")
                return False

            # Séparation ligne par ligne : nouvelles observations (INSERT) vs
            # observations
            # existantes dont les valeurs doivent être mises à jour (UPSERT).
            rows_to_insert, rows_to_update = self._split_dataframe_by_pk_existence(
                prepared_df, primary_keys
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
            conditions = " AND ".join([f"f.{key} = upd.{key}" for key in primary_keys])

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
            """).pl(),
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
            """).pl(),
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

    # Méthode auxiliaire de compaction DuckLake
    def _run_ducklake_compaction(self, fact_table: str = "fact_table") -> None:
        """Trigger DuckLake compaction on the fact table after a successful update.

        Merges small adjacent Parquet delta files and rewrites delete files to
        maintain optimal read performance. Failures are non-fatal: a warning is
        logged and execution continues normally.

        The catalog alias and schema are read from ``self.ducklake_catalog_alias``
        and ``self.schema``, which can be set at construction time.

        Args:
            fact_table: Name of the fact table to compact. Defaults to ``'fact_table'``.

        Examples:
            >>> updater._run_ducklake_compaction()
            >>> updater._run_ducklake_compaction('my_fact_table')
        """
        # Extraction des alias et du schéma
        alias = self.ducklake_catalog_alias
        schema = self.schema
        try:
            # Fusion des petits fichiers delta adjacents
            # Note : les table functions DuckLake sont enregistrées dans le catalogue
            # mémoire
            # (où l'extension est chargée), pas dans le catalogue attaché.
            self.conn.execute(
                f"CALL ducklake_merge_adjacent_files('{alias}', '{fact_table}', schema"
                f":= '{schema}')"
            )
            # Réécriture des fichiers de suppression (delete files) pour optimiser les
            # lectures
            self.conn.execute(
                f"CALL ducklake_rewrite_data_files('{alias}', '{fact_table}', schema"
                f":= '{schema}')"
            )
            self.logger.info(f"Compaction DuckLake is finished for '{fact_table}'")
        except Exception as e:
            # Erreur non bloquante : la compaction est une optimisation, pas une étape
            # critique
            self.logger.warning(f"Compaction DuckLake failed : {e}")

    # Méthodes de rollback
    # Méthode auxiliaire de rollback des changements de métadonnées
    def _rollback_metadata_changes(self) -> bool:
        """Rollback metadata changes by invalidating cache.

        Returns:
            True if rollback succeeded, False on error.
        """
        try:
            # Invalidation du cache pour forcer le rechargement
            self._invalidate_metadata_cache()
            # Logging
            self.logger.info("Metadata changes rolled back")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error rolling back metadata changes: {e}")
            return False

    # Méthode auxiliaire de rollback des changements de dimensions
    def _rollback_dimension_changes(self) -> bool:
        """Rollback dimension table changes (handled by DuckDB transaction).

        Returns:
            True if rollback succeeded, False on error.
        """
        try:
            # Les changements de dimension sont gérés par la transaction DuckDB
            self.logger.info("Dimension changes rolled back")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error rolling back dimension changes: {e}")
            return False

    # Méthode auxiliaire de rollback des changements de la fact table.
    def _rollback_fact_changes(self) -> bool:
        """Rollback fact table changes (handled by DuckDB transaction).

        Returns:
            True if rollback succeeded, False on error.
        """
        try:
            # Les changements de fact table sont gérés par la transaction DuckDB
            # Logging
            self.logger.info("Fact table changes rolled back")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error rolling back fact table changes: {e}")
            return False

    # Méthode auxiliaire de rollback des changements d'index.
    def _rollback_index_changes(self) -> bool:
        """Rollback index changes (handled by DuckDB transaction).

        Returns:
            True if rollback succeeded, False on error.
        """
        try:
            # Les changements d'index sont gérés par la transaction DuckDB
            # Logging
            self.logger.info("Index changes rolled back")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error rolling back index changes: {e}")
            return False

    # Méthodes utilitaires
    # Méthode auxiliaire de préparation du jeu de données pour la table des faits
    def _prepare_dataframe_for_fact_table(
        self, df: nw.DataFrame[Any]
    ) -> nw.DataFrame[Any]:
        """Prepare DataFrame for fact table insertion.

        Converts categorical columns to their dimension table values
        and handles unmapped values.

        Args:
            df: Original DataFrame to prepare.

        Returns:
            Prepared DataFrame with categorical columns mapped to dimension values.
        """
        try:
            # Clonage du DataFrame (narwhals)
            prepared_df = df.clone()

            # Chargement des métadonnées
            current_metadata = self._load_current_metadata()

            # Extraction des colonnes catégorielles via filtrage narwhals
            cat_names = current_metadata.filter(nw.col("is_categorical"))[
                "name"
            ].to_list()

            # Conversion des colonnes catégorielles : label → valeur entière de
            # dimension
            for col_name in cat_names:
                if col_name not in prepared_df.columns:
                    continue

                # Mise à jour préalable de la table de dimension avec les nouvelles
                # valeurs
                self.dimension_mgr.update_dimension_values(
                    col_name, prepared_df[col_name]
                )

                # Récupération du mapping label → value (nw.DataFrame)
                mapping_df = self.dimension_mgr.get_dimension_mapping(col_name)

                if mapping_df is not None and len(mapping_df) > 0:
                    label_to_value = dict(
                        zip(
                            mapping_df["label"].to_list(), mapping_df["value"].to_list()
                        )
                    )
                    # Sauvegarde de l'ordre des colonnes
                    original_columns = prepared_df.columns
                    # Comptage des NULL avant le join : un NULL d'origine doit rester
                    # NULL dans la fact_table (pas de pendant dans la dim_*) ; seuls
                    # les NULL apparus *après* le join correspondent à des labels non
                    # mappés (anomalie data qualité à signaler).
                    pre_join_null_count = int(prepared_df[col_name].is_null().sum())
                    # Jointure narwhals pour remplacer les labels par leurs IDs
                    # Création du DataFrame de mapping avec le même backend que
                    # prepared_df
                    # pour éviter une incompatibilité lors du join narwhals.
                    dim_nw = nw.from_dict(
                        {
                            "_lbl_": list(label_to_value.keys()),
                            "_val_": list(label_to_value.values()),
                        },
                        backend=nw.get_native_namespace(prepared_df),
                    )
                    prepared_df = (
                        prepared_df.rename({col_name: "_lbl_"})
                        .join(dim_nw, on="_lbl_", how="left")
                        .drop("_lbl_")
                        .rename({"_val_": col_name})
                        .select(original_columns)
                    )

                    # Diagnostic des valeurs non mappées : différence entre les NULL
                    # post-join et les NULL d'entrée. Les NULL légitimes sont laissés
                    # tels quels — ils ne doivent jamais avoir d'ID dans la dim_*.
                    post_join_null_count = int(prepared_df[col_name].is_null().sum())
                    unmapped_count = post_join_null_count - pre_join_null_count
                    if unmapped_count > 0:
                        self.logger.warning(
                            f"Found {unmapped_count} unmapped (non-null) values in"
                            f" {col_name}; they will be stored as NULL in fact_table"
                        )

            return prepared_df

        except Exception as e:
            # Logging
            self.logger.error(f"Error preparing DataFrame for fact table: {e}")
            return df

    # Méthode auxiliaire de suppression des doublons des données de mise à jour
    def _remove_update_duplicates(
        self,
        update_df: nw.DataFrame[Any],
        keep: Literal["any", "none", "first", "last"],
    ) -> bool:
        """Remove duplicates from update DataFrame.

        Args:
            update_df: DataFrame to deduplicate.
            keep: Strategy for keeping duplicates ('first', 'last', or False).

        Returns:
            True if deduplication succeeded, False on error.
        """
        try:
            # Cette méthode modifie le DataFrame en place via la référence
            _ = remove_dataframe_duplicates(update_df, keep, self.logger, "update")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error removing update duplicates: {e}")
            return False

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

    # Méthode auxuliaire de restoration de la base de données
    def _restore_database_state(self) -> bool:
        """Restore database state (placeholder - handled by DuckDB transaction).

        Returns:
            True (actual restoration handled by database transaction rollback).
        """
        # Cette méthode est un placeholder pour la restauration d'état
        # En pratique, cela serait géré par la transaction DuckDB
        self.logger.info("Database state restoration handled by transaction")
        return True

    # Méthode auxiliaire de nettoyage des données orphelines
    def _cleanup_orphaned_data(self) -> None:
        """Clean up orphaned data after update operations.

        Removes orphaned dimension entries and drops null-only columns.
        """
        try:
            # Nettoyage des entrées orphelines dans les dimensions
            removed_counts = self.dimension_mgr.cleanup_orphaned_dimension_entries()

            if removed_counts:
                self.logger.info(
                    f"Cleaned orphaned dimension entries: {removed_counts}"
                )

            # Suppression des colonnes ne contenant que des nulles
            null_only_columns = self._get_null_only_columns()
            if null_only_columns:
                dropped_columns = self.data_mgr.drop_columns(null_only_columns)
                if dropped_columns:
                    self.logger.info(f"Dropped null-only columns: {dropped_columns}")

        except Exception as e:
            self.logger.error(f"Error cleaning orphaned data: {e}")

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
                "active_transactions": 0,
                "validation_enabled": self.enable_validation,
                "batch_size": self.batch_size,
                "max_workers": self.max_workers,
            }

            # Vérification des transactions actives
            active_txs = self.transaction_mgr.list_active_transactions()
            status["active_transactions"] = len(active_txs)

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

    # Méthode de validation de l'état de la base de données
    def validate_database_state(
        self, validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Any:
        """
        Validate the current state of the database.

        Args:
            validation_level: Level of validation to perform

        Returns:
            ValidationReport from the auditor

        Example:
            >>> report = updater.validate_database_state(ValidationLevel.COMPREHENSIVE)
            >>> if report.get_critical_issues_count() > 0:
            ...     print("Critical issues detected!")
        """
        # Vérification qu'un auditeur est renseigné
        if not self.auditor:
            # Logging
            self.logger.warning("Validation disabled - no auditor available")
            return None

        return self.auditor.validate_database(validation_level)

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

            # Nettoyage des données orphelines
            self._cleanup_orphaned_data()

            # Nettoyage des anciennes transactions
            cleaned_txs = self.transaction_mgr.cleanup_old_transactions()
            if cleaned_txs > 0:
                self.logger.info(f"Cleaned up {cleaned_txs} old transactions")

            # Logging
            self.logger.info("Database optimization completed")
            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error during database optimization: {e}")
            return False
