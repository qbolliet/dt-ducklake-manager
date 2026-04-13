# Importation des modules
# Modules de base
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# DuckDB
import duckdb
import narwhals as nw

# Import du gestionnaire de base
from .base import BaseSchemaManager

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe de gestion des opérations sur la table d
class DimensionManager(BaseSchemaManager):
    """
    Manages dimension table operations and categorical variable conversions.

    Handles creation, updates, and deletion of dimension tables while maintaining
    referential integrity with the fact table. Provides thread-safe operations
    for parallel dimension updates.

    Attributes:
        max_workers (int): Maximum number of parallel workers for dimension operations
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        categorical_threshold: int | None = 50,
        log_filename: os.PathLike | None = None,
        max_workers: int = 4,
    ):
        """
        Initialize the dimension manager.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            categorical_threshold: Threshold for determining categorical variables.
            log_filename: Path to log file.
            max_workers: Maximum number of parallel workers.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> dim_mgr = DimensionManager(conn, max_workers=8)
        """
        # Initialisation du parent
        super().__init__(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
        )

        # Configuration pour le traitement parallèle
        self.max_workers = max_workers

        # Verrou pour les opérations thread-safe sur les dimensions
        self._dimension_lock = threading.RLock()

    # Méthode de validation de l'opération
    def validate_operation(self, operation_type: str, **kwargs: Any) -> bool:
        """
        Validate dimension operations before execution.

        Args:
            operation_type: Type of operation ('create', 'update', 'delete', 'convert')
            **kwargs: Operation-specific parameters

        Returns:
            True if operation is valid
        """
        # Distinction de la bonne méthode de validation suivant l'opération
        if operation_type == "create":
            return self._validate_create_dimension(**kwargs)
        elif operation_type == "update":
            return self._validate_update_dimension(**kwargs)
        elif operation_type == "delete":
            return self._validate_delete_dimension(**kwargs)
        elif operation_type == "convert":
            return self._validate_convert_dimension(**kwargs)
        else:
            # Logging
            self.logger.warning(f"Unknown operation type: {operation_type}")
            return False

    # Méthode de validation de la création de dimensions
    def _validate_create_dimension(
        self, column_name: str, values: nw.Series[Any], **kwargs: Any
    ) -> bool:
        """Validate dimension creation parameters."""
        # Validation du nom de colonne
        if not column_name or column_name.strip() == "":
            # Logging
            self.logger.error("Column name cannot be empty")
            return False
        # Validation des valeurs
        if values is None or len(values) == 0:
            # Logging
            self.logger.error("Values cannot be empty")
            return False

        return True

    # Méthode de validation de la mise à jour de dimensions
    def _validate_update_dimension(
        self, column_name: str, values: nw.Series[Any], **kwargs: Any
    ) -> bool:
        """Validate dimension update parameters."""
        return self._validate_create_dimension(column_name, values, **kwargs)

    # Méthode de validation de la suppresion de dimension
    def _validate_delete_dimension(self, column_name: str, **kwargs: Any) -> bool:
        """Validate dimension deletion parameters."""
        if not column_name or column_name.strip() == "":
            self.logger.error("Column name cannot be empty")
            return False

        return True

    # Méthode de validation de la conversion d'une dimension
    def _validate_convert_dimension(
        self, column_name: str, values: nw.Series[Any], **kwargs: Any
    ) -> bool:
        """Validate dimension conversion parameters."""
        return self._validate_create_dimension(column_name, values, **kwargs)

    # Méthodes principales de gestion des dimensions
    # Méthode de création de la table de dimensions
    def create_dimension_table(self, dimension_name: str, values: nw.Series) -> bool:
        """
        Create a new dimension table with unique values.

        Args:
            dimension_name: Name of the dimension
            values: Series containing dimension values

        Returns:
            True if creation was successful

        Example:
            >>> values = pd.Series(['A', 'B', 'C', 'A'])
            >>> success = dim_mgr.create_dimension_table('category', values)
        """
        # Validation de l'opération de création
        if not self.validate_operation(
            "create", column_name=dimension_name, values=values
        ):
            return False

        with self._dimension_lock:
            try:
                # Nom de la table de dimension
                table_name = f"dim_{dimension_name}"

                # Création de la table
                # L'unicité de 'value' est garantie applicativement par l'utilisation de
                # enumerate() qui produit des indices strictement croissants.
                self.conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        value VARCHAR,
                        label VARCHAR
                    )
                """)

                # Préparation des données uniques (narwhals : drop_nulls + unique)
                unique_values = values.drop_nulls().unique().to_list()

                # Insertion des valeurs avec index automatique.
                # enumerate() garantit que chaque 'value' est unique → pas de conflit
                # possible.
                for idx, label in enumerate(unique_values):
                    self.conn.execute(
                        f"INSERT INTO {table_name} (value, label) VALUES (?, ?)",
                        [str(idx), str(label)],
                    )

                # Logging
                self.logger.info(
                    f"Created dimension table {table_name} with {len(unique_values)}"
                    f" unique values"
                )
                return True

            except Exception as e:
                # Logging
                self.logger.error(
                    f"Failed to create dimension table for {dimension_name}: {e}"
                )
                return False

    # Méthode de mise à jour d'une table de dimension
    def update_dimension_values(self, dimension_name: str, values: nw.Series) -> int:
        """
        Update dimension table with new values.

        Args:
            dimension_name: Name of the dimension
            values: Series containing new values

        Returns:
            Number of new values added

        Example:
            >>> new_values = pd.Series(['D', 'E', 'A'])  # A already exists
            >>> added_count = dim_mgr.update_dimension_values('category', new_values)
            >>> print(f"Added {added_count} new values")
        """
        # Validation de l'opération de mise à jour
        if not self.validate_operation(
            "update", column_name=dimension_name, values=values
        ):
            return 0

        with self._dimension_lock:
            try:
                # Nom de la table de dimension
                table_name = f"dim_{dimension_name}"

                # Création de la table si elle n'existe pas
                if not self._table_exists(table_name):
                    if self.create_dimension_table(dimension_name, values):
                        return values.drop_nulls().n_unique()
                    else:
                        return 0

                # Récupération des valeurs existantes (narwhals via polars)
                dim_result = nw.from_native(
                    self.conn.execute(f"SELECT value, label FROM {table_name}").pl(),
                    eager_only=True,
                )
                existing_labels = (
                    set(dim_result["label"].to_list()) if len(dim_result) > 0 else set()
                )
                existing_values = (
                    set(dim_result["value"].to_list()) if len(dim_result) > 0 else set()
                )

                # Identification des nouvelles valeurs (narwhals Series)
                unique_labels = set(values.drop_nulls().unique().to_list())
                new_labels = unique_labels - existing_labels

                if not new_labels:
                    return 0

                # Construction des nouveaux index
                if len(existing_values) > 0:
                    max_value = max(int(v) for v in existing_values)
                    new_values = list(
                        range(max_value + 1, max_value + 1 + len(new_labels))
                    )
                else:
                    new_values = list(range(len(new_labels)))

                # Insertion des nouvelles valeurs.
                # La valeur numérique 'value' est calculée comme max_existant + offset,
                # ce qui garantit l'unicité sans contrainte DDL.
                values_added = 0
                for value, label in zip(new_values, new_labels):
                    try:
                        self.conn.execute(
                            f"INSERT INTO {table_name} (value, label) VALUES (?, ?)",
                            [str(value), str(label)],
                        )
                        values_added += 1
                    except Exception as e:
                        # Logging
                        self.logger.warning(
                            f"Cannot insert {value}:{label} into {table_name}: {e}"
                        )

                # Logging
                if values_added > 0:
                    self.logger.info(f"Added {values_added} new values to {table_name}")

                return values_added

            except Exception as e:
                # Logging
                self.logger.error(f"Failed to update dimension {dimension_name}: {e}")
                return 0

    # Méthode de suppression de la table de dimension
    def delete_dimension_table(self, dimension_name: str) -> bool:
        """
        Delete a dimension table completely.

        Args:
            dimension_name: Name of the dimension to delete

        Returns:
            True if deletion was successful

        Example:
            >>> success = dim_mgr.delete_dimension_table('old_category')
        """
        # Validation de l'opération
        if not self.validate_operation("delete", column_name=dimension_name):
            return False

        with self._dimension_lock:
            try:
                # Nom de la table de dimension
                table_name = f"dim_{dimension_name}"

                # Suppression de la table
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

                # Logging
                self.logger.info(f"Deleted dimension table {table_name}")
                return True

            except Exception as e:
                # Logging
                self.logger.error(
                    f"Failed to delete dimension table {dimension_name}: {e}"
                )
                return False

    # Méthodes de conversion entre catégoriel et non-catégoriel
    # Méthode de conversion en variable catégorielle
    def convert_to_categorical(self, col_name: str, values: nw.Series) -> bool:
        """
        Convert a non-categorical column to categorical.

        Args:
            col_name: Column name
            values: Series with column values

        Returns:
            True if conversion was successful

        Example:
            >>> values = df['status']  # Series with few unique values
            >>> success = dim_mgr.convert_to_categorical('status', values)
        """
        # Validation de l'opération
        if not self.validate_operation("convert", column_name=col_name, values=values):
            return False

        # Vérification que le seuil catégoriel est satisfait
        if not self._check_categorical_threshold(values):
            # Logging
            self.logger.warning(f"Column {col_name} exceeds categorical threshold")
            return False

        try:
            # Création de la table de dimension
            if not self.create_dimension_table(col_name, values):
                return False

            # Mise à jour du statut catégoriel dans les métadonnées
            self._update_categorical_status(col_name, True)

            # Conversion des valeurs dans la fact table
            if not self._convert_fact_table_dimension_mapping(
                col_name, values_to_labels=False
            ):
                # En cas d'échec, nettoyage
                self.delete_dimension_table(col_name)
                self._update_categorical_status(col_name, False)
                return False

            # Logging
            self.logger.info(f"Successfully converted {col_name} to categorical")
            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Failed to convert {col_name} to categorical: {e}")
            return False

    # Méthode de conversion en variable non-catégorielle
    def convert_to_non_categorical(self, col_name: str) -> bool:
        """
        Convert a categorical column to non-categorical.

        Args:
            col_name: Column name

        Returns:
            True if conversion was successful

        Example:
            >>> success = dim_mgr.convert_to_non_categorical('category')
        """
        # Validation de l'opération
        if not self.validate_operation("delete", column_name=col_name):
            return False

        try:
            # Conversion des valeurs dans la fact table (values -> labels)
            if not self._convert_fact_table_dimension_mapping(
                col_name, values_to_labels=True
            ):
                # Logging
                self.logger.error(f"Failed to convert fact table values for {col_name}")
                return False

            # Suppression de la table de dimension
            if not self.delete_dimension_table(col_name):
                return False

            # Mise à jour du statut dans les métadonnées
            self._update_categorical_status(col_name, False)

            # Logging
            self.logger.info(f"Successfully converted {col_name} to non-categorical")
            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Failed to convert {col_name} to non-categorical: {e}")
            return False

    # Méthodes de traitement parallèle
    # Méthode de traitement de la mise à jour des dimensions en parallèle
    def batch_update_dimensions(
        self, dimension_data: dict[str, nw.Series], use_parallel: bool = True
    ) -> dict[str, int]:
        """
        Update multiple dimensions in batch, optionally using parallel processing.

        Args:
            dimension_data: Dictionary mapping dimension names to their values
            use_parallel: Whether to use parallel processing

        Returns:
            Dictionary mapping dimension names to number of values added

        Example:
            >>> data = {'cat1': series1, 'cat2': series2}
            >>> results = dim_mgr.batch_update_dimensions(data, use_parallel=True)
        """
        # Initialisation des résultats
        results = {}

        if use_parallel and len(dimension_data) > 1 and self.max_workers > 1:
            # Traitement parallèle
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Soumission des tâches
                future_to_dimension = {
                    executor.submit(
                        self.update_dimension_values, dim_name, values
                    ): dim_name
                    for dim_name, values in dimension_data.items()
                }

                # Récupération des résultats
                for future in as_completed(future_to_dimension):
                    dimension = future_to_dimension[future]
                    try:
                        results[dimension] = future.result()
                    except Exception as e:
                        # Logging
                        self.logger.error(
                            f"Error during parallel update of {dimension}: {e}"
                        )
                        results[dimension] = 0
        else:
            # Traitement séquentiel
            for dim_name, values in dimension_data.items():
                results[dim_name] = self.update_dimension_values(dim_name, values)

        return results

    # Méthodes de nettoyage et maintenance
    # méthode de suppression des tables de dimension qui ne sont plus référencées dans
    # la table des faits
    def cleanup_orphaned_dimension_entries(
        self, dimension_name: str | None = None
    ) -> dict[str, int]:
        """
        Clean up dimension entries that are no longer referenced in fact table.

        Args:
            dimension_name: Specific dimension to clean (None for all)

        Returns:
            Dictionary mapping dimension names to number of entries removed

        Example:
            >>> removed = dim_mgr.cleanup_orphaned_dimension_entries()
            >>> print(f"Cleaned dimensions: {removed}")
        """
        # Initialisation du dictionnaire des suppressions
        removed_counts = {}

        # Détermination des dimensions à nettoyer
        if dimension_name:
            dimensions_to_clean = (
                [dimension_name] if self._is_dimension_column(dimension_name) else []
            )
        else:
            dimensions_to_clean = self._get_categorical_columns()

        for dim_col in dimensions_to_clean:
            try:
                # Nom de la table de dimension
                table_name = f"dim_{dim_col}"

                # Vérification de l'existence de la table
                if not self._table_exists(table_name):
                    continue

                # Vérification que la colonne existe dans la fact table
                if not self._column_exists(dim_col):
                    # Si la colonne n'existe plus, supprimer toute la dimension
                    self.delete_dimension_table(dim_col)
                    removed_counts[dim_col] = -1  # Indicateur de suppression complète
                    continue

                # Comptage initial
                count_before = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]

                # Suppression des entrées orphelines.
                # Cast explicite en VARCHAR pour garantir la compatibilité de types
                # entre
                # dim_*.value (VARCHAR) et fact_table.{dim_col} (type quelconque)
                cleanup_query = f"""
                    DELETE FROM {table_name}
                    WHERE value NOT IN (
                        SELECT DISTINCT CAST({dim_col} AS VARCHAR)
                        FROM fact_table
                        WHERE {dim_col} IS NOT NULL
                    )
                """
                self.conn.execute(cleanup_query)

                # Comptage final
                count_after = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]

                # Comptage des suppressions
                removed_count = count_before - count_after
                if removed_count > 0:
                    removed_counts[dim_col] = removed_count
                    # Logging
                    self.logger.info(
                        f"Removed {removed_count} orphaned entries from {table_name}"
                    )

            except Exception as e:
                # Logging
                self.logger.error(f"Error cleaning dimension {dim_col}: {e}")

        return removed_counts

    # Méthode d'extration de la correspondance valeur-label dans la table de dimension
    def get_dimension_mapping(self, dimension_name: str) -> nw.DataFrame | None:
        """
        Get the complete mapping for a dimension table.

        Args:
            dimension_name: Name of the dimension

        Returns:
            DataFrame with value-label mapping or None if not found

        Example:
            >>> mapping = dim_mgr.get_dimension_mapping('category')
            >>> print(mapping[['value', 'label']])
        """
        try:
            # Nom de la table de dimension
            table_name = f"dim_{dimension_name}"

            # Vérification de l'existence de la table
            if not self._table_exists(table_name):
                # Logging
                self.logger.warning(f"Dimension table {table_name} does not exist")
                return None

            return nw.from_native(
                self.conn.execute(
                    f"SELECT value, label FROM {table_name} ORDER BY value"
                ).pl(),
                eager_only=True,
            )

        except Exception as e:
            # Logging
            self.logger.error(
                f"Error getting dimension mapping for {dimension_name}: {e}"
            )
            return None

    # Méthodes privées pour la gestion des conversions
    def _convert_fact_table_dimension_mapping(
        self, col_name: str, values_to_labels: bool = True
    ) -> bool:
        """
        Convert between dimension values and labels in the fact table.

        Args:
            col_name: Column name
            values_to_labels: If True, converts values→labels; if False, converts
            labels→values

        Returns:
            True if conversion was successful
        """
        # Nom de la table de dimension
        table_name = f"dim_{col_name}"

        try:
            # Vérification de l'existence de la colonne
            if not self._column_exists(col_name):
                # Logging
                self.logger.error(f"Column {col_name} does not exist in fact table")
                return False

            # Récupération du mapping de la table de dimension (polars natif pour
            # DuckDB)
            dim_result_pl = self.conn.execute(
                f"SELECT value, label FROM {table_name}"
            ).pl()

            if len(dim_result_pl) == 0:
                # Logging
                self.logger.warning(f"Dimension table {table_name} is empty")
                return True  # Pas d'erreur si la table est vide

            # Création d'une vue temporaire pour le mapping
            self.conn.register("temp_dim_mapping", dim_result_pl)

            if values_to_labels:
                # Conversion values → labels (pour revenir aux données originales)
                update_query = f"""
                    UPDATE fact_table
                    SET {col_name} = (
                        SELECT label FROM temp_dim_mapping
                        WHERE temp_dim_mapping.value = fact_table.{col_name}
                    )
                    WHERE {col_name} IS NOT NULL
                """
                operation = "values to labels"
            else:
                # Conversion labels → values (pour utiliser les index de dimension)
                update_query = f"""
                    UPDATE fact_table
                    SET {col_name} = (
                        SELECT value FROM temp_dim_mapping
                        WHERE temp_dim_mapping.label = fact_table.{col_name}
                    )
                    WHERE {col_name} IS NOT NULL
                """
                operation = "labels to values"

            # Exécution de la mise à jour
            self.conn.execute(update_query)

            # Suppression de la vue temporaire
            self.conn.execute("DROP VIEW temp_dim_mapping")

            # Logging
            self.logger.info(f"Converted {operation} for {col_name} in fact table")
            return True

        except Exception as e:
            # Nettoyage en cas d'erreur
            try:
                self.conn.execute("DROP VIEW IF EXISTS temp_dim_mapping")
            except Exception:
                pass

            self.logger.error(f"Error converting dimension mapping for {col_name}: {e}")
            return False
