# Importation des modules
# Modules de base
import os
import warnings
from typing import Any

import narwhals as nw
from narwhals.typing import IntoDataFrame

# Module d'initialisation du logger
from ..utils.hierarchy import validate_hierarchy_forest
from ..utils.logger import _init_logger

# Utilitaires de traitement des données
from ..utils.types import (
    UI_METADATA_FIELDS,
    map_python_to_sql_type,
    validate_column_metadata,
)
from ..utils.value_labels import validate_value_labels


# Classe de création d'une base de données DuckDB avec :
# - Une "Fact table" : contenant les données, libellés d'origine compris
# - Une "Metadata table" : contenant les caractéristiques des variables de la fact table
# (libellé, type SQL, statut catégoriel, clé primaire)
class SchemaBuilder:
    """
    A class to automate the inference of the metadata and fact tables from a given
    DataFrame (pandas, polars, or narwhals-compatible).

    Categorical columns keep their original labels in the fact table — there is no
    dimension table and no synthetic code anywhere in the schema.

    Attributes:
        df (nw.DataFrame): The input dataset (converted to narwhals).
        categorical_threshold (int): Threshold below which a textual column is
            flagged as categorical, a UI-only piece of metadata inferred once, at
            build time.
        categorical_overrides (dict[str, bool]): Per-column forcing of the
            categorical status, independent of the threshold.
        primary_keys (list[str]): Logical primary key columns.
        logger (logging.Logger): Logger instance for tracking processing steps.
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
        log_filename: str | os.PathLike[str] | None = None,
    ) -> None:
        """
        Initialize the SchemaBuilder with a DataFrame and optional parameters.

        Args:
            df: The input dataset (pandas, polars, or narwhals-compatible).
            categorical_threshold (Optional[int]): Maximum number of unique values
                for a textual column to be flagged as categorical. When None
                (default), no column is inferred as categorical. The flag is pure UI
                metadata: it drives no storage decision, since the fact table always
                stores the original labels.
            primary_keys (List[str], optional): List of column names to use as primary
            key.
                Can be a single column or composite key. Defaults to None (no primary
                key).
                When None, deduplication will use all columns and a UserWarning is
                raised.
            categorical_overrides (Optional[Dict[str, bool]]): Per-column forcing of
                the categorical status, independent of the threshold. The status is
                never re-evaluated by a subsequent update anyway; on a live base it
                is corrected with ``update_column_metadata(col,
                is_categorical=...)``. Defaults to None.
            hierarchies (Optional[Dict[str, str]]): Column hierarchy declared as a
                mapping of child column name to parent column name (e.g.
                ``{'commune': 'departement', 'departement': 'region'}``). Written to
                ``metadata.parent_name``. Must agree with any ``parent_name`` also
                supplied through ``column_metadata``. Defaults to None.
            value_labels (Optional[Dict[str, str]]): Code/label column pairs,
                declared as a mapping of label column name to code column name (e.g.
                ``{'nc8_libelle': 'nc8'}``). Written to ``metadata.label_for``, carried
                by the label column. Must agree with any ``label_for`` also supplied
                through ``column_metadata``. Defaults to None.
            log_filename (os.PathLike, optional): Path to the log file. Defaults to
                a file named `schema_builder.log` in a logs directory.

        Raises:
            ValueError: If a primary key, a ``categorical_overrides`` key, a
                ``hierarchies``/``value_labels`` key or value does not exist in the
                DataFrame.

        Examples:
            >>> # Single primary key (with polars)
            >>> import polars as pl
            >>> df = pl.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
            >>> builder = SchemaBuilder(df, primary_keys=['id'])

            >>> # Composite primary key
            >>> builder = SchemaBuilder(df, primary_keys=['date', 'country',
            'indicator'])

            >>> # No primary key — raises UserWarning
            >>> builder = SchemaBuilder(df)

            >>> # Threshold disabled — no column is inferred as categorical
            >>> builder = SchemaBuilder(df, categorical_threshold=None,
            primary_keys=['id'])

            >>> # A column forced as categorical whatever its cardinality
            >>> builder = SchemaBuilder(df, categorical_threshold=50,
            ...     primary_keys=['id'], categorical_overrides={'city': True})

            >>> # A column hierarchy: commune -> departement -> region
            >>> builder = SchemaBuilder(df, categorical_threshold=50,
            ...     primary_keys=['id'],
            ...     hierarchies={'commune': 'departement', 'departement': 'region'})

            >>> # A code/label column pair: nc8_libelle restitutes nc8's label
            >>> builder = SchemaBuilder(df, categorical_threshold=50,
            ...     primary_keys=['id'], value_labels={'nc8_libelle': 'nc8'})
        """
        # Conversion vers narwhals
        self.df = nw.from_native(df, eager_only=True)
        # Initialisation du seuil en deçà duquel une colonne textuelle est signalée
        # comme catégorielle dans la table de méta-données
        self.categorical_threshold = categorical_threshold

        # Validation du forçage du statut catégoriel si spécifié
        if categorical_overrides:
            # Vérification de l'existence des colonnes visées
            unknown_cols = set(categorical_overrides) - set(self.df.columns)
            if unknown_cols:
                raise ValueError(
                    f"The following categorical_overrides columns do not exist in the"
                    f" DataFrame: {sorted(unknown_cols)}"
                )
        self.categorical_overrides = categorical_overrides or {}

        # Validation de la hiérarchie de colonnes si spécifiée : les colonnes
        # enfants et parentes doivent toutes exister dans le DataFrame. La
        # cohérence avec un éventuel parent_name fourni via column_metadata et la
        # détection de cycle sont vérifiées plus tard, dans create_metadata_table,
        # où les deux sources sont fusionnées.
        if hierarchies:
            # Vérification que l'ensemble des enfants est dans le jeu de données
            unknown_children = set(hierarchies) - set(self.df.columns)
            if unknown_children:
                raise ValueError(
                    f"The following hierarchies columns do not exist in the"
                    f" DataFrame: {sorted(unknown_children)}"
                )
            # Vérification que l'ensemble des parents est dans le jeu de données
            unknown_parents = set(hierarchies.values()) - set(self.df.columns)
            if unknown_parents:
                raise ValueError(
                    f"The following hierarchies parent columns do not exist in the"
                    f" DataFrame: {sorted(unknown_parents)}"
                )
        # Instanciation des hiérarchies
        self.hierarchies = dict(hierarchies) if hierarchies else {}

        # État de la hiérarchie résolue (colonne -> parente), renseigné par
        # _resolve_hierarchies et consommé par _resolve_value_labels (contrôle de
        # forme de la colonne de libellés : elle doit être hors de toute hiérarchie).
        self._hierarchy_parent_of: dict[str, str | None] = {}

        # Validation des colonnes de libellés si spécifiées : les colonnes de
        # libellés et de code doivent toutes exister dans le DataFrame. La cohérence
        # avec un éventuel label_for fourni via column_metadata est vérifiée plus tard,
        # dans create_metadata_table, où les deux sources sont fusionnées.
        if value_labels:
            # Vérification que l'ensemble des colonnes de libellés est dans le jeu de
            # données
            unknown_labels = set(value_labels) - set(self.df.columns)
            if unknown_labels:
                raise ValueError(
                    f"The following value_labels columns do not exist in the"
                    f" DataFrame: {sorted(unknown_labels)}"
                )
            # Vérification que l'ensemble des colonnes de code est dans le jeu de
            # données
            unknown_codes = set(value_labels.values()) - set(self.df.columns)
            if unknown_codes:
                raise ValueError(
                    f"The following value_labels target columns do not exist in the"
                    f" DataFrame: {sorted(unknown_codes)}"
                )
        # Instanciation des colonnes de libellés
        self.value_labels = dict(value_labels) if value_labels else {}

        # Mapping résolu (label -> code), renseigné par create_metadata_table via
        # _resolve_value_labels ; initialisé vide pour rester lisible avant tout appel
        # à create_metadata_table.
        self.value_labels_resolved: dict[str, str] = {}

        # Validation des clés primaires si spécifiées
        if primary_keys is not None and len(primary_keys) > 0:
            # Vérification de l'existence des colonnes
            missing_cols = set(primary_keys) - set(self.df.columns)
            if missing_cols:
                raise ValueError(
                    f"The following primary key columns do not exist in the DataFrame:"
                    f"{missing_cols}"
                )

            self.primary_keys = primary_keys
        else:
            self.primary_keys = []

        # Avertissement en l'absence de clés primaires : la déduplication portera sur
        # l'ensemble des colonnes, ce qui peut produire des résultats inattendus lorsque
        # plusieurs colonnes de valeurs (value, lower_bound, upper_bound…) diffèrent
        # entre deux lignes qui représentent pourtant le même enregistrement.
        if not self.primary_keys:
            warnings.warn(
                "No primary key specified. The deduplication will apply to "
                "all columns. Pass primary_keys=['col1', ...] to "
                "avoid this warning.",
                UserWarning,
                stacklevel=2,
            )

        # Initialisation du logger nommé.
        # Chemin par défaut centralisé dans utils.logger : <cwd>/logs/<name>.log.
        self.logger = _init_logger(filename=log_filename, name="schema_builder")

        # Logging de la validation des clés primaires (après initialisation du logger)
        if len(self.primary_keys) > 0:
            self.logger.info(
                f"Primary keys validated successfully: {self.primary_keys}"
            )

    # Méthode de résolution et de validation de la hiérarchie de colonnes
    def _resolve_hierarchies(
        self, column_metadata_norm: dict[str, dict[str, str | None]]
    ) -> set[str]:
        """
        Merge, validate and inject the column hierarchy into ``column_metadata_norm``.

        Combines ``self.hierarchies`` (the dedicated constructor parameter) with any
        ``parent_name`` supplied per column through ``column_metadata``, checks both
        sources agree where they overlap, validates that every parent column exists
        in the DataFrame and that the resulting graph is a forest (no cycle), then
        writes the resolved ``parent_name`` back into ``column_metadata_norm`` so it
        is picked up like any other UI field.

        Args:
            column_metadata_norm (dict[str, dict[str, str | None]]): Normalized
                ``column_metadata`` mapping, mutated in place with the resolved
                ``parent_name`` for every column that has one.

        Returns:
            set[str]: Every column participating in a hierarchy (as a child, a
            parent, or both), which must end up categorical.

        Raises:
            ValueError: If ``self.hierarchies`` and ``column_metadata`` disagree on a
                column's parent, if a parent column does not exist in the DataFrame,
                or if the merged graph contains a cycle.
        """
        # Extraction du parent_name éventuellement déclaré via column_metadata.
        # L'affectation via l'opérateur walrus permet à mypy de rétrécir le type
        # de "parent" à str (fields.get retourne str | None).
        from_column_metadata: dict[str, str] = {
            col: parent
            for col, fields in column_metadata_norm.items()
            if (parent := fields.get("parent_name")) is not None
        }

        # Fusion des deux sources avec contrôle de cohérence
        parent_of: dict[str, str] = {}
        for col, parent in self.hierarchies.items():
            parent_of[col] = parent
        for col, parent in from_column_metadata.items():
            if col in parent_of and parent_of[col] != parent:
                raise ValueError(
                    f"Conflicting parent_name for column {col!r}: hierarchies says"
                    f" {parent_of[col]!r}, column_metadata says {parent!r}"
                )
            parent_of[col] = parent

        # Rien à valider en l'absence de toute hiérarchie déclarée
        if not parent_of:
            return set()

        # Vérification de l'existence des colonnes parentes (celles issues de
        # column_metadata n'ont pas encore été validées ; celles de hierarchies l'ont
        # été à l'initialisation).
        unknown_parents = set(parent_of.values()) - set(self.df.columns)
        if unknown_parents:
            raise ValueError(
                f"The following parent_name columns do not exist in the DataFrame:"
                f" {sorted(unknown_parents)}"
            )

        # Détection de cycle : le graphe des parent_name doit être une forêt
        validate_hierarchy_forest(parent_of)

        # Injection du parent_name résolu dans column_metadata_norm : lu ensuite
        # comme n'importe quel autre champ d'UI par la boucle de create_metadata_table
        for col, parent in parent_of.items():
            column_metadata_norm.setdefault(col, {})["parent_name"] = parent

        # Conservation de l'état de la hiérarchie résolue, consommé par
        # _resolve_value_labels (les labels doivent être en dehors de la hiérarchie)
        self._hierarchy_parent_of = dict(parent_of)

        # Colonnes participant à la hiérarchie (enfants et parents), qui doivent
        # toutes être catégorielles
        return set(parent_of.keys()) | set(parent_of.values())

    # Méthode de résolution et de validation des colonnes de libellés
    def _resolve_value_labels(
        self, column_metadata_norm: dict[str, dict[str, str | None]]
    ) -> dict[str, str]:
        """
        Merge, validate and inject the code/label column pairs into
        ``column_metadata_norm``.

        Combines ``self.value_labels`` (the dedicated constructor parameter) with any
        ``label_for`` supplied per column through ``column_metadata``, checks both
        sources agree where they overlap, validates the label_for structural checks
        (target exists and differs from the label column, no chaining, label column
        is VARCHAR / not a primary key / outside any hierarchy), then writes the
        resolved ``label_for`` back into ``column_metadata_norm`` so it is picked up
        like any other UI field. Must run after :meth:`_resolve_hierarchies`, whose
        result (``self._hierarchy_parent_of``) the label column shape check depends
        on. Has no effect on ``is_categorical``.

        Args:
            column_metadata_norm (dict[str, dict[str, str | None]]): Normalized
                ``column_metadata`` mapping, mutated in place with the resolved
                ``label_for`` for every label column.

        Returns:
            dict[str, str]: The resolved mapping of label column name to code column
            name. The functional dependency itself is not checked here: it needs
            actual row data and is checked by the caller
            (``DuckLakeTablesBuilder.build_schema``) on the deduplicated DataFrame.

        Raises:
            ValueError: If ``self.value_labels`` and ``column_metadata`` disagree on a
                column's target, if a target column does not exist in the DataFrame,
                or if any of the label_for structural checks is violated.
        """
        # Extraction du label_for éventuellement déclaré via column_metadata.
        from_column_metadata: dict[str, str] = {
            col: code
            for col, fields in column_metadata_norm.items()
            if (code := fields.get("label_for")) is not None
        }

        # Fusion des deux sources avec contrôle de cohérence
        label_for: dict[str, str] = {}
        for label_col, code_col in self.value_labels.items():
            label_for[label_col] = code_col
        for label_col, code_col in from_column_metadata.items():
            if label_col in label_for and label_for[label_col] != code_col:
                raise ValueError(
                    f"Conflicting label_for for column {label_col!r}: value_labels"
                    f" says {label_for[label_col]!r}, column_metadata says"
                    f" {code_col!r}"
                )
            label_for[label_col] = code_col

        # Rien à valider en l'absence de toute colonne de libellés déclarée
        if not label_for:
            return {}

        # Vérification de l'existence des colonnes cibles (celles issues de
        # column_metadata n'ont pas encore été validées ; celles de value_labels
        # l'ont été à l'initialisation).
        unknown_codes = set(label_for.values()) - set(self.df.columns)
        if unknown_codes:
            raise ValueError(
                f"The following label_for target columns do not exist in the"
                f" DataFrame: {sorted(unknown_codes)}"
            )

        # Types SQL de toutes les colonnes du DataFrame
        columns_sql_types = {
            col: map_python_to_sql_type(self.df.schema[col]) for col in self.df.columns
        }

        # Validation des contrôles structurels de label_for
        validate_value_labels(
            label_for, columns_sql_types, self.primary_keys, self._hierarchy_parent_of
        )

        # Injection du label_for résolu dans column_metadata_norm : lu ensuite comme
        # n'importe quel autre champ d'UI par la boucle de create_metadata_table
        for label_col, code_col in label_for.items():
            column_metadata_norm.setdefault(label_col, {})["label_for"] = code_col

        return label_for

    # Méthode inférant le type des colonnes du jeu de données
    def create_metadata_table(
        self,
        column_labels: dict[str, str] | None | None = None,
        column_metadata: dict[str, dict[str, str]] | None = None,
    ) -> nw.DataFrame[Any]:
        """
        Automatically infer metadata for the DataFrame's columns, including SQL types,
        labels, the categorical UI flag and the producer-owned UI fields.

        The ``is_categorical`` flag is inferred once from the distinct non-null count
        of textual columns and can be forced per column through
        ``categorical_overrides``. No later update re-evaluates it.

        The UI fields ``unit``, ``display_format``, ``family``, ``description`` and
        ``default_aggregation`` are all VARCHAR, nullable, and default to ``None``.
        They are supplied per column through ``column_metadata``. ``parent_name``
        (also VARCHAR, nullable) declares a column hierarchy: it can be
        supplied here, through the constructor's ``hierarchies`` parameter, or both
        (in which case they must agree). Any column participating in a hierarchy is
        forced categorical, with a warning, if it is not already. ``label_for``
        declares a code/label column pair: it can be supplied here, through
        the constructor's ``value_labels`` parameter, or both (in which case they
        must agree). Only the label_for structural checks are performed here (the
        functional dependency itself needs row data and is checked by the caller);
        it has no effect on ``is_categorical``.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.
            column_metadata (dict, optional): Mapping of column name to a
                sub-dictionary with optional keys ``label``, ``unit``,
                ``display_format`` (a d3-format string), ``family``, ``description``,
                ``default_aggregation`` (one of ``SUM``, ``AVG``, ``MIN``,
                ``MAX``, ``COUNT``, ``MEDIAN``, ``MODE``, validated on write),
                ``parent_name`` (the parent column of a column hierarchy) and
                ``label_for`` (the code column a label column restitutes). When
                a label is given both here and in ``column_labels``, this mapping
                wins. Defaults to None.

        Returns:
            nw.DataFrame: A DataFrame containing metadata for each column in the input
            dataset.

        Raises:
            ValueError: If ``column_metadata`` references a column absent from the
                DataFrame, carries an unknown sub-dictionary key, supplies an
                invalid ``default_aggregation``, disagrees with ``hierarchies``/
                ``value_labels`` on a column's parent/target, references a
                ``parent_name``/``label_for`` column absent from the DataFrame, if the
                resulting ``parent_name`` graph contains a cycle, or if a
                ``label_for`` pair violates the label_for structural checks.

        Examples:
            >>> metadata = builder.create_metadata_table()
            >>> sorted(metadata.columns)  # doctest: +NORMALIZE_WHITESPACE
            ['default_aggregation', 'description', 'display_format', 'family',
             'is_categorical', 'is_primary_key', 'label', 'label_for', 'name',
             'parent_name', 'sql_type', 'unit']
        """
        # Validation et normalisation des métadonnées d'UI fournies par le producteur
        column_metadata_norm = validate_column_metadata(
            column_metadata, list(self.df.columns)
        )

        # Fusion des deux sources de hiérarchie (paramètre dédié hierarchies et clé
        # parent_name de column_metadata), avec contrôle de cohérence, validation de
        # l'existence des colonnes parentes et détection de cycle (forêt). Le
        # parent_name résolu est injecté dans column_metadata_norm, d'où le champ
        # sera lu comme n'importe quel autre champ d'UI par la boucle ci-dessous.
        hierarchy_columns = self._resolve_hierarchies(column_metadata_norm)

        # Fusion des deux sources de colonnes de libellés (paramètre dédié
        # value_labels et clé label_for de column_metadata), avec contrôle de
        # cohérence et validation des contrôles structurels (cible, chaînage, forme
        # de colonne). Le label_for résolu est injecté dans column_metadata_norm,
        # d'où le champ sera lu comme n'importe quel autre champ d'UI par la boucle
        # ci-dessous. Exposé sur self.value_labels_resolved pour le contrôle de la
        # dépendance fonctionnelle, effectué par l'appelant sur les données du
        # DataFrame.
        self.value_labels_resolved = self._resolve_value_labels(column_metadata_norm)

        # Initialisation de la liste des méta-données
        list_metadata = []
        # Parcours des colonnes du jeu de données
        for col in self.df.columns:
            # Extraction du type de la colonne (narwhals DType)
            dtype_obj = self.df.schema[col]
            # Sous-dictionnaire d'UI éventuel pour la colonne courante
            ui_fields = column_metadata_norm.get(col, {})
            # Résolution du libellé : column_metadata prime sur column_labels, qui
            # prime sur le libellé dérivé du nom technique.
            if "label" in ui_fields and ui_fields["label"] is not None:
                label = ui_fields["label"]
            elif column_labels is not None and col in column_labels:
                label = column_labels[col]
            else:
                label = col.replace("_", " ").title()

            # Initialisation des méta-données associées à la colonne
            metadata = {
                "name": col,
                "label": label,
                "sql_type": map_python_to_sql_type(dtype_obj),
                "is_categorical": False,
                "is_primary_key": col in self.primary_keys,
            }
            # Champs d'UI : valeur fournie ou NULL par défaut
            for field in UI_METADATA_FIELDS:
                metadata[field] = ui_fields.get(field)

            # Logging
            self.logger.info(f"Successfully extracted meta-data from column '{col}'")

            # Colonne textuelle : inférence du statut catégoriel par le seuil
            if isinstance(dtype_obj, nw.String | nw.Categorical | nw.Enum):
                # Calcul du nombre de modalités, valeurs manquantes exclues (même
                # règle que pour les colonnes ajoutées plus tard)
                n_modalities = self.df[col].drop_nulls().n_unique()
                # Vérification du seuil : si categorical_threshold vaut None, aucune
                # colonne n'est traitée comme catégorielle.
                if self.categorical_threshold is not None:
                    if n_modalities <= self.categorical_threshold:
                        # Mise à jour du type de la variable
                        metadata["is_categorical"] = True
                        # Logging
                        self.logger.info(
                            f"The column '{col}' is of type 'String' and the number of"
                            f" modalities {n_modalities} satisfies the categorical"
                            f" threshold criteria {self.categorical_threshold}"
                        )
                    else:
                        # Logging
                        self.logger.warning(
                            f"The column '{col}' is of type 'String' but the number of"
                            f" modalities {n_modalities} exceeds the categorical"
                            f" threshold criteria {self.categorical_threshold}"
                        )

            # Forçage explicite du statut catégoriel, indépendant du seuil
            if col in self.categorical_overrides:
                metadata["is_categorical"] = self.categorical_overrides[col]
                # Logging
                self.logger.info(
                    f"The categorical status of column '{col}' is forced to"
                    f" {metadata['is_categorical']} by categorical_overrides"
                )

            # Forçage catégoriel des colonnes appartenant à une hiérarchie :
            # invariant non négociable, il l'emporte donc sur un categorical_overrides
            # explicite à False pour la même colonne.
            if col in hierarchy_columns and not metadata["is_categorical"]:
                # Forçage du statut catégoriel
                metadata["is_categorical"] = True
                # Warning indiquant la conversion
                warnings.warn(
                    f"Column {col!r} is part of a column hierarchy but is not"
                    f" categorical; forcing is_categorical=True",
                    UserWarning,
                    stacklevel=2,
                )
                # Logging
                self.logger.info(
                    f"Column '{col}' is part of a column hierarchy: forcing"
                    f" is_categorical=True"
                )

            # Ajout au dictionnaire
            list_metadata.append(metadata)

        # Transformation de la liste de dicts en dict de listes (format attendu par
        # nw.from_dict)
        keys = list(list_metadata[0].keys())
        col_oriented = {k: [d[k] for d in list_metadata] for k in keys}
        # Création du DataFrame de métadonnées via narwhals (même backend que self.df)
        self.df_metadata = nw.from_dict(
            col_oriented, backend=nw.get_native_namespace(self.df)
        ).sort("label")

        # Typage explicite des champs d'UI en VARCHAR : une colonne entièrement NULL
        # serait sinon inférée en type ``Null`` par le backend, incompatible avec le
        # DDL VARCHAR de la table metadata.
        self.df_metadata = self.df_metadata.with_columns(
            nw.col(field).cast(nw.String) for field in UI_METADATA_FIELDS
        )

        # Logging
        self.logger.info("Successfully built the meta-data DataFrame")

        return self.df_metadata

    # Méthode créant la table des informations
    def create_fact_table(
        self, column_labels: dict[str, str] | None | None = None
    ) -> nw.DataFrame[Any]:
        """
        Return the fact table, i.e. the input dataset itself.

        Categorical columns keep their **original labels**: Parquet
        dictionary-encoding absorbs the storage cost, so no synthetic code and no
        dimension table are involved. The frame is copied rather than aliased so
        that later mutations of the fact table do not reach the input dataset.

        Args:
            column_labels (dict, optional): Accepted for signature parity with the
                other builders; unused here. Defaults to None.

        Returns:
            nw.DataFrame: The fact table, holding the input values verbatim.

        Examples:
            >>> fact_table = builder.create_fact_table()
            >>> fact_table['category'].to_list()
            ['A', 'B', 'A']
        """
        # Table des faits : copie du jeu de données d'entrée, sans substitution
        self.df_fact = self.df.clone()

        # Logging
        self.logger.info("Successfully built fact table")

        return self.df_fact

    # Méthode créant les différentes tables
    def build(
        self,
        column_labels: dict[str, str] | None | None = None,
        column_metadata: dict[str, dict[str, str]] | None = None,
    ) -> tuple[nw.DataFrame[Any], nw.DataFrame[Any]]:
        """
        Execute the full pipeline to create the metadata and fact tables.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.
            column_metadata (dict, optional): Per-column UI metadata forwarded to
                :meth:`create_metadata_table`. Defaults to None.

        Returns:
            tuple: A tuple containing the metadata DataFrame and the fact table
            DataFrame, both narwhals frames.

        Examples:
            >>> metadata, fact_table = builder.build()
            >>> len(metadata) == len(fact_table.columns)
            True
        """
        # Création de la table des méta-données
        _ = self.create_metadata_table(
            column_labels=column_labels, column_metadata=column_metadata
        )
        # Création de la table des faits
        _ = self.create_fact_table(column_labels=column_labels)

        return self.df_metadata, self.df_fact
