# Importation des modules
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from .sql import qualify_table, quote_ident, resolve_catalog

if TYPE_CHECKING:
    import duckdb

# Nombre maximal de codes fautifs listés dans un message d'erreur de dépendance
# fonctionnelle.
_MAX_FAULTY_CODES_LISTED = 10


# Fonction de validation structurelle des colonnes de libellés (cible valide,
# absence de chaînage, forme de la colonne de libellés)
def validate_value_labels(
    label_for: Mapping[str, str],
    columns: Mapping[str, str],
    primary_keys: Iterable[str],
    parent_of: Mapping[str, str | None],
) -> None:
    """
    Validate a label column -> code column mapping: target validity, no
    chaining, and label column shape.

    A code/label pair is declared by ``metadata.label_for``, carried by the
    **label** column and pointing at the **code** column. The mechanism mirrors
    ``metadata.parent_name`` (see :func:`~.hierarchy.validate_hierarchy_forest`):
    the child (here, the label column) points at its target, and several label
    columns may point at the same code (languages, short/long label).

    Args:
        label_for (Mapping[str, str]): Mapping of label column name to code column
            name, covering every declared pair (current state merged with any
            pending change).
        columns (Mapping[str, str]): Mapping of every fact table column name to
            its SQL type (e.g. ``{'nc8': 'VARCHAR', 'nc8_libelle': 'VARCHAR'}``).
        primary_keys (Iterable[str]): Names of the primary key columns.
        parent_of (Mapping[str, str | None]): Column hierarchy state (name ->
            parent column name, or ``None``), as read from
            ``metadata.parent_name``.

    Raises:
        ValueError: If, for any pair, the target code column does not exist or
            equals the label column (target validity); the target is itself a
            label column, i.e. chaining (no chaining); or the label column is
            not ``VARCHAR``, is a primary key, or participates in a column
            hierarchy (label column shape).

    Examples:
        >>> validate_value_labels(
        ...     {'nc8_libelle': 'nc8'},
        ...     {'nc8': 'VARCHAR', 'nc8_libelle': 'VARCHAR'},
        ...     primary_keys=[],
        ...     parent_of={},
        ... )
    """
    # Ensemble des clés primaires, pour un test d'appartenance rapide
    primary_key_set = set(primary_keys)
    # Colonnes parentes (au sens hiérarchie de colonnes) : une colonne de libellés
    # ne peut appartenir à aucune hiérarchie, ni comme enfant ni comme parente.
    hierarchy_parents = set(parent_of.values())

    for label_col, code_col in label_for.items():
        # Cible valide : la cible existe et diffère de la colonne de libellés
        if code_col not in columns:
            raise ValueError(
                f"label_for target {code_col!r} of column {label_col!r} does not"
                " exist in the fact table"
            )
        if code_col == label_col:
            raise ValueError(f"Column {label_col!r} cannot be its own label_for target")

        # Absence de chaînage : la cible n'est pas elle-même une colonne de
        # libellés
        if code_col in label_for:
            raise ValueError(
                f"label_for target {code_col!r} of column {label_col!r} is itself a"
                " label column (chaining is not allowed)"
            )

        # Forme de la colonne de libellés : VARCHAR, non clé primaire, et hors
        # de toute hiérarchie
        if columns.get(label_col) != "VARCHAR":
            raise ValueError(
                f"label_for column {label_col!r} must be VARCHAR, got"
                f" {columns.get(label_col)!r}"
            )
        if label_col in primary_key_set:
            raise ValueError(f"label_for column {label_col!r} cannot be a primary key")
        if parent_of.get(label_col) is not None or label_col in hierarchy_parents:
            raise ValueError(
                f"label_for column {label_col!r} cannot be part of a column hierarchy"
            )


# Fonction de contrôle de la dépendance fonctionnelle code -> libellé
def check_value_label_dependency(
    conn: "duckdb.DuckDBPyConnection",
    fact_table_qualified: str,
    code: str,
    label: str,
    restrict_to: str | None = None,
) -> None:
    """
    Check the functional dependency ``code -> label``.

    For every non-null value of ``code``, ``label`` must carry a single value
    across the whole table, ``NULL`` included (a code with some labeled and some
    unlabeled rows is invalid); a row whose ``code`` is ``NULL`` must have a
    ``NULL`` ``label``.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection to run the check on.
        fact_table_qualified (str): Already schema-/catalog-qualified fact table
            reference (or a registered view/table name, e.g. at build time before
            the fact table itself exists).
        code (str): Name of the code column.
        label (str): Name of the label column.
        restrict_to (str | None): Name of a temporary view/table exposing a
            ``code`` column, used to restrict the check to the codes touched by a
            batch (an update or ``add_columns`` call) rather than scanning the
            whole table. Built by the caller as the table's *current*, post-write
            code values for the rows the batch touched (so it may itself contain
            a ``NULL`` row when the batch touched a row whose code is ``NULL``).
            ``None`` (default) checks the whole table.

    Raises:
        ValueError: If the dependency is violated, listing at most
            :data:`_MAX_FAULTY_CODES_LISTED` faulty codes with their competing
            labels. When ``restrict_to`` is given, the message also points to
            ``DatabaseUpdater.update_value_labels`` as the way to correct a
            genuine relabeling.

    Examples:
        >>> check_value_label_dependency(conn, '"main"."fact_table"', 'nc8',
        ...                               'nc8_libelle')
    """
    quoted_code = quote_ident(code)
    quoted_label = quote_ident(label)

    # Restriction optionnelle aux codes touchés par un lot (update/add_columns) :
    # une jointure IN suffit, une valeur NULL de restrict_to n'y correspondant
    # jamais (sémantique SQL à trois valeurs), ce qui est sans effet ici puisque
    # cette première requête exclut déjà les codes NULL.
    restrict_clause = (
        f" AND {quoted_code} IN (SELECT {quoted_code} FROM {restrict_to})"
        if restrict_to is not None
        else ""
    )

    # Unicité du libellé par code : au plus un libellé par code non NULL, NULL
    # compris (un code partiellement libellé est incohérent)
    faulty_rows = conn.execute(f"""
        SELECT {quoted_code}, array_agg(DISTINCT {quoted_label})
        FROM {fact_table_qualified}
        WHERE {quoted_code} IS NOT NULL{restrict_clause}
        GROUP BY {quoted_code}
        HAVING COUNT(DISTINCT {quoted_label}) > 1
            OR (COUNT({quoted_label}) > 0 AND COUNT({quoted_label}) < COUNT(*))
        LIMIT {_MAX_FAULTY_CODES_LISTED}
    """).fetchall()

    if faulty_rows:
        details = ", ".join(f"{c!r} -> {labels}" for c, labels in faulty_rows)
        message = (
            f"Functional dependency {code!r} -> {label!r} violated for"
            f" {len(faulty_rows)} code(s) (showing up to"
            f" {_MAX_FAULTY_CODES_LISTED}): {details}"
        )
        if restrict_to is not None:
            message += (
                ". To relabel a code deliberately, use"
                " DatabaseUpdater.update_value_labels instead."
            )
        raise ValueError(message)

    # Propagation du NULL : un code NULL implique un libellé NULL. La restriction ne
    # peut pas s'exprimer par un simple "IN" (NULL n'est jamais "dans" une liste,
    # même porteuse d'un NULL, en logique SQL à trois valeurs) : un EXISTS dédié est
    # nécessaire pour ne déclencher ce contrôle que si le lot touchait bien une
    # ligne à code NULL.
    null_code_clause = (
        f" AND EXISTS (SELECT 1 FROM {restrict_to} r WHERE r.{quoted_code} IS NULL)"
        if restrict_to is not None
        else ""
    )
    null_code_row = conn.execute(f"""
        SELECT COUNT(*) FROM {fact_table_qualified}
        WHERE {quoted_code} IS NULL AND {quoted_label} IS NOT NULL{null_code_clause}
    """).fetchone()
    null_code_count = null_code_row[0] if null_code_row is not None else 0

    if null_code_count > 0:
        message = (
            f"Functional dependency {code!r} -> {label!r} violated:"
            f" {null_code_count} row(s) with a NULL {code!r} carry a non-NULL"
            f" {label!r}"
        )
        if restrict_to is not None:
            message += (
                ". To relabel a code deliberately, use"
                " DatabaseUpdater.update_value_labels instead."
            )
        raise ValueError(message)


# Fonction utilitaire publique de lecture des colonnes de libellés par code
def get_value_label_columns(
    conn: "duckdb.DuckDBPyConnection",
    schema: str = "main",
    catalog_alias: str = "db",
) -> dict[str, list[str]]:
    """
    Read every declared code -> label columns mapping from ``metadata.label_for``.

    Reference implementation for API layers, symmetric to
    :func:`~.hierarchy.get_column_hierarchies`.

    Args:
        conn (duckdb.DuckDBPyConnection): DuckLake-attached connection to read
            ``metadata`` from.
        schema (str): DuckLake schema holding the ``metadata`` table. Defaults to
            ``'main'``.
        catalog_alias (str): Alias of the attached DuckLake catalog. Defaults to
            ``'db'``.

    Returns:
        dict[str, list[str]]: Code column name -> sorted list of its label column
        names. Empty when no column declares a ``label_for``.

    Examples:
        >>> get_value_label_columns(conn, schema="predictions")
        {'nc8': ['nc8_libelle_en', 'nc8_libelle_fr']}
    """
    # Qualification de la table de méta-données par le schéma (et le catalogue)
    catalog = resolve_catalog(conn, catalog_alias)
    metadata_table = qualify_table("metadata", schema, catalog)

    # Lecture des colonnes de libellés déclarées
    rows = conn.execute(
        f"SELECT name, label_for FROM {metadata_table} WHERE label_for IS NOT NULL"
    ).fetchall()

    # Regroupement par colonne de code, tri des colonnes de libellés
    result: dict[str, list[str]] = {}
    for label_col, code_col in rows:
        result.setdefault(code_col, []).append(label_col)
    for label_cols in result.values():
        label_cols.sort()

    return result
