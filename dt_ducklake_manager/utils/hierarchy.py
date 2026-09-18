# Importation des modules
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .sql import qualify_table, resolve_catalog

if TYPE_CHECKING:
    import duckdb


# Fonction de validation de la forêt des colonnes parentes (absence de cycle)
def validate_hierarchy_forest(parent_of: Mapping[str, str | None]) -> None:
    """
    Validate that a column -> parent column mapping forms a forest (no cycle).

    A column hierarchy is a chain of ``fact_table``
    columns declared through ``metadata.parent_name``. Since each column carries
    at most one parent, the mapping is structurally a forest as long as no chain
    of parent pointers loops back on itself. Detection walks "de proche en
    proche" (one hop at a time) from every column, which also catches a column
    declared as its own parent.

    Args:
        parent_of (Mapping[str, str | None]): Mapping of column name to its
            parent column name, or ``None`` when the column has no parent
            (``Mapping`` rather than ``dict`` so a ``dict[str, str]`` -- whose
            values are never ``None`` -- can be passed directly). The mapping
            should be complete enough to resolve every parent pointer reachable
            from the columns being validated (an absent key ends the walk, as if
            the column had no parent).

    Raises:
        ValueError: If following parent pointers from any column revisits a
            column already seen on that walk (a cycle, including a
            self-reference).

    Examples:
        >>> validate_hierarchy_forest(
        ...     {"commune": "departement", "departement": "region", "region": None}
        ... )
        >>> validate_hierarchy_forest({"a": "a"})
        Traceback (most recent call last):
            ...
        ValueError: Cycle detected in parent_name hierarchy: column 'a' eventually
        points back to 'a'
    """
    # Parcours de proche en proche depuis chaque colonne portant un parent
    for start, start_parent in parent_of.items():
        if start_parent is None:
            continue
        seen = {start}
        current: str | None = start_parent
        while current is not None:
            if current in seen:
                raise ValueError(
                    f"Cycle detected in parent_name hierarchy: column {start!r} "
                    f"eventually points back to {current!r}"
                )
            seen.add(current)
            current = parent_of.get(current)


# Fonction utilitaire publique de reconstitution des chaînes de hiérarchie de colonnes
def get_column_hierarchies(
    conn: "duckdb.DuckDBPyConnection",
    schema: str = "main",
    catalog_alias: str = "db",
) -> list[list[str]]:
    """
    Reconstruct root-to-leaf column hierarchy chains from ``metadata.parent_name``.

    A column hierarchy is a chain of ``fact_table`` columns (e.g. ``region`` ->
    ``departement`` -> ``commune``), declared one link at a time via
    ``metadata.parent_name``. This is a reference implementation for API layers
    that need the chains rather than the raw per-column pointers: one list per
    leaf column, ordered from the root down to that leaf.

    Convention for an irregular value tree (leaves at different depths): the
    **column** chains returned here are always complete (a column either has a
    parent or it does not); it is the *values* fetched for a given chain
    (``SELECT DISTINCT region, departement, commune FROM fact_table``) that may
    carry ``NULL`` at the deeper levels for a given row. Consumers should stop
    building a menu branch at the first ``NULL`` level and never repeat the
    value of the level above (doing so would fabricate a node that does not
    exist in the data).

    Args:
        conn (duckdb.DuckDBPyConnection): DuckLake-attached connection to read
            ``metadata`` from.
        schema (str): DuckLake schema holding the ``metadata`` table. Defaults
            to ``'main'``.
        catalog_alias (str): Alias of the attached DuckLake catalog. Defaults to
            ``'db'``.

    Returns:
        list[list[str]]: One chain per leaf column, each ordered from the root
        column to the leaf column. Columns outside any hierarchy (no parent and
        never referenced as a parent) are omitted. Empty when no column
        declares a ``parent_name``.

    Raises:
        RuntimeError: If the stored ``parent_name`` graph contains a cycle,
            which should never happen given the write-time validation performed
            by :func:`validate_hierarchy_forest`.

    Examples:
        >>> chains = get_column_hierarchies(conn, schema="predictions")
        >>> chains
        [['region', 'departement', 'commune']]
    """
    # Qualification de la table de méta-données par le schéma (et le catalogue)
    catalog = resolve_catalog(conn, catalog_alias)
    metadata_table = qualify_table("metadata", schema, catalog)

    # Lecture du graphe des colonnes parentes
    rows = conn.execute(f"SELECT name, parent_name FROM {metadata_table}").fetchall()
    parent_of: dict[str, str] = {
        name: parent for name, parent in rows if parent is not None
    }

    # Une feuille est une colonne qui a un parent mais n'est elle-même parente
    # d'aucune autre colonne. Tri pour un ordre déterministe.
    parents_used = set(parent_of.values())
    leaves = sorted(name for name in parent_of if name not in parents_used)

    # Reconstruction de chaque chaîne racine → feuille par remontée des parents
    chains: list[list[str]] = []
    for leaf in leaves:
        chain = [leaf]
        current: str | None = parent_of.get(leaf)
        while current is not None:
            if current in chain:
                raise RuntimeError(
                    f"Cycle detected in stored parent_name hierarchy while "
                    f"walking up from column {leaf!r}"
                )
            chain.append(current)
            current = parent_of.get(current)
        chains.append(list(reversed(chain)))

    return chains
