# Importation des modules
# Modules de base
from typing import Any

import duckdb

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.utils.hierarchy import (
    get_column_hierarchies,
    validate_hierarchy_forest,
)

# ---------------------------------------------------------------------------
# Tests de validate_hierarchy_forest()
# ---------------------------------------------------------------------------


# Test qu'une chaîne linéaire simple ne lève rien
def test_validate_hierarchy_forest_linear_chain_ok() -> None:
    """Test that a simple root-to-leaf chain passes validation without error."""
    validate_hierarchy_forest(
        {"commune": "departement", "departement": "region", "region": None}
    )


# Test qu'une hiérarchie profonde (5+ niveaux) passe la validation
def test_validate_hierarchy_forest_deep_chain_ok() -> None:
    """Test that a hierarchy with 5+ levels passes validation without error."""
    validate_hierarchy_forest(
        {
            "l5": "l4",
            "l4": "l3",
            "l3": "l2",
            "l2": "l1",
            "l1": "l0",
            "l0": None,
        }
    )


# Test que deux hiérarchies indépendantes passent la validation
def test_validate_hierarchy_forest_two_independent_trees_ok() -> None:
    """Test that two unrelated hierarchy chains coexist without error."""
    validate_hierarchy_forest(
        {
            "commune": "departement",
            "departement": "region",
            "subcategory": "category",
        }
    )


# Test qu'une auto-référence est détectée comme un cycle
def test_validate_hierarchy_forest_self_reference_raises() -> None:
    """Test that a column declared as its own parent raises ValueError."""
    with pytest.raises(ValueError, match="Cycle detected"):
        validate_hierarchy_forest({"a": "a"})


# Test qu'un cycle à deux colonnes (A -> B -> A) est détecté
def test_validate_hierarchy_forest_two_node_cycle_raises() -> None:
    """Test that a two-column cycle (A -> B -> A) raises ValueError."""
    with pytest.raises(ValueError, match="Cycle detected"):
        validate_hierarchy_forest({"a": "b", "b": "a"})


# Test qu'un cycle plus long (A -> B -> C -> A) est détecté
def test_validate_hierarchy_forest_longer_cycle_raises() -> None:
    """Test that a longer cycle (A -> B -> C -> A) raises ValueError."""
    with pytest.raises(ValueError, match="Cycle detected"):
        validate_hierarchy_forest({"a": "b", "b": "c", "c": "a"})


# Test qu'une hiérarchie à un seul niveau (une seule colonne, pas de parent) est valide
def test_validate_hierarchy_forest_single_level_ok() -> None:
    """Test that a single column with no parent is trivially valid."""
    validate_hierarchy_forest({"region": None})


# ---------------------------------------------------------------------------
# Tests de get_column_hierarchies()
# ---------------------------------------------------------------------------


# Fixture d'une connexion in-memory portant une table metadata avec hiérarchie
@pytest.fixture
def conn_with_hierarchy() -> Any:
    """Provide an in-memory connection with a metadata table declaring a hierarchy.

    Builds a ``main.metadata`` table with a 3-level geographic chain
    (region -> departement -> commune) and one isolated column with no parent.
    """
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA IF NOT EXISTS main")
    conn.execute("""
        CREATE TABLE main.metadata (name VARCHAR, parent_name VARCHAR)
    """)
    conn.execute("""
        INSERT INTO main.metadata (name, parent_name) VALUES
            ('region', NULL),
            ('departement', 'region'),
            ('commune', 'departement'),
            ('value', NULL)
    """)
    return conn


# Test de la reconstitution d'une chaîne racine → feuille
def test_get_column_hierarchies_reconstructs_chain(
    conn_with_hierarchy: Any,
) -> None:
    """Test that get_column_hierarchies rebuilds the root-to-leaf chain.

    Args:
        conn_with_hierarchy: DuckDB connection fixture with a 3-level hierarchy.
    """
    chains = get_column_hierarchies(conn_with_hierarchy, schema="main")
    assert chains == [["region", "departement", "commune"]]


# Test qu'une colonne sans parent et jamais parente n'apparaît dans aucune chaîne
def test_get_column_hierarchies_excludes_standalone_columns(
    conn_with_hierarchy: Any,
) -> None:
    """Test that a column outside any hierarchy is not returned.

    Args:
        conn_with_hierarchy: DuckDB connection fixture with a 3-level hierarchy.
    """
    chains = get_column_hierarchies(conn_with_hierarchy, schema="main")
    flattened = {col for chain in chains for col in chain}
    assert "value" not in flattened


# Test que l'absence de toute hiérarchie renvoie une liste vide
def test_get_column_hierarchies_empty_when_no_parent_declared() -> None:
    """Test that get_column_hierarchies returns [] when no column has a parent."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA IF NOT EXISTS main")
    conn.execute("CREATE TABLE main.metadata (name VARCHAR, parent_name VARCHAR)")
    conn.execute(
        "INSERT INTO main.metadata (name, parent_name) VALUES ('a', NULL), ('b', NULL)"
    )
    assert get_column_hierarchies(conn, schema="main") == []


# Test que deux hiérarchies indépendantes produisent deux chaînes distinctes
def test_get_column_hierarchies_two_independent_trees() -> None:
    """Test that get_column_hierarchies returns one chain per independent hierarchy."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA IF NOT EXISTS main")
    conn.execute("CREATE TABLE main.metadata (name VARCHAR, parent_name VARCHAR)")
    conn.execute("""
        INSERT INTO main.metadata (name, parent_name) VALUES
            ('region', NULL),
            ('departement', 'region'),
            ('category', NULL),
            ('subcategory', 'category')
    """)
    chains = get_column_hierarchies(conn, schema="main")
    assert sorted(chains) == [["category", "subcategory"], ["region", "departement"]]
