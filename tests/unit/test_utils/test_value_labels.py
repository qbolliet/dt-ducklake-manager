# Importation des modules
from typing import Any

import duckdb

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.utils.value_labels import (
    check_value_label_dependency,
    get_value_label_columns,
    validate_value_labels,
)

# ---------------------------------------------------------------------------
# Tests de validate_value_labels()
# ---------------------------------------------------------------------------

# Colonnes de type SQL par défaut réutilisées par plusieurs tests
_COLUMNS = {"nc8": "VARCHAR", "nc8_libelle": "VARCHAR", "score": "DOUBLE"}


# Test qu'une cible inexistante lève une ValueError
def test_validate_value_labels_missing_target_raises() -> None:
    """Test that a label_for target absent from columns raises ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        validate_value_labels(
            {"nc8_libelle": "unknown"}, _COLUMNS, primary_keys=[], parent_of={}
        )


# Test qu'une colonne pointant vers elle-même lève une ValueError
def test_validate_value_labels_self_reference_raises() -> None:
    """Test that a label column targeting itself raises ValueError."""
    with pytest.raises(ValueError, match="own label_for target"):
        validate_value_labels(
            {"nc8_libelle": "nc8_libelle"}, _COLUMNS, primary_keys=[], parent_of={}
        )


# Test qu'une chaîne de libellés (cible elle-même colonne de libellés) lève une erreur
def test_validate_value_labels_chaining_raises() -> None:
    """Test that a target which is itself a label column raises ValueError."""
    with pytest.raises(ValueError, match="chaining"):
        validate_value_labels(
            {"a": "b", "b": "nc8"},
            {"a": "VARCHAR", "b": "VARCHAR", "nc8": "VARCHAR"},
            primary_keys=[],
            parent_of={},
        )


# Test qu'une colonne de libellés clé primaire lève une ValueError
def test_validate_value_labels_primary_key_label_raises() -> None:
    """Test that a label column that is a primary key raises ValueError."""
    with pytest.raises(ValueError, match="primary key"):
        validate_value_labels(
            {"nc8_libelle": "nc8"},
            _COLUMNS,
            primary_keys=["nc8_libelle"],
            parent_of={},
        )


# Test qu'une colonne de libellés non VARCHAR lève une ValueError
def test_validate_value_labels_non_varchar_label_raises() -> None:
    """Test that a non-VARCHAR label column raises ValueError."""
    with pytest.raises(ValueError, match="VARCHAR"):
        validate_value_labels({"score": "nc8"}, _COLUMNS, primary_keys=[], parent_of={})


# Test qu'une colonne de libellés enfant d'une hiérarchie lève une ValueError
def test_validate_value_labels_hierarchy_child_label_raises() -> None:
    """Test that a label column with a parent_name raises ValueError."""
    with pytest.raises(ValueError, match="hierarchy"):
        validate_value_labels(
            {"nc8_libelle": "nc8"},
            _COLUMNS,
            primary_keys=[],
            parent_of={"nc8_libelle": "score"},
        )


# Test qu'une colonne de libellés parente d'une hiérarchie lève une ValueError
def test_validate_value_labels_hierarchy_parent_label_raises() -> None:
    """Test that a label column that is a hierarchy parent raises ValueError."""
    with pytest.raises(ValueError, match="hierarchy"):
        validate_value_labels(
            {"nc8_libelle": "nc8"},
            _COLUMNS,
            primary_keys=[],
            parent_of={"score": "nc8_libelle"},
        )


# Test que deux paires code/libellé indépendantes et valides ne lèvent rien
def test_validate_value_labels_two_independent_pairs_ok() -> None:
    """Test that two unrelated, valid code/label pairs pass validation."""
    validate_value_labels(
        {"nc8_libelle": "nc8", "b_libelle": "b"},
        {
            "nc8": "VARCHAR",
            "nc8_libelle": "VARCHAR",
            "b": "VARCHAR",
            "b_libelle": "VARCHAR",
        },
        primary_keys=[],
        parent_of={},
    )


# ---------------------------------------------------------------------------
# Tests de check_value_label_dependency()
# ---------------------------------------------------------------------------


# Fixture d'une connexion in-memory portant une fact_table clean (dépendance respectée)
@pytest.fixture
def conn_clean_dependency() -> Any:
    """Provide an in-memory connection with a table respecting code -> label."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE fact_table (nc8 VARCHAR, nc8_libelle VARCHAR)
    """)
    conn.execute("""
        INSERT INTO fact_table VALUES
            ('01', 'Chevaux'), ('01', 'Chevaux'), ('02', 'Bovins'), (NULL, NULL)
    """)
    return conn


# Test qu'une table respectant la dépendance ne lève rien
def test_check_value_label_dependency_clean_table_ok(
    conn_clean_dependency: Any,
) -> None:
    """Test that a table respecting the functional dependency passes."""
    check_value_label_dependency(
        conn_clean_dependency, "fact_table", "nc8", "nc8_libelle"
    )


# Test que deux libellés distincts pour un même code lèvent une ValueError
def test_check_value_label_dependency_two_labels_for_one_code_raises() -> None:
    """Test that two distinct labels for the same code raise ValueError."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE fact_table (nc8 VARCHAR, nc8_libelle VARCHAR)")
    conn.execute("""
        INSERT INTO fact_table VALUES ('01', 'Chevaux'), ('01', 'Autre libelle')
    """)
    with pytest.raises(ValueError, match="Functional dependency"):
        check_value_label_dependency(conn, "fact_table", "nc8", "nc8_libelle")


# Test qu'un libellé NULL sur une partie des lignes d'un code lève une ValueError
def test_check_value_label_dependency_partial_null_label_raises() -> None:
    """Test that a code with some NULL and some non-NULL labels raises ValueError."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE fact_table (nc8 VARCHAR, nc8_libelle VARCHAR)")
    conn.execute("""
        INSERT INTO fact_table VALUES ('01', 'Chevaux'), ('01', NULL)
    """)
    with pytest.raises(ValueError, match="Functional dependency"):
        check_value_label_dependency(conn, "fact_table", "nc8", "nc8_libelle")


# Test qu'un libellé non NULL sous un code NULL lève une ValueError
def test_check_value_label_dependency_non_null_label_under_null_code_raises() -> None:
    """Test that a non-NULL label under a NULL code raises ValueError."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE fact_table (nc8 VARCHAR, nc8_libelle VARCHAR)")
    conn.execute("INSERT INTO fact_table VALUES (NULL, 'Orphan label')")
    with pytest.raises(ValueError, match="Functional dependency"):
        check_value_label_dependency(conn, "fact_table", "nc8", "nc8_libelle")


# Test que restrict_to limite le contrôle aux codes du lot (violation hors lot ignorée)
def test_check_value_label_dependency_restrict_to_narrows_check() -> None:
    """Test that restrict_to skips a violation outside the restricted codes."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE fact_table (nc8 VARCHAR, nc8_libelle VARCHAR)")
    # '01' est en violation (deux libellés), '02' est propre
    conn.execute("""
        INSERT INTO fact_table VALUES
            ('01', 'Chevaux'), ('01', 'Autre'), ('02', 'Bovins')
    """)
    conn.execute("CREATE TEMP VIEW restrict_codes AS SELECT '02' AS nc8")
    # Restreint à '02' : aucune violation détectée
    check_value_label_dependency(
        conn, "fact_table", "nc8", "nc8_libelle", restrict_to="restrict_codes"
    )
    # Sans restriction : la violation sur '01' est détectée
    with pytest.raises(ValueError, match="Functional dependency"):
        check_value_label_dependency(conn, "fact_table", "nc8", "nc8_libelle")

    # Avec restriction incluant le code fautif : la violation est aussi détectée,
    # avec un message pointant vers update_value_labels
    conn.execute("CREATE OR REPLACE TEMP VIEW restrict_codes AS SELECT '01' AS nc8")
    with pytest.raises(ValueError, match="update_value_labels"):
        check_value_label_dependency(
            conn, "fact_table", "nc8", "nc8_libelle", restrict_to="restrict_codes"
        )


# Test que restrict_to avec un code NULL déclenche bien le contrôle NULL -> libellé
def test_check_value_label_dependency_restrict_to_null_code_triggers_check() -> None:
    """Test that a restrict_to view carrying a NULL code still catches a violation."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE fact_table (nc8 VARCHAR, nc8_libelle VARCHAR)")
    conn.execute("INSERT INTO fact_table VALUES (NULL, 'Orphan label')")
    # La vue restrict_to porte une ligne à code NULL (lot ayant touché cette ligne)
    conn.execute(
        "CREATE TEMP VIEW restrict_codes AS SELECT CAST(NULL AS VARCHAR) AS nc8"
    )
    with pytest.raises(ValueError, match="Functional dependency"):
        check_value_label_dependency(
            conn, "fact_table", "nc8", "nc8_libelle", restrict_to="restrict_codes"
        )


# Test que restrict_to sans code NULL ne déclenche pas le contrôle NULL -> libellé
def test_check_value_label_dependency_restrict_to_without_null_code_skips_check() -> (
    None
):
    """Test that restrict_to skips the NULL-code check when it carries no NULL row."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE fact_table (nc8 VARCHAR, nc8_libelle VARCHAR)")
    conn.execute(
        "INSERT INTO fact_table VALUES (NULL, 'Orphan label'), ('02', 'Bovins')"
    )
    conn.execute("CREATE TEMP VIEW restrict_codes AS SELECT '02' AS nc8")
    check_value_label_dependency(
        conn, "fact_table", "nc8", "nc8_libelle", restrict_to="restrict_codes"
    )


# ---------------------------------------------------------------------------
# Tests de get_value_label_columns()
# ---------------------------------------------------------------------------


# Test d'une seule paire code/libellé déclarée
def test_get_value_label_columns_single_pair() -> None:
    """Test that a single declared pair is returned."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA IF NOT EXISTS main")
    conn.execute("CREATE TABLE main.metadata (name VARCHAR, label_for VARCHAR)")
    conn.execute("""
        INSERT INTO main.metadata (name, label_for) VALUES
            ('nc8', NULL), ('nc8_libelle', 'nc8')
    """)
    assert get_value_label_columns(conn, schema="main") == {"nc8": ["nc8_libelle"]}


# Test de deux colonnes de libellés pour un même code
def test_get_value_label_columns_two_label_columns_for_one_code() -> None:
    """Test that two label columns targeting the same code are both returned, sorted."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA IF NOT EXISTS main")
    conn.execute("CREATE TABLE main.metadata (name VARCHAR, label_for VARCHAR)")
    conn.execute("""
        INSERT INTO main.metadata (name, label_for) VALUES
            ('nc8', NULL),
            ('nc8_libelle_fr', 'nc8'),
            ('nc8_libelle_en', 'nc8')
    """)
    assert get_value_label_columns(conn, schema="main") == {
        "nc8": ["nc8_libelle_en", "nc8_libelle_fr"]
    }


# Test que l'absence de toute colonne de libellés renvoie un dictionnaire vide
def test_get_value_label_columns_empty_when_no_label_for_declared() -> None:
    """Test that get_value_label_columns returns {} when no label_for is declared."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA IF NOT EXISTS main")
    conn.execute("CREATE TABLE main.metadata (name VARCHAR, label_for VARCHAR)")
    conn.execute("INSERT INTO main.metadata (name, label_for) VALUES ('a', NULL)")
    assert get_value_label_columns(conn, schema="main") == {}
