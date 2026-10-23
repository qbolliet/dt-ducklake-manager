# Importation des modules
# Modules de base
import logging
from typing import Any

import duckdb
import narwhals as nw
import polars as pl
import pyarrow as pa

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.utils.sql import (
    _build_sql_filter,
    _build_where_clause,
    fetch_arrow,
    fetch_one,
    quote_literal,
    register_view,
    remove_dataframe_duplicates,
)

# ===========================================================================
# Tests de remove_dataframe_duplicates
# ===========================================================================


# Test de la déduplication sur toutes les colonnes quand primary_keys est None
def test_remove_duplicates_all_columns_no_primary_keys() -> None:
    """Test that all columns are used when primary_keys is None.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 10.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=None)
        >>> len(result)
        2
    """
    # Deux lignes identiques sur toutes les colonnes
    df = pl.DataFrame({"id": [1, 1, 2], "value": [10.0, 10.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep="first", primary_keys=None)
    assert len(result) == 2
    assert set(result["id"].to_list()) == {1, 2}


# Test de la déduplication sur les clés primaires quand primary_keys est fourni
def test_remove_duplicates_with_primary_keys() -> None:
    """Test that only primary key columns are used when primary_keys is provided.

    Even though 'value' differs between the two duplicated rows, the deduplication
    must identify them as duplicates because they share the same primary key 'id'.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 99.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=['id'])
        >>> len(result)
        2
    """
    df = pl.DataFrame({"id": [1, 1, 2], "value": [10.0, 99.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep="first", primary_keys=["id"])
    # Les deux lignes id=1 sont des doublons selon la PK → une seule doit rester
    assert len(result) == 2
    assert set(result["id"].to_list()) == {1, 2}


# Test de la conservation du premier doublon avec primary_keys
def test_remove_duplicates_primary_keys_keep_first() -> None:
    """Test that keep='first' retains the first occurrence when using primary_keys.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 99.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=['id'])
        >>> result.filter(nw.col('id') == 1)['value'][0]
        10.0
    """
    df = pl.DataFrame({"id": [1, 1, 2], "value": [10.0, 99.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep="first", primary_keys=["id"])
    # Le premier enregistrement (value=10.0) doit être conservé
    value = result.filter(nw.col("id") == 1)["value"][0]
    assert value == 10.0


# Test de la conservation du dernier doublon avec primary_keys
def test_remove_duplicates_primary_keys_keep_last() -> None:
    """Test that keep='last' retains the last occurrence when using primary_keys.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 99.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='last', primary_keys=['id'])
        >>> result.filter(nw.col('id') == 1)['value'][0]
        99.0
    """
    df = pl.DataFrame({"id": [1, 1, 2], "value": [10.0, 99.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep="last", primary_keys=["id"])
    # Le dernier enregistrement (value=99.0) doit être conservé
    value = result.filter(nw.col("id") == 1)["value"][0]
    assert value == 99.0


# Test confirmant que la colonne 'value' n'est plus exclue automatiquement
def test_remove_duplicates_value_column_not_excluded() -> None:
    """Test that the 'value' column is NOT automatically excluded from deduplication.

    With primary_keys=None, all columns (including 'value') are used, so two rows
    that differ only in 'value' are NOT considered duplicates.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1], 'value': [10.0, 99.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=None)
        >>> len(result)
        2
    """
    df = pl.DataFrame({"id": [1, 1], "value": [10.0, 99.0]})
    result = remove_dataframe_duplicates(df, keep="first", primary_keys=None)
    # Les deux lignes diffèrent sur 'value' → aucune n'est un doublon
    assert len(result) == 2


# Test que primary_keys=[] est équivalent à primary_keys=None (toutes les colonnes)
def test_remove_duplicates_empty_primary_keys_list() -> None:
    """Test that an empty list for primary_keys behaves the same as None.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 10.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=[])
        >>> len(result)
        2
    """
    df = pl.DataFrame({"id": [1, 1, 2], "value": [10.0, 10.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep="first", primary_keys=[])
    assert len(result) == 2


# Test de l'absence de modification quand il n'y a pas de doublons
def test_remove_duplicates_no_duplicates() -> None:
    """Test that the DataFrame is unchanged when there are no duplicates.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 2, 3], 'value': [10.0, 20.0, 30.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first')
        >>> len(result)
        3
    """
    df = pl.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    result = remove_dataframe_duplicates(df, keep="first")
    assert len(result) == 3


# Test du logging lorsque des doublons sont supprimés
def test_remove_duplicates_logging() -> None:
    """Test that a warning is logged when duplicates are removed.

    Examples:
        >>> # See function body for the expected log message format
    """
    df = pl.DataFrame({"id": [1, 1, 2], "value": [10.0, 10.0, 20.0]})

    # Création d'un logger de test avec un handler en mémoire
    test_logger = logging.getLogger("test_remove_duplicates_logging")
    records = []

    class ListHandler(logging.Handler):
        def emit(self: Any, record: Any) -> None:
            records.append(record)

    handler = ListHandler()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.WARNING)

    remove_dataframe_duplicates(
        df, keep="first", logger=test_logger, source="TestSource"
    )

    test_logger.removeHandler(handler)

    # Vérification qu'un message de warning a bien été émis avec le bon contenu
    assert len(records) == 1
    assert "Removing duplicates from" in records[0].getMessage()
    assert "TestSource" in records[0].getMessage()


# Test de l'absence de logging quand il n'y a pas de doublons
def test_remove_duplicates_no_logging_when_no_duplicates() -> None:
    """Test that nothing is logged when no duplicates are removed.

    Examples:
        >>> # No log record should be emitted
    """
    df = pl.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})

    test_logger = logging.getLogger("test_no_log_sql")
    records = []

    class ListHandler(logging.Handler):
        def emit(self: Any, record: Any) -> None:
            records.append(record)

    handler = ListHandler()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.WARNING)

    remove_dataframe_duplicates(df, keep="first", logger=test_logger)

    test_logger.removeHandler(handler)
    assert len(records) == 0


# ===========================================================================
# Tests de quote_literal
# ===========================================================================


# Test de la mise entre apostrophes d'un littéral simple et d'un littéral piégé
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("data/", "'data/'"),
        ("it's", "'it''s'"),
        ("", "''"),
        ("a'b'c", "'a''b''c'"),
    ],
)
def test_quote_literal(value: str, expected: str) -> None:
    """Test that quote_literal wraps and escapes embedded single quotes.

    Args:
        value: Raw value.
        expected: Expected SQL literal.
    """
    assert quote_literal(value) == expected


# Test que le littéral produit est relu à l'identique par DuckDB
def test_quote_literal_round_trip() -> None:
    """Test that DuckDB reads the quoted literal back as the original value."""
    import duckdb

    value = "C:\\data\\it's here/"
    row = duckdb.connect().execute(f"SELECT {quote_literal(value)}").fetchone()
    assert row is not None and row[0] == value


# Test de la déduplication sur les clés primaires avec keep='none'
def test_remove_duplicates_primary_keys_keep_none() -> None:
    """Test that keep='none' drops every row of a duplicated key."""
    df = pl.DataFrame({"id": [1, 1, 2], "value": [10.0, 99.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep="none", primary_keys=["id"])
    assert result["id"].to_list() == [2]


# ===========================================================================
# Tests de _build_sql_filter
# ===========================================================================


# Test d'un filtre AND (liste de tuples)
def test_build_sql_filter_list_of_tuples() -> None:
    """Test that a list of tuples generates an AND condition.

    The column names are preserved as-is; only the operator is uppercased.

    Examples:
        >>> _build_sql_filter([('col1', '=', 'val1'), ('col2', '=', 'val2')])
        "col1 = 'val1' AND col2 = 'val2'"
    """
    filters = [("col1", "=", "val1"), ("col2", "=", "val2")]
    result = _build_sql_filter(filters)
    assert "AND" in result
    # Les noms de colonnes sont conservés tels quels (non mis en majuscule)
    assert "col1" in result
    assert "col2" in result


# Test d'un filtre OR (liste de listes de tuples)
def test_build_sql_filter_list_of_lists_of_tuples() -> None:
    """Test that a list of lists of tuples generates
    an AND ... OR AND condition.

    Examples:
        >>> _build_sql_filter([[('a', '=', 1)], [('b', '=', 2)]])
        "A = '1' OR B = '2'"
    """
    filters = [[("col1", "=", "val1")], [("col2", "=", "val2")]]
    result = _build_sql_filter(filters)
    assert "OR" in result


# Test que l'opérateur IN est géré correctement
def test_build_sql_filter_in_operator() -> None:
    """Test that the 'in' operator generates an IN (...) clause.

    Examples:
        >>> _build_sql_filter([('status', 'in', ['active', 'pending'])])
        "STATUS IN (active, pending)"
    """
    filters = [("status", "in", ["active", "pending"])]
    result = _build_sql_filter(filters)
    assert "IN" in result


# Test que l'opérateur NOT IN est géré correctement
def test_build_sql_filter_not_in_operator() -> None:
    """Test that the 'not in' operator generates a NOT IN (...) clause.

    Examples:
        >>> _build_sql_filter([('status', 'not in', ['deleted'])])
        "STATUS NOT IN (deleted)"
    """
    filters = [("status", "not in", ["deleted"])]
    result = _build_sql_filter(filters)
    assert "NOT IN" in result


# Test qu'un type invalide lève une TypeError
def test_build_sql_filter_invalid_type_raises_type_error() -> None:
    """Test that passing an invalid filter type raises a TypeError.

    Examples:
        >>> _build_sql_filter("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Raises:
            TypeError
    """
    with pytest.raises(TypeError):
        _build_sql_filter("invalid_filter")  # type: ignore[arg-type]


# ===========================================================================
# Tests de _build_where_clause
# ===========================================================================


# Test que filters=None retourne une chaîne vide
def test_build_where_clause_none() -> None:
    """Test that filters=None returns an empty string.

    Examples:
        >>> _build_where_clause(None)
        ''
    """
    assert _build_where_clause(None) == ""


# Test que filters est une chaîne SQL brute
def test_build_where_clause_string() -> None:
    """Test that a string filter is wrapped in a WHERE clause.

    Examples:
        >>> _build_where_clause("id = 1")
        'WHERE id = 1'
    """
    result = _build_where_clause("id = 1")
    assert result == "WHERE id = 1"


# Test que filters est une liste de tuples
def test_build_where_clause_list_of_tuples() -> None:
    """Test that a list of tuples generates a WHERE clause.

    Examples:
        >>> result = _build_where_clause([('id', '=', 1)])
        >>> result.startswith('WHERE')
        True
    """
    result = _build_where_clause([("id", "=", 1)])
    assert result.startswith("WHERE")


# Test qu'un type invalide lève une TypeError
def test_build_where_clause_invalid_type_raises_type_error() -> None:
    """Test that passing an unsupported filter type raises a TypeError.

    Examples:
        >>> _build_where_clause(42)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Raises:
            TypeError
    """
    with pytest.raises(TypeError):
        _build_where_clause(42)  # type: ignore[arg-type]


# ===========================================================================
# Tests de fetch_one / fetch_arrow / register_view (résultats ouverts, DuckDB 2.0)
# ===========================================================================


# Test que fetch_one renvoie la première ligne et libère le résultat
def test_fetch_one_returns_first_row_and_releases_result() -> None:
    """Test that fetch_one returns the first row and leaves no open result.

    A bare ``fetchone()`` on a multi-row result keeps it open under DuckDB 2.0,
    which makes the next ``sql()`` raise ``ResourceInUseException``.
    """
    conn = duckdb.connect(":memory:")
    row = fetch_one(conn.execute("SELECT * FROM range(5)"))
    assert row == (0,)
    # La connexion est de nouveau utilisable par une relation
    assert conn.sql("SELECT 42").fetchall() == [(42,)]


# Test que fetch_one renvoie None sur un résultat vide
def test_fetch_one_empty_result_returns_none() -> None:
    """Test that fetch_one returns None for an empty result."""
    conn = duckdb.connect(":memory:")
    assert fetch_one(conn.execute("SELECT 1 WHERE false")) is None
    assert conn.sql("SELECT 1").fetchall() == [(1,)]


# Test que fetch_one n'interrompt pas la transaction ouverte
def test_fetch_one_keeps_open_transaction() -> None:
    """Test that fetch_one inside a transaction leaves it usable and committable."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.begin()
    conn.execute("INSERT INTO t VALUES (1), (2)")
    assert fetch_one(conn.execute("SELECT count(*) FROM t")) == (2,)
    conn.execute("INSERT INTO t VALUES (3)")
    conn.commit()
    assert conn.execute("SELECT count(*) FROM t").fetchall() == [(3,)]


# Test que fetch_arrow fonctionne malgré un résultat execute() laissé ouvert
def test_fetch_arrow_after_pending_execute_result() -> None:
    """Test that fetch_arrow completes a caller's open execute() result first."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(3)")
    # Résultat partiellement lu par l'appelant : bloquerait un sql() nu
    conn.execute("SELECT * FROM t").fetchone()
    table = fetch_arrow(conn, "SELECT a FROM t ORDER BY a")
    assert isinstance(table, pa.Table)
    assert table.column("a").to_pylist() == [0, 1, 2]


# Test que fetch_arrow conserve le schéma d'un résultat vide
def test_fetch_arrow_empty_result_keeps_schema() -> None:
    """Test that an empty result still carries its column names."""
    conn = duckdb.connect(":memory:")
    table = fetch_arrow(conn, "SELECT 1 AS a, 'x' AS b WHERE false")
    assert table.num_rows == 0
    assert table.column_names == ["a", "b"]


# Test que fetch_arrow propage l'erreur de catalogue d'une table absente
def test_fetch_arrow_missing_table_raises_catalog_exception() -> None:
    """Test that a missing table raises duckdb.CatalogException (not swallowed)."""
    conn = duckdb.connect(":memory:")
    with pytest.raises(duckdb.CatalogException):
        fetch_arrow(conn, "SELECT * FROM missing_table")


# Test que register_view fonctionne malgré un résultat execute() laissé ouvert
def test_register_view_after_pending_execute_result() -> None:
    """Test that register_view completes a caller's open execute() result first."""
    conn = duckdb.connect(":memory:")
    conn.execute("SELECT * FROM range(10)").fetchone()
    register_view(conn, "v", pa.table({"a": [1, 2, 3]}))
    assert conn.execute("SELECT sum(a) FROM v").fetchall() == [(6,)]


# Test que register_view accepte un DataFrame Polars
def test_register_view_polars_dataframe() -> None:
    """Test that register_view accepts any object DuckDB can register."""
    conn = duckdb.connect(":memory:")
    register_view(conn, "v", pl.DataFrame({"a": [1, 2]}))
    assert conn.execute("SELECT count(*) FROM v").fetchall() == [(2,)]
