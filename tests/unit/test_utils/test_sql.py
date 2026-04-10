# Importation des modules
# Modules de base
import logging
import narwhals as nw
import polars as pl
# Module de tests
import pytest
# Modules du package à tester
from dashboard_template_database.utils.sql import (
    remove_dataframe_duplicates,
    build_database_duplicate_removal_query,
    _build_sql_filter,
    _build_where_clause,
)


# ===========================================================================
# Tests de remove_dataframe_duplicates
# ===========================================================================

# Test de la déduplication sur toutes les colonnes quand primary_keys est None
def test_remove_duplicates_all_columns_no_primary_keys():
    """Test that all columns are used when primary_keys is None.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 10.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=None)
        >>> len(result)
        2
    """
    # Deux lignes identiques sur toutes les colonnes
    df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 10.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=None)
    assert len(result) == 2
    assert set(result['id'].to_list()) == {1, 2}


# Test de la déduplication sur les clés primaires quand primary_keys est fourni
def test_remove_duplicates_with_primary_keys():
    """Test that only primary key columns are used when primary_keys is provided.

    Even though 'value' differs between the two duplicated rows, the deduplication
    must identify them as duplicates because they share the same primary key 'id'.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 99.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=['id'])
        >>> len(result)
        2
    """
    df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 99.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=['id'])
    # Les deux lignes id=1 sont des doublons selon la PK → une seule doit rester
    assert len(result) == 2
    assert set(result['id'].to_list()) == {1, 2}


# Test de la conservation du premier doublon avec primary_keys
def test_remove_duplicates_primary_keys_keep_first():
    """Test that keep='first' retains the first occurrence when using primary_keys.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 99.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=['id'])
        >>> result.filter(nw.col('id') == 1)['value'][0]
        10.0
    """
    df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 99.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=['id'])
    # Le premier enregistrement (value=10.0) doit être conservé
    value = result.filter(nw.col('id') == 1)['value'][0]
    assert value == 10.0


# Test de la conservation du dernier doublon avec primary_keys
def test_remove_duplicates_primary_keys_keep_last():
    """Test that keep='last' retains the last occurrence when using primary_keys.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 99.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='last', primary_keys=['id'])
        >>> result.filter(nw.col('id') == 1)['value'][0]
        99.0
    """
    df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 99.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep='last', primary_keys=['id'])
    # Le dernier enregistrement (value=99.0) doit être conservé
    value = result.filter(nw.col('id') == 1)['value'][0]
    assert value == 99.0


# Test confirmant que la colonne 'value' n'est plus exclue automatiquement
def test_remove_duplicates_value_column_not_excluded():
    """Test that the 'value' column is NOT automatically excluded from deduplication.

    With primary_keys=None, all columns (including 'value') are used, so two rows
    that differ only in 'value' are NOT considered duplicates.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1], 'value': [10.0, 99.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=None)
        >>> len(result)
        2
    """
    df = pl.DataFrame({'id': [1, 1], 'value': [10.0, 99.0]})
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=None)
    # Les deux lignes diffèrent sur 'value' → aucune n'est un doublon
    assert len(result) == 2


# Test que primary_keys=[] est équivalent à primary_keys=None (toutes les colonnes)
def test_remove_duplicates_empty_primary_keys_list():
    """Test that an empty list for primary_keys behaves the same as None.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 10.0, 20.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=[])
        >>> len(result)
        2
    """
    df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 10.0, 20.0]})
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=[])
    assert len(result) == 2


# Test de l'absence de modification quand il n'y a pas de doublons
def test_remove_duplicates_no_duplicates():
    """Test that the DataFrame is unchanged when there are no duplicates.

    Examples:
        >>> df = pl.DataFrame({'id': [1, 2, 3], 'value': [10.0, 20.0, 30.0]})
        >>> result = remove_dataframe_duplicates(df, keep='first')
        >>> len(result)
        3
    """
    df = pl.DataFrame({'id': [1, 2, 3], 'value': [10.0, 20.0, 30.0]})
    result = remove_dataframe_duplicates(df, keep='first')
    assert len(result) == 3


# Test du logging lorsque des doublons sont supprimés
def test_remove_duplicates_logging():
    """Test that a warning is logged when duplicates are removed.

    Examples:
        >>> # See function body for the expected log message format
    """
    df = pl.DataFrame({'id': [1, 1, 2], 'value': [10.0, 10.0, 20.0]})

    # Création d'un logger de test avec un handler en mémoire
    test_logger = logging.getLogger('test_remove_duplicates_logging')
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = ListHandler()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.WARNING)

    remove_dataframe_duplicates(df, keep='first', logger=test_logger, source='TestSource')

    test_logger.removeHandler(handler)

    # Vérification qu'un message de warning a bien été émis avec le bon contenu
    assert len(records) == 1
    assert 'Removing duplicates from' in records[0].getMessage()
    assert 'TestSource' in records[0].getMessage()


# Test de l'absence de logging quand il n'y a pas de doublons
def test_remove_duplicates_no_logging_when_no_duplicates():
    """Test that nothing is logged when no duplicates are removed.

    Examples:
        >>> # No log record should be emitted
    """
    df = pl.DataFrame({'id': [1, 2, 3], 'value': [10.0, 20.0, 30.0]})

    test_logger = logging.getLogger('test_no_log_sql')
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = ListHandler()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.WARNING)

    remove_dataframe_duplicates(df, keep='first', logger=test_logger)

    test_logger.removeHandler(handler)
    assert len(records) == 0


# ===========================================================================
# Tests de build_database_duplicate_removal_query
# ===========================================================================

# Test de la construction de la requête SQL avec keep='first'
def test_build_duplicate_query_keep_first():
    """Test that keep='first' produces a query with ROW_NUMBER and ASC ordering.

    Examples:
        >>> query = build_database_duplicate_removal_query(['id'], 'first')
        >>> 'ROW_NUMBER()' in query
        True
    """
    query = build_database_duplicate_removal_query(['id', 'date'], 'first')
    assert 'ROW_NUMBER()' in query
    assert 'ASC' in query
    assert 'DELETE FROM fact_table' in query


# Test de la construction de la requête SQL avec keep='last'
def test_build_duplicate_query_keep_last():
    """Test that keep='last' produces a query with DESC ordering.

    Examples:
        >>> query = build_database_duplicate_removal_query(['id'], 'last')
        >>> 'DESC' in query
        True
    """
    query = build_database_duplicate_removal_query(['id'], 'last')
    assert 'DESC' in query
    assert 'ROW_NUMBER()' in query


# Test de la construction de la requête SQL avec keep=False
def test_build_duplicate_query_keep_false():
    """Test that keep=False produces a query using HAVING COUNT(*) > 1.

    Examples:
        >>> query = build_database_duplicate_removal_query(['id'], False)
        >>> 'HAVING COUNT(*) > 1' in query
        True
    """
    query = build_database_duplicate_removal_query(['id'], False)
    assert 'HAVING COUNT(*) > 1' in query
    assert 'DELETE FROM fact_table' in query


# Test que columns_to_check=[] retourne une chaîne vide
def test_build_duplicate_query_empty_columns():
    """Test that an empty columns_to_check list returns an empty string.

    Examples:
        >>> build_database_duplicate_removal_query([], 'first')
        ''
    """
    result = build_database_duplicate_removal_query([], 'first')
    assert result == ""


# Test de la construction avec un nom de table personnalisé
def test_build_duplicate_query_custom_table_name():
    """Test that the custom table_name is used in the generated query.

    Examples:
        >>> query = build_database_duplicate_removal_query(['id'], 'first', 'my_table')
        >>> 'DELETE FROM my_table' in query
        True
    """
    query = build_database_duplicate_removal_query(['id'], 'first', 'my_table')
    assert 'DELETE FROM my_table' in query


# ===========================================================================
# Tests de _build_sql_filter
# ===========================================================================

# Test d'un filtre AND (liste de tuples)
def test_build_sql_filter_list_of_tuples():
    """Test that a list of tuples generates an AND condition.

    The column names are preserved as-is; only the operator is uppercased.

    Examples:
        >>> _build_sql_filter([('col1', '=', 'val1'), ('col2', '=', 'val2')])
        "col1 = 'val1' AND col2 = 'val2'"
    """
    filters = [('col1', '=', 'val1'), ('col2', '=', 'val2')]
    result = _build_sql_filter(filters)
    assert 'AND' in result
    # Les noms de colonnes sont conservés tels quels (non mis en majuscule)
    assert 'col1' in result
    assert 'col2' in result


# Test d'un filtre OR (liste de listes de tuples)
def test_build_sql_filter_list_of_lists_of_tuples():
    """Test that a list of lists of tuples generates an AND ... OR AND condition.

    Examples:
        >>> _build_sql_filter([[('a', '=', 1)], [('b', '=', 2)]])
        "A = '1' OR B = '2'"
    """
    filters = [[('col1', '=', 'val1')], [('col2', '=', 'val2')]]
    result = _build_sql_filter(filters)
    assert 'OR' in result


# Test que l'opérateur IN est géré correctement
def test_build_sql_filter_in_operator():
    """Test that the 'in' operator generates an IN (...) clause.

    Examples:
        >>> _build_sql_filter([('status', 'in', ['active', 'pending'])])
        "STATUS IN (active, pending)"
    """
    filters = [('status', 'in', ['active', 'pending'])]
    result = _build_sql_filter(filters)
    assert 'IN' in result


# Test que l'opérateur NOT IN est géré correctement
def test_build_sql_filter_not_in_operator():
    """Test that the 'not in' operator generates a NOT IN (...) clause.

    Examples:
        >>> _build_sql_filter([('status', 'not in', ['deleted'])])
        "STATUS NOT IN (deleted)"
    """
    filters = [('status', 'not in', ['deleted'])]
    result = _build_sql_filter(filters)
    assert 'NOT IN' in result


# Test qu'un type invalide lève une TypeError
def test_build_sql_filter_invalid_type_raises_type_error():
    """Test that passing an invalid filter type raises a TypeError.

    Examples:
        >>> _build_sql_filter("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Raises:
            TypeError
    """
    with pytest.raises(TypeError):
        _build_sql_filter("invalid_filter")


# ===========================================================================
# Tests de _build_where_clause
# ===========================================================================

# Test que filters=None retourne une chaîne vide
def test_build_where_clause_none():
    """Test that filters=None returns an empty string.

    Examples:
        >>> _build_where_clause(None)
        ''
    """
    assert _build_where_clause(None) == ""


# Test que filters est une chaîne SQL brute
def test_build_where_clause_string():
    """Test that a string filter is wrapped in a WHERE clause.

    Examples:
        >>> _build_where_clause("id = 1")
        'WHERE id = 1'
    """
    result = _build_where_clause("id = 1")
    assert result == "WHERE id = 1"


# Test que filters est une liste de tuples
def test_build_where_clause_list_of_tuples():
    """Test that a list of tuples generates a WHERE clause.

    Examples:
        >>> result = _build_where_clause([('id', '=', 1)])
        >>> result.startswith('WHERE')
        True
    """
    result = _build_where_clause([('id', '=', 1)])
    assert result.startswith("WHERE")


# Test qu'un type invalide lève une TypeError
def test_build_where_clause_invalid_type_raises_type_error():
    """Test that passing an unsupported filter type raises a TypeError.

    Examples:
        >>> _build_where_clause(42)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Raises:
            TypeError
    """
    with pytest.raises(TypeError):
        _build_where_clause(42)
