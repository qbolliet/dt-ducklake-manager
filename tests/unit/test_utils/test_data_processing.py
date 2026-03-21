# Importation des modules
# Modules de base
import logging
import pandas as pd
# Module de tests
import pytest
# Modules du package à tester
from dashboard_template_database.utils.data_processing import remove_dataframe_duplicates


# ---------------------------------------------------------------------------
# Tests de la fonction remove_dataframe_duplicates
# ---------------------------------------------------------------------------

# Test de la déduplication sur toutes les colonnes quand primary_keys est None
def test_remove_duplicates_all_columns_no_primary_keys():
    """Test that all columns are used when primary_keys is None."""
    # Deux lignes identiques sur toutes les colonnes
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'value': [10.0, 10.0, 20.0],
    })
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=None)
    assert len(result) == 2
    assert list(result['id']) == [1, 2]


# Test de la déduplication sur les clés primaires quand primary_keys est fourni
def test_remove_duplicates_with_primary_keys():
    """Test that only primary key columns are used when primary_keys is provided.

    Even though 'value' differs between the two duplicated rows, the deduplication
    must identify them as duplicates because they share the same primary key 'id'.
    """
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'value': [10.0, 99.0, 20.0],  # Valeurs différentes pour id=1
    })
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=['id'])
    # Les deux lignes id=1 sont des doublons selon la PK → une seule doit rester
    assert len(result) == 2
    assert set(result['id']) == {1, 2}


# Test de la conservation du premier doublon avec primary_keys
def test_remove_duplicates_primary_keys_keep_first():
    """Test that keep='first' retains the first occurrence when using primary_keys."""
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'value': [10.0, 99.0, 20.0],
    })
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=['id'])
    # Le premier enregistrement (value=10.0) doit être conservé
    assert result.loc[result['id'] == 1, 'value'].iloc[0] == 10.0


# Test de la conservation du dernier doublon avec primary_keys
def test_remove_duplicates_primary_keys_keep_last():
    """Test that keep='last' retains the last occurrence when using primary_keys."""
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'value': [10.0, 99.0, 20.0],
    })
    result = remove_dataframe_duplicates(df, keep='last', primary_keys=['id'])
    # Le dernier enregistrement (value=99.0) doit être conservé
    assert result.loc[result['id'] == 1, 'value'].iloc[0] == 99.0


# Test de suppression totale des doublons avec keep=False et primary_keys
def test_remove_duplicates_primary_keys_keep_false():
    """Test that keep=False removes all rows that have a duplicate primary key."""
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'value': [10.0, 99.0, 20.0],
    })
    result = remove_dataframe_duplicates(df, keep=False, primary_keys=['id'])
    # Les deux lignes id=1 sont supprimées ; seule id=2 reste
    assert len(result) == 1
    assert result.iloc[0]['id'] == 2


# Test confirmant que la colonne 'value' n'est plus exclue automatiquement
def test_remove_duplicates_value_column_no_longer_excluded():
    """Test that the 'value' column is NOT automatically excluded from deduplication.

    Before the fix, the column named 'value' was hard-coded as excluded. Now, with
    primary_keys=None, all columns (including 'value') are used, so two rows that
    differ only in 'value' are NOT considered duplicates.
    """
    df = pd.DataFrame({
        'id': [1, 1],
        'value': [10.0, 99.0],  # Différentes → plus de doublon sur toutes les colonnes
    })
    # Sans primary_keys, toutes les colonnes sont utilisées : les deux lignes diffèrent
    # sur 'value', donc aucune n'est un doublon
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=None)
    assert len(result) == 2


# Test que primary_keys=[] est équivalent à primary_keys=None (toutes les colonnes)
def test_remove_duplicates_empty_primary_keys_list():
    """Test that an empty list for primary_keys behaves the same as None."""
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'value': [10.0, 10.0, 20.0],  # Lignes id=1 identiques sur toutes les colonnes
    })
    result = remove_dataframe_duplicates(df, keep='first', primary_keys=[])
    # Comportement identique à primary_keys=None : toutes les colonnes → 1 doublon détecté
    assert len(result) == 2


# Test de l'absence de modification quand il n'y a pas de doublons
def test_remove_duplicates_no_duplicates():
    """Test that the DataFrame is unchanged when there are no duplicates."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10.0, 20.0, 30.0],
    })
    result = remove_dataframe_duplicates(df, keep='first')
    assert len(result) == 3


# Test du logging lorsque des doublons sont supprimés
def test_remove_duplicates_logging():
    """Test that a warning is logged when duplicates are removed."""
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'value': [10.0, 10.0, 20.0],
    })

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

    # Vérification qu'un message de warning a bien été émis
    assert len(records) == 1
    assert 'Suppression des doublons' in records[0].getMessage()
    assert 'TestSource' in records[0].getMessage()


# Test de l'absence de logging quand il n'y a pas de doublons
def test_remove_duplicates_no_logging_when_no_duplicates():
    """Test that nothing is logged when no duplicates are removed."""
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [10.0, 20.0, 30.0]})

    import logging as _logging
    test_logger = _logging.getLogger('test_no_log')
    records = []

    class ListHandler(_logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = ListHandler()
    test_logger.addHandler(handler)
    test_logger.setLevel(_logging.WARNING)

    remove_dataframe_duplicates(df, keep='first', logger=test_logger)

    test_logger.removeHandler(handler)
    assert len(records) == 0
