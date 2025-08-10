# Importation des modules
# Modules de base
import pandas as pd
from typing import Dict, Literal, Optional
# Logging
import logging

def map_python_to_sql_type(dtype: str) -> str:
    """
    Map Python data types to SQL-compatible data types.
    
    Args:
        dtype (str): The Python data type as a string.
    
    Returns:
        str: The corresponding SQL data type.
    
    Examples:
        >>> map_python_to_sql_type('object')
        'VARCHAR'
        >>> map_python_to_sql_type('int64')
        'INTEGER'
        >>> map_python_to_sql_type('float64')
        'DOUBLE'
    """
    # Dictionnaire des correspondances entre les types Python et SQL
    type_mapping = {
        'object': 'VARCHAR',
        'int64': 'INTEGER', 
        'float64': 'DOUBLE',
        'datetime64[ns]': 'TIMESTAMP',
        'bool': 'BOOLEAN'
    }
    return type_mapping.get(dtype, 'VARCHAR')


def remove_dataframe_duplicates(df: pd.DataFrame, 
                               keep: Literal[False, 'first', 'last'],
                               logger: Optional[logging.Logger] = None,
                               source: str = "DataFrame") -> pd.DataFrame:
    """
    Remove duplicates from a DataFrame excluding the 'value' column.
    
    Args:
        df (pd.DataFrame): DataFrame to process
        keep (Literal[False, 'first', 'last']): Strategy for keeping duplicates
        logger (Optional[logging.Logger]): Logger instance for tracking
        source (str): Data source identifier for logging
        
    Returns:
        pd.DataFrame: DataFrame without duplicates
        
    Examples:
        >>> df = pd.DataFrame({'A': [1, 1, 2], 'B': ['x', 'x', 'y']})
        >>> result = remove_dataframe_duplicates(df, keep='first')
        >>> len(result)
        2
    """
    # Comptage du nombre d'observations initial
    initial_count = len(df)
    
    # Identification des colonnes à vérifier (toutes sauf 'value' si elle existe)
    columns_to_check = [col for col in df.columns if col != 'value']
    
    # Suppression des doublons selon la stratégie choisie
    if keep == False:
        df_cleaned = df.drop_duplicates(subset=columns_to_check, keep=False)
    else:
        df_cleaned = df.drop_duplicates(subset=columns_to_check, keep=keep)
    
    # Comptage des observations supprimées et logging
    removed_count = initial_count - len(df_cleaned)
    if removed_count > 0 and logger:
        logger.warning(f"Suppression des doublons ({source}): {removed_count} observations supprimées")
    
    return df_cleaned


def build_database_duplicate_removal_query(columns_to_check: list, 
                                         keep: Literal[False, 'first', 'last'],
                                         table_name: str = 'fact_table') -> str:
    """
    Build SQL query for removing duplicates from a database table.
    
    Args:
        columns_to_check (list): List of column names to check for duplicates
        keep (Literal[False, 'first', 'last']): Strategy for keeping duplicates
        table_name (str): Name of the table to deduplicate
        
    Returns:
        str: SQL DELETE query for removing duplicates
        
    Examples:
        >>> query = build_database_duplicate_removal_query(['col1', 'col2'], 'first')
        >>> 'DELETE FROM fact_table' in query
        True
    """
    if not columns_to_check:
        return ""
    
    # Construction de la chaîne des colonnes
    columns_str = ', '.join(columns_to_check)
    
    if keep == False:
        # Suppression de tous les doublons
        return f"""
        DELETE FROM {table_name} 
        WHERE rowid NOT IN (
            SELECT MIN(rowid) 
            FROM {table_name} 
            GROUP BY {columns_str}
            HAVING COUNT(*) = 1
        )
        """
    else:
        # Conservation du premier ou dernier doublon
        order_clause = "ASC" if keep == 'first' else "DESC"
        return f"""
        DELETE FROM {table_name} 
        WHERE rowid NOT IN (
            SELECT rowid FROM (
                SELECT rowid, ROW_NUMBER() OVER (
                    PARTITION BY {columns_str} 
                    ORDER BY rowid {order_clause}
                ) as rn
                FROM {table_name}
            ) WHERE rn = 1
        )
        """


def check_categorical_threshold(values: pd.Series, threshold: int) -> bool:
    """
    Check if a pandas Series should be considered categorical based on unique value count.
    
    Args:
        values (pd.Series): Series to evaluate
        threshold (int): Maximum number of unique values for categorical classification
        
    Returns:
        bool: True if series should be categorical, False otherwise
        
    Examples:
        >>> series = pd.Series(['A', 'B', 'A', 'C'])
        >>> check_categorical_threshold(series, 5)
        True
        >>> check_categorical_threshold(series, 2)
        False
    """
    return (str(values.dtype) == 'object' and 
            values.nunique() <= threshold)