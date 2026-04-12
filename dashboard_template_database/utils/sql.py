# Importation des modules
import narwhals as nw
from narwhals.typing import IntoDataFrame
from typing import Any, Tuple, List, Literal, Optional, Union
import logging

# Fonction de suppression des duplicats d'un jeu de données
def remove_dataframe_duplicates(df: IntoDataFrame,
                               keep: Literal['any', 'none', 'first', 'last'],
                               logger: Optional[logging.Logger] = None,
                               source: str = "DataFrame",
                               primary_keys: Optional[List[str]] = None) -> nw.DataFrame :
    """
    Remove duplicates from a DataFrame based on primary keys or all columns.

    Args:
        df: DataFrame to process (pandas, polars, or any narwhals-compatible format)
        keep (Literal['any', 'none', 'first', 'last']): Strategy for keeping duplicates
        logger (Optional[logging.Logger]): Logger instance for tracking
        source (str): Data source identifier for logging
        primary_keys (Optional[List[str]]): List of column names to use as the
            duplicate detection key. If provided and non-empty, only these columns
            are used to identify duplicates. If None or empty, all columns are used.

    Returns:
        nw.DataFrame: DataFrame without duplicates

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({'A': [1, 1, 2], 'B': ['x', 'x', 'y']})
        >>> result = remove_dataframe_duplicates(df, keep='first')
        >>> len(result)
        2

        >>> df = pl.DataFrame({'id': [1, 1, 2], 'value': [10, 99, 30]})
        >>> result = remove_dataframe_duplicates(df, keep='first', primary_keys=['id'])
        >>> len(result)
        2
    """
    # Conversion vers narwhals
    df_nw = nw.from_native(df, eager_only=True)

    # Comptage du nombre d'observations initial
    initial_count = len(df_nw)

    # Sélection des colonnes servant à identifier les doublons :
    # - Si des clés primaires sont fournies, on les utilise exclusivement
    # - Sinon, toutes les colonnes du DataFrame sont utilisées
    if primary_keys:
        columns_to_check = primary_keys
    else:
        columns_to_check = list(df_nw.columns)

    # Suppression des doublons selon la stratégie choisie
    df_cleaned = df_nw.unique(subset=columns_to_check, keep=keep)

    # Comptage des observations supprimées et logging
    removed_count = initial_count - len(df_cleaned)
    if removed_count > 0 and logger:
        logger.warning(f"Removing duplicates from {source}: {removed_count} removed observations")

    return df_cleaned

# Fonction de construction d'une requête SQL de suppression des duplicats en base
def build_database_duplicate_removal_query(columns_to_check: list,
                                         keep: Literal['any', 'none', 'first', 'last'],
                                         table_name: str = 'fact_table') -> str:
    """
    Build SQL query for removing duplicates from a database table.

    Args:
        columns_to_check (list): List of column names to check for duplicates
        keep (Literal['any', 'none', 'first', 'last']): Strategy for keeping duplicates
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

    if keep == 'none':
        # Suppression de TOUTES les occurrences des clés en doublon.
        # On identifie les groupes ayant plus d'une ligne par leur valeur de colonnes.
        return f"""
        DELETE FROM {table_name}
        WHERE ({columns_str}) IN (
            SELECT {columns_str}
            FROM {table_name}
            GROUP BY {columns_str}
            HAVING COUNT(*) > 1
        )
        """
    else:
        # Conservation du premier ou dernier doublon selon l'ordre de scan DuckLake.
        # rowid est utilisé comme discriminant au sein d'une seule instruction DML :
        # il est stable dans ce contexte (file + row-group offset Parquet, cohérent
        # pour toute la durée du DELETE). Il n'est pas supposé stable entre sessions.
        # 'any' est traité comme 'first' (conservation d'une occurrence arbitraire).
        order_clause = "ASC" if keep in ('first', 'any') else "DESC"
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

# Méthode auxiliaire de création d'un filtre de conjonction
def _build_conjonction_filter(filters: List[Tuple[str, str, Any]]) -> str:
    """Constructs a SQL 'AND' filter condition from a list of filter tuples.

    Args:
        filters (List[Tuple[str, str, Any]]): A list of filter conditions, where each filter is represented as a tuple (column, operator, value).
        The operator can be comparison operators like '=', '!=', '<', '>', or set operators like 'in', 'not in'.

    Returns:
        str: A string representing the conjunction of all the filter conditions, joined with 'AND'.
    """
    # Initialisation de la condition
    conditions = []
    # Parcours des filtres
    for column, operator, value in filters:
        # Distinction suivant le type d'opération
        if operator in ["in", "not in"]:
            value = "(" + ", ".join(map(str, value)) + ")"
            # Ajout des filtres
            conditions.append(f"{column} {operator.upper()} {value}")
        else:
            # Ajout des filtres
            conditions.append(f"{column} {operator.upper()} '{value}'")

    # Retourne la conjonction des conditions
    return " AND ".join(conditions)


# Méthode de création des filtres
def _build_sql_filter(
    filters: Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]]],
) -> str:
    """Constructs a SQL filter condition from a list of filter tuples or a list of lists of filter tuples.

    Args:
        filters (Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]]]): Filter syntax: [[(column, op, val), …],…] where op is [==, =, >, >=, <, <=, !=, in, not in].
        The innermost tuples are transposed into a set of filters applied through an AND operation.
        The outer list combines these sets of filters through an OR operation.
        A single list of tuples can also be used, meaning that no OR operation between set of filters is to be conducted.

    Raises:
        TypeError: If the filters are not provided in the expected format.

    Returns:
        str: A string representing the complete filter condition for the SQL query, either as a conjunction (AND) or
        a disjunction (OR) of filter conditions.
    """

    # Disjonction de cas suivant le type de l'argument "filters"
    # Si filters est une liste de tuples
    if all(isinstance(i, tuple) for i in filters) and isinstance(filters, list):
        return _build_conjonction_filter(filters=filters)
    # Si filters est une liste de liste de tuples
    elif all(
        isinstance(i, list) and all(isinstance(j, tuple) for j in i) for i in filters
    ) and isinstance(filters, list):
        # Calul indépendant de chaque filtre de conjonction
        conditions = [
            _build_conjonction_filter(filters=conjonction_filter)
            for conjonction_filter in filters
        ]
        return " OR ".join(conditions)
    # Cas d'erreur de typage
    else:
        raise TypeError(
            f"Invalid type for 'filters' : {filters}. Shoud be in [List[Tuple], List[List[Tuple]]]"
        )

# Méthode de construction d'une requête SQL avec clause WHERE
def _build_where_clause(
    filters: Optional[
        Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]], str, None]
    ] = None,
) -> str:
    """Constructs a SQL SELECT WHERE clause based on the given filters.

    Args:
        filters (Optional[ Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]], str, None] ], optional): A filter condition for the rows. Filter syntax: [[(column, op, val), …],…] where op is [=, >, >=, <, <=, !=, in, not in].
        The innermost tuples are transposed into a set of filters applied through an AND operation.
        The outer list combines these sets of filters through an OR operation.
        A single list of tuples can also be used, meaning that no OR operation between set of filters is to be conducted. Defaults to None.

    Raises:
        TypeError: If the filters are not provided in the expected format.

    Returns:
        str: A SQL SELECT query string with optional row filters.
    """
    # Computation de la requête SQL
    if filters is None:
        return ""
    elif isinstance(filters, str):
        # Computation de la commande
        sql_request = f"WHERE {filters}"
    elif isinstance(filters, list):
        # Computation du filtre sur les lignes
        sql_filters = _build_sql_filter(filters=filters)
        # Computation de la commande
        sql_request = (
            f"WHERE {sql_filters}"
        )
    else :
        raise TypeError("Invalid type for 'filters'. Should be in [list, str, None]")

    return sql_request
