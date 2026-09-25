# Importation des modules
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import narwhals as nw
from narwhals.typing import IntoDataFrame

if TYPE_CHECKING:
    import duckdb


# Fonction de mise entre guillemets d'un identifiant SQL
def quote_ident(name: str) -> str:
    """
    Quote a SQL identifier following the SQL standard.

    Wraps ``name`` in double quotes and doubles any embedded double quote, so that
    an identifier derived from data (table name, column name) can be interpolated
    into a query without breaking it or allowing injection through the identifier.

    Args:
        name (str): Raw identifier (e.g. a column or table name).

    Returns:
        str: The double-quoted identifier.

    Examples:
        >>> quote_ident("fact_table")
        '"fact_table"'
        >>> quote_ident('weird"name')
        '"weird""name"'
    """
    # Doublage des guillemets internes puis encadrement (norme SQL)
    return '"' + str(name).replace('"', '""') + '"'


# Fonction de mise entre apostrophes d'un littéral de chaîne SQL
def quote_literal(value: str) -> str:
    """
    Quote a SQL string literal following the SQL standard.

    Wraps ``value`` in single quotes and doubles any embedded single quote, so that
    a value that cannot be passed as a bound parameter (arguments of ``ATTACH``,
    catalog and table names given to the ``ducklake_*`` table functions) can be
    interpolated into a statement without breaking it.

    Args:
        value (str): Raw value (e.g. a path, a table name or a timestamp).

    Returns:
        str: The single-quoted literal.

    Examples:
        >>> quote_literal("data/")
        "'data/'"
        >>> quote_literal("it's")
        "'it''s'"
    """
    # Doublage des apostrophes internes puis encadrement (norme SQL)
    return "'" + str(value).replace("'", "''") + "'"


# Fonction de résolution de l'alias de catalogue effectif d'une connexion
def resolve_catalog(
    conn: "duckdb.DuckDBPyConnection | None", catalog_alias: str | None
) -> str | None:
    """
    Return ``catalog_alias`` only when a database of that name is actually attached.

    Schema-aware managers always carry a ``catalog_alias`` (default ``'db'``), but
    in-memory test connections never attach such a catalog. Qualifying a table by a
    non-existent catalog would break every query, so the alias is used only when
    :func:`resolve_catalog` confirms it is attached; otherwise ``None`` is returned
    and callers fall back to schema-only qualification.

    Args:
        conn (duckdb.DuckDBPyConnection | None): Connection to inspect.
        catalog_alias (str | None): Candidate catalog alias.

    Returns:
        str | None: ``catalog_alias`` when attached, else ``None``.

    Examples:
        >>> import duckdb
        >>> resolve_catalog(duckdb.connect(":memory:"), "db") is None
        True
    """
    # Absence d'alias ou de connexion : rien à qualifier par le catalogue
    if catalog_alias is None or conn is None:
        return None
    try:
        # Recherche de l'alias parmi les bases attachées à la connexion
        row = conn.execute(
            "SELECT 1 FROM duckdb_databases() WHERE database_name = ?",
            [catalog_alias],
        ).fetchone()
    except Exception:
        return None
    return catalog_alias if row is not None else None


# Fonction de qualification d'un nom de table par son schéma (et son catalogue)
def qualify_table(table: str, schema: str = "main", catalog: str | None = None) -> str:
    """
    Build a schema- (and optionally catalog-) qualified SQL table identifier.

    A single DuckLake catalog may hold several schemas, each carrying its own
    ``fact_table``, ``metadata`` and ``dataset_metadata`` tables, so references
    must be schema-qualified. Without catalog qualification a query resolves against the
    connection's **current** catalog (the last ``USE``), which would silently
    write to the wrong database as soon as a second catalog is attached. When
    ``catalog`` is provided, the identifier is fully qualified by it; when it is
    ``None`` (in-memory test connections, which have no attached catalog), only
    the schema prefix is used. Every part is quoted via :func:`quote_ident`.

    Args:
        table (str): Bare table name (e.g. ``'fact_table'``, ``'metadata'``,
            ``'dataset_metadata'``).
        schema (str): Target DuckLake schema. Defaults to ``'main'``.
        catalog (str | None): Attached catalog alias. When ``None`` (default), the
            identifier is only schema-qualified.

    Returns:
        str: The ``"catalog"."schema"."table"`` identifier when ``catalog`` is
        given, otherwise ``"schema"."table"``.

    Examples:
        >>> qualify_table("fact_table")
        '"main"."fact_table"'
        >>> qualify_table("fact_table", "predictions")
        '"predictions"."fact_table"'
        >>> qualify_table("fact_table", "predictions", "db")
        '"db"."predictions"."fact_table"'
    """
    # Qualification par le catalogue lorsqu'un alias est fourni : indispensable pour
    # ne pas dépendre du catalogue courant de la connexion (dernier USE).
    if catalog is not None:
        return f"{quote_ident(catalog)}.{quote_ident(schema)}.{quote_ident(table)}"
    # Sinon, qualification par le seul schéma (connexions in-memory sans alias).
    return f"{quote_ident(schema)}.{quote_ident(table)}"


# Mixin des classes rattachées à un schéma (et un catalogue) DuckLake
class SchemaScoped:
    """Mixin giving schema-aware classes their shared table helpers.

    Hosts the single definition of the helpers every schema-aware class needs
    (managers, builder, auditor, recovery manager): qualifying a bare table name,
    checking a table exists and counting its rows. The host class must set
    ``conn``, ``schema`` and ``_catalog`` (the effective alias returned by
    :func:`resolve_catalog`) before calling them.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Connection used by the helpers.
        schema (str): Target DuckLake schema.
        _catalog (str | None): Attached catalog alias, ``None`` when no catalog of
            that name is attached (in-memory test connections).

    Examples:
        >>> import duckdb
        >>> class Reader(SchemaScoped):
        ...     def __init__(self, conn):
        ...         self.conn, self.schema, self._catalog = conn, "main", None
        >>> Reader(duckdb.connect())._qualified("fact_table")
        '"main"."fact_table"'
    """

    conn: "duckdb.DuckDBPyConnection"
    schema: str
    _catalog: str | None

    # Méthode de qualification d'un nom de table par le schéma (et le catalogue)
    def _qualified(self, table: str) -> str:
        """Return ``table`` qualified by this instance's schema and catalog.

        Args:
            table: Bare table name (e.g. ``'fact_table'``, ``'metadata'``).

        Returns:
            str: The quoted identifier, catalog-qualified only when an alias is
            actually attached.

        Examples:
            >>> manager._qualified("fact_table")
            '"main"."fact_table"'
        """
        return qualify_table(table, self.schema, self._catalog)

    # Méthode de vérification de l'existence d'une table dans le schéma
    def _table_exists(self, table_name: str) -> bool:
        """Check whether a bare-named table exists in this instance's schema.

        Filtering on the schema (and on the catalog when one is attached) matters:
        the same ``fact_table`` may exist in a neighbouring schema or catalog and
        would otherwise give a false positive.

        Args:
            table_name: Bare table name.

        Returns:
            bool: True if the table exists, False otherwise (including when the
            lookup itself fails).

        Examples:
            >>> manager._table_exists("fact_table")
            True
        """
        # Filtre optionnel sur le catalogue effectif
        catalog_filter = " AND table_catalog = ?" if self._catalog is not None else ""
        params: list[str] = [table_name, self.schema]
        if self._catalog is not None:
            params.append(self._catalog)
        try:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables"
                f" WHERE table_name = ? AND table_schema = ?{catalog_filter}",
                params,
            ).fetchone()
        except Exception:
            return False
        return row is not None and row[0] > 0

    # Méthode de comptage des lignes d'une table du schéma
    def _count_rows(self, table: str) -> int:
        """Count the rows of a bare-named table of this instance's schema.

        Args:
            table: Bare table name (e.g. ``'fact_table'``).

        Returns:
            int: Row count, or ``0`` if the table cannot be read (e.g. not created
            yet).

        Examples:
            >>> manager._count_rows("fact_table")
            3
        """
        try:
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._qualified(table)}"
            ).fetchone()
        except Exception:
            return 0
        return int(row[0]) if row is not None else 0


# Fonction de suppression des duplicats d'un jeu de données
def remove_dataframe_duplicates(
    df: IntoDataFrame,
    keep: Literal["any", "none", "first", "last"],
    logger: logging.Logger | None = None,
    source: str = "DataFrame",
    primary_keys: list[str] | None = None,
) -> nw.DataFrame[Any]:
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
        logger.warning(
            f"Removing duplicates from {source}: {removed_count} removed observations"
        )

    return df_cleaned


# Méthode auxiliaire de création d'un filtre de conjonction
def _build_conjonction_filter(filters: list[tuple[str, str, Any]]) -> str:
    """Constructs a SQL 'AND' filter condition from a list of filter tuples.

    Args:
        filters (List[Tuple[str, str, Any]]): A list of filter conditions, where each
        filter is represented as a tuple (column, operator, value).
        The operator can be comparison operators like '=', '!=', '<', '>', or set
        operators like 'in', 'not in'.

    Returns:
        str: A string representing the conjunction of all the filter conditions, joined
        with 'AND'.
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
    filters: list[tuple[str, str, Any]] | list[list[tuple[str, str, Any]]],
) -> str:
    """Constructs a SQL filter condition from a list of filter tuples
    or a list of lists of filter tuples.

    Args:
        filters (Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]]]):
        Filter syntax: [[(column, op, val), …],…] where op is [==, =, >, >=, <, <=, !=,
        in, not in].
        The innermost tuples are transposed into a set of filters applied through an AND
        operation.
        The outer list combines these sets of filters through an OR operation.
        A single list of tuples can also be used, meaning that no OR operation between
        set of filters is to be conducted.

    Raises:
        TypeError: If the filters are not provided in the expected format.

    Returns:
        str: A string representing the complete filter condition for the SQL query,
        either as a conjunction (AND) or
        a disjunction (OR) of filter conditions.
    """

    # Disjonction de cas suivant le type de l'argument "filters"
    # Si filters est une liste de tuples
    if all(isinstance(i, tuple) for i in filters) and isinstance(filters, list):
        return _build_conjonction_filter(
            filters=cast(list[tuple[str, str, Any]], filters)
        )
    # Si filters est une liste de liste de tuples
    elif all(
        isinstance(i, list) and all(isinstance(j, tuple) for j in i) for i in filters
    ) and isinstance(filters, list):
        # Calul indépendant de chaque filtre de conjonction
        conditions = [
            _build_conjonction_filter(
                filters=cast(list[tuple[str, str, Any]], conjonction_filter)
            )
            for conjonction_filter in filters
        ]
        return " OR ".join(conditions)
    # Cas d'erreur de typage
    else:
        raise TypeError(
            f"Invalid type for 'filters' : {filters}. Shoud be in [List[Tuple],"
            f"List[List[Tuple]]]"
        )


# Méthode de construction d'une requête SQL avec clause WHERE
def _build_where_clause(
    filters: list[tuple[str, str, Any]]
    | list[list[tuple[str, str, Any]]]
    | str
    | None
    | None = None,
) -> str:
    """Constructs a SQL SELECT WHERE clause based on the given filters.

    Args:
        filters (Optional[ Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str,
        Any]]], str, None] ], optional): A filter condition for the rows. Filter syntax:
        [[(column, op, val), …],…] where op is [=, >, >=, <, <=, !=, in, not in].
        The innermost tuples are transposed into a set of filters applied through an AND
        operation.
        The outer list combines these sets of filters through an OR operation.
        A single list of tuples can also be used, meaning that no OR operation between
        set of filters is to be conducted. Defaults to None.

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
        sql_request = f"WHERE {sql_filters}"
    else:
        raise TypeError("Invalid type for 'filters'. Should be in [list, str, None]")

    return sql_request
