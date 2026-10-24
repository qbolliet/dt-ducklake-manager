# Importation des modules
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import narwhals as nw
from narwhals.typing import IntoDataFrame

if TYPE_CHECKING:
    import duckdb
    import pyarrow as pa


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


# Fonction de lecture de la première ligne d'un résultat execute()
def fetch_one(cursor: "duckdb.DuckDBPyConnection") -> tuple[Any, ...] | None:
    """
    Fetch the first row of the pending ``execute()`` result, then close it.

    Since DuckDB 2.0, a result is only released once a fetch returns no rows: a
    bare ``fetchone()`` leaves it open, and the next ``register()`` or ``sql()``
    on the connection — ours or the caller's — raises ``ResourceInUseException``.
    ``complete()`` runs the rest of the result to its end (a no-op once it is
    exhausted) without failing an open transaction, unlike ``abort()``.

    Args:
        cursor (duckdb.DuckDBPyConnection): Connection (or cursor) returned by
            ``execute()``, holding the result to read.

    Returns:
        tuple[Any, ...] | None: The first row, or ``None`` for an empty result.

    Examples:
        >>> import duckdb
        >>> conn = duckdb.connect(":memory:")
        >>> fetch_one(conn.execute("SELECT 42"))
        (42,)
        >>> fetch_one(conn.execute("SELECT 1 WHERE false")) is None
        True
    """
    # Fetch
    row = cursor.fetchone()
    # Libération du résultat : la connexion redevient utilisable par sql()/register()
    cursor.complete()
    return row


# Fonction de lecture d'une requête en table Arrow
def fetch_arrow(conn: "duckdb.DuckDBPyConnection", query: str) -> "pa.Table":
    """
    Run ``query`` and return its whole result as a PyArrow table.

    Goes through a relation (``conn.sql(query).to_arrow_table()``): since DuckDB
    2.0, Arrow output on an ``execute()`` result copies the rows and warns. A
    relation, unlike ``execute()``, refuses to run while an earlier ``execute()``
    result is still open — e.g. a caller's ``fetchone()`` — so any such result is
    first completed, as the next ``execute()`` would have replaced it anyway.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection to query.
        query (str): SQL query without bound parameters (relations take none
            here).

    Returns:
        pa.Table: The full result.

    Examples:
        >>> import duckdb
        >>> fetch_arrow(duckdb.connect(":memory:"), "SELECT 1 AS a").num_rows
        1
    """
    # Libération d'un éventuel résultat execute() encore ouvert
    conn.complete()
    return conn.sql(query).to_arrow_table()


# Fonction d'enregistrement d'une vue temporaire sur des données Python
def register_view(conn: "duckdb.DuckDBPyConnection", name: str, data: Any) -> None:
    """
    Register ``data`` (Arrow table, pandas or Polars DataFrame) as view ``name``.

    Same as ``conn.register(name, data)``, except that an ``execute()`` result
    still open on the connection is completed first: since DuckDB 2.0,
    ``register()`` raises ``ResourceInUseException`` otherwise.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection to register the view on.
        name (str): View name.
        data (Any): Object accepted by ``DuckDBPyConnection.register``.

    Examples:
        >>> import duckdb, pyarrow as pa
        >>> conn = duckdb.connect(":memory:")
        >>> register_view(conn, "v", pa.table({"a": [1, 2]}))
        >>> conn.execute("SELECT count(*) FROM v").fetchall()
        [(2,)]
    """
    # Libération d'un éventuel résultat execute() encore ouvert
    conn.complete()
    conn.register(name, data)


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
        row = fetch_one(
            conn.execute(
                "SELECT 1 FROM duckdb_databases() WHERE database_name = ?",
                [catalog_alias],
            )
        )
    except Exception:
        return None
    return catalog_alias if row is not None else None


# Fonction de détection d'un catalogue DuckLake attaché
def is_ducklake_catalog(
    conn: "duckdb.DuckDBPyConnection | None", catalog_alias: str | None
) -> bool:
    """
    Tell whether ``catalog_alias`` is an attached **DuckLake** catalog.

    A plain DuckDB database attached under the same alias (or no attached database
    at all, as on in-memory test connections) supports neither the ``ducklake_*``
    procedures nor ``SORTED BY``: callers check this before using them, rather than
    letting the statement fail, since a failed statement aborts the enclosing
    DuckDB transaction.

    Args:
        conn (duckdb.DuckDBPyConnection | None): Connection to inspect.
        catalog_alias (str | None): Candidate catalog alias.

    Returns:
        bool: True when a database of that name is attached and is of type
        ``'ducklake'``, False otherwise (including when the lookup fails).

    Examples:
        >>> import duckdb
        >>> is_ducklake_catalog(duckdb.connect(":memory:"), "memory")
        False
        >>> is_ducklake_catalog(duckdb.connect(":memory:"), None)
        False
    """
    # Absence d'alias ou de connexion : aucun catalogue DuckLake possible
    if catalog_alias is None or conn is None:
        return False
    try:
        # Type de la base attachée sous cet alias ('ducklake', 'duckdb', ...)
        row = fetch_one(
            conn.execute(
                "SELECT type FROM duckdb_databases() WHERE database_name = ?",
                [catalog_alias],
            )
        )
    except Exception:
        return False
    return row is not None and row[0] == "ducklake"


# Fonction de construction de la clause SORTED BY d'une table DuckLake
def sorted_by_clause(columns: list[str]) -> str:
    """
    Build the DuckLake ``SORTED BY (...)`` clause of a list of sort columns.

    Used both in ``CREATE TABLE ... SORTED BY (...)`` and in ``ALTER TABLE ... SET
    SORTED BY (...)``. Every column is quoted via :func:`quote_ident` and sorted
    ascending (DuckLake's default, ``NULLS LAST``).

    Args:
        columns (list[str]): Non-empty list of column names, in sort order.

    Returns:
        str: The ``SORTED BY ("a", "b")`` clause.

    Raises:
        ValueError: If ``columns`` is empty (DuckLake rejects ``SORTED BY ()``;
            removing the sort key is ``RESET SORTED BY``).

    Examples:
        >>> sorted_by_clause(["date", "region"])
        'SORTED BY ("date", "region")'
    """
    # Clé de tri vide : rejetée par le parseur, la suppression passe par RESET
    if not columns:
        raise ValueError("columns must not be empty; use RESET SORTED BY instead")
    return f"SORTED BY ({', '.join(quote_ident(c) for c in columns)})"


# Séparateur des niveaux d'un chemin de schéma imbriqué ('sales.europe')
SCHEMA_PATH_SEPARATOR = "."


# Fonction de découpage d'un chemin de schéma en niveaux
def schema_parts(schema: str) -> tuple[str, ...]:
    """
    Split a schema path into its levels.

    Since DuckDB 2.0 / DuckLake 1.1, schemas can be nested (``CREATE SCHEMA
    db.sales.europe``). A nested schema is named by its levels joined with a dot
    (``'sales.europe'``), the same convention as the GraphQL API. The dot is
    therefore reserved as the level separator: a flat schema whose own name
    contains a dot cannot be targeted.

    Args:
        schema (str): Schema path (``'main'``, ``'sales.europe'``).

    Returns:
        tuple[str, ...]: The levels of the path, from the root to the leaf.

    Raises:
        ValueError: If ``schema`` is empty or has an empty level (``'a..b'``,
            ``'.a'``, ``'a.'``).

    Examples:
        >>> schema_parts("main")
        ('main',)
        >>> schema_parts("sales.europe")
        ('sales', 'europe')
    """
    # Extraction des chemins du schéma
    parts = tuple(str(schema).split(SCHEMA_PATH_SEPARATOR))
    # Niveau vide : chemin mal formé, rejeté
    if any(part == "" for part in parts):
        raise ValueError(
            f"Invalid schema path {schema!r}: levels are separated by"
            f" {SCHEMA_PATH_SEPARATOR!r} and must not be empty"
        )
    return parts


# Fonction de qualification d'un schéma (éventuellement imbriqué) par son catalogue
def qualify_schema(schema: str = "main", catalog: str | None = None) -> str:
    """
    Build the quoted SQL identifier of a (possibly nested) schema.

    Each level of the path is quoted on its own (``"db"."sales"."europe"``):
    quoting the whole path at once (``"sales.europe"``) would target a distinct,
    flat schema whose name contains a dot.

    Args:
        schema (str): Schema path (``'main'``, ``'sales.europe'``). Defaults to
            ``'main'``.
        catalog (str | None): Attached catalog alias. When ``None`` (default), the
            identifier is not catalog-qualified.

    Returns:
        str: The quoted, dot-joined identifier.

    Raises:
        ValueError: If the path is malformed (see :func:`schema_parts`), or if a
            nested schema is given without ``catalog``: DuckDB would read the
            first level as a catalog name (``sales.europe.t`` is ``catalog.schema.
            table``).

    Examples:
        >>> qualify_schema("predictions", "db")
        '"db"."predictions"'
        >>> qualify_schema("sales.europe", "db")
        '"db"."sales"."europe"'
        >>> qualify_schema("main")
        '"main"'
    """
    # Extraction du schéma
    parts = schema_parts(schema)
    # Schéma imbriqué sans catalogue : identifiant ambigu pour le parseur DuckDB
    if catalog is None and len(parts) > 1:
        raise ValueError(
            f"Nested schema {schema!r} requires an attached catalog: without it,"
            " its first level would be read as a catalog name"
        )
    prefix = [catalog] if catalog is not None else []
    return ".".join(quote_ident(part) for part in [*prefix, *parts])


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
    the schema prefix is used. Every part is quoted via :func:`quote_ident`, and a
    nested schema path (``'sales.europe'``) is quoted level by level (see
    :func:`qualify_schema`).

    Args:
        table (str): Bare table name (e.g. ``'fact_table'``, ``'metadata'``,
            ``'dataset_metadata'``).
        schema (str): Target DuckLake schema path. Defaults to ``'main'``.
        catalog (str | None): Attached catalog alias. When ``None`` (default), the
            identifier is only schema-qualified.

    Returns:
        str: The ``"catalog"."schema"."table"`` identifier when ``catalog`` is
        given, otherwise ``"schema"."table"``.

    Raises:
        ValueError: If the schema path is malformed, or nested without
            ``catalog`` (see :func:`qualify_schema`).

    Examples:
        >>> qualify_table("fact_table")
        '"main"."fact_table"'
        >>> qualify_table("fact_table", "predictions")
        '"predictions"."fact_table"'
        >>> qualify_table("fact_table", "predictions", "db")
        '"db"."predictions"."fact_table"'
        >>> qualify_table("fact_table", "sales.europe", "db")
        '"db"."sales"."europe"."fact_table"'
    """
    # Qualification par le catalogue lorsqu'un alias est fourni : indispensable pour
    # ne pas dépendre du catalogue courant de la connexion (dernier USE).
    return f"{qualify_schema(schema, catalog)}.{quote_ident(table)}"


# Fonction de création d'un schéma (et de ses parents) s'il n'existe pas
def create_schema(
    conn: "duckdb.DuckDBPyConnection", schema: str, catalog: str | None = None
) -> None:
    """
    Create a (possibly nested) schema and its missing parents.

    ``CREATE SCHEMA db.sales.europe`` fails when ``sales`` does not exist yet, so
    every level is created in turn, from the root to the leaf, with the
    idempotent ``CREATE SCHEMA IF NOT EXISTS``.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection with the catalog attached.
        schema (str): Schema path (``'main'``, ``'sales.europe'``).
        catalog (str | None): Catalog alias. Defaults to ``None`` (current catalog,
            flat schemas only).

    Raises:
        ValueError: If the path is malformed, or nested without ``catalog``.

    Examples:
        >>> import duckdb
        >>> conn = duckdb.connect(":memory:")
        >>> create_schema(conn, "predictions")
    """
    # Validation du chemin complet avant toute création : un échec à mi-chemin
    # laisserait des niveaux parents orphelins
    qualify_schema(schema, catalog)
    parts = schema_parts(schema)
    # Création niveau par niveau, de la racine vers la feuille
    for depth in range(1, len(parts) + 1):
        level = SCHEMA_PATH_SEPARATOR.join(parts[:depth])
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {qualify_schema(level, catalog)}")


# Fonction de résolution de l'oid DuckDB d'un schéma (éventuellement imbriqué)
def resolve_schema_oid(
    conn: "duckdb.DuckDBPyConnection", schema: str, catalog: str | None = None
) -> int | None:
    """
    Resolve the DuckDB ``oid`` of a schema from its full path.

    ``information_schema`` and ``duckdb_schemas().schema_name`` only report the
    **last** level of a nested schema (``europe`` for ``sales.europe``), so
    filtering on the name alone confuses ``sales.europe`` with ``hr.europe`` or
    with a top-level ``europe``. The path is walked level by level through
    ``duckdb_schemas().parent_schema_oid``; callers then filter
    ``duckdb_tables()`` / ``duckdb_columns()`` on ``schema_oid``.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection to inspect.
        schema (str): Schema path (``'main'``, ``'sales.europe'``).
        catalog (str | None): Catalog alias. When ``None``, the connection's
            current database is searched, the one an unqualified identifier
            resolves against.

    Returns:
        int | None: The schema ``oid``, or ``None`` when a level does not exist.

    Raises:
        ValueError: If the path is malformed.
        duckdb.Error: If the lookup itself fails.

    Examples:
        >>> import duckdb
        >>> resolve_schema_oid(duckdb.connect(":memory:"), "main") is not None
        True
        >>> resolve_schema_oid(duckdb.connect(":memory:"), "missing") is None
        True
    """
    # Filtre sur la base : alias explicite, sinon base courante de la connexion
    database_filter = "database_name = " + (
        "?" if catalog is not None else "current_database()"
    )
    oid: int | None = None
    # Descente niveau par niveau : le parent du premier niveau est NULL
    for part in schema_parts(schema):
        params: list[Any] = [part, oid] + ([catalog] if catalog is not None else [])
        row = fetch_one(
            conn.execute(
                "SELECT oid FROM duckdb_schemas() WHERE schema_name = ?"
                f" AND parent_schema_oid IS NOT DISTINCT FROM ? AND {database_filter}",
                params,
            )
        )
        if row is None:
            return None
        oid = int(row[0])
    return oid


# Fonction de résolution de l'identifiant DuckLake d'un schéma actif
def resolve_ducklake_schema_id(
    conn: "duckdb.DuckDBPyConnection", catalog_alias: str, schema: str
) -> int | None:
    """
    Resolve the DuckLake ``schema_id`` of an active schema from its full path.

    The internal ``__ducklake_metadata_<alias>.ducklake_schema`` table stores only
    the last level of a nested schema in ``schema_name``; its parent is given by
    ``parent_schema_id`` (DuckLake 1.1 specification). The path is walked level by
    level. A catalog still on the 1.0 specification has no ``parent_schema_id``
    column and holds flat schemas only: a nested path then resolves to ``None``.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection with the catalog attached.
        catalog_alias (str): Alias of the attached DuckLake catalog.
        schema (str): Schema path (``'main'``, ``'sales.europe'``).

    Returns:
        int | None: The ``schema_id``, or ``None`` when a level does not exist.

    Raises:
        ValueError: If the path is malformed.
        duckdb.Error: If the metadata catalog cannot be read (no DuckLake catalog
            attached under ``catalog_alias``).

    Examples:
        >>> resolve_ducklake_schema_id(conn, "db", "sales.europe")
        2
    """
    # Schéma
    schema_table = (
        f"{quote_ident(f'__ducklake_metadata_{catalog_alias}')}.ducklake_schema"
    )
    parts = schema_parts(schema)
    # Présence de la colonne parent_schema_id (spécification 1.1 uniquement), lue
    # par DESCRIBE : le catalogue de métadonnées n'apparaît pas dans
    # duckdb_columns(), et une requête en échec annulerait la transaction.
    columns = {
        str(row[0]) for row in conn.execute(f"DESCRIBE {schema_table}").fetchall()
    }
    nested = "parent_schema_id" in columns
    if not nested and len(parts) > 1:
        return None
    parent_filter = " AND parent_schema_id IS NOT DISTINCT FROM ?" if nested else ""
    schema_id: int | None = None
    # Descente niveau par niveau parmi les schémas actifs (sans snapshot de fin)
    for part in parts:
        params: list[Any] = [part] + ([schema_id] if nested else [])
        row = fetch_one(
            conn.execute(
                f"SELECT schema_id FROM {schema_table}"
                f" WHERE schema_name = ? AND end_snapshot IS NULL{parent_filter}",
                params,
            )
        )
        if row is None:
            return None
        schema_id = int(row[0])
    return schema_id


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
        schema (str): Target DuckLake schema path (``'sales.europe'`` for a
            nested schema).
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

    # Méthode de détection d'un catalogue DuckLake sous l'alias effectif
    def _is_ducklake(self) -> bool:
        """Tell whether this instance's tables live in an attached DuckLake catalog.

        Re-read at each call (one ``duckdb_databases()`` lookup) rather than cached:
        the connection can be attached or detached after the instance is built.

        Returns:
            bool: True when ``_catalog`` is an attached DuckLake catalog, False on
            a plain DuckDB database or an in-memory test connection.

        Examples:
            >>> manager._is_ducklake()
            True
        """
        return is_ducklake_catalog(self.conn, self._catalog)

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
        try:
            # Résolution du schéma par son chemin complet : information_schema ne
            # donne que le dernier niveau d'un schéma imbriqué.
            schema_oid = resolve_schema_oid(self.conn, self.schema, self._catalog)
            if schema_oid is None:
                return False
            row = fetch_one(
                self.conn.execute(
                    "SELECT COUNT(*) FROM duckdb_tables()"
                    " WHERE schema_oid = ? AND table_name = ?",
                    [schema_oid, table_name],
                )
            )
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
            row = fetch_one(
                self.conn.execute(f"SELECT COUNT(*) FROM {self._qualified(table)}")
            )
        except Exception:
            return 0
        return int(row[0]) if row is not None else 0

    # Méthode de lecture des types physiques des colonnes d'une table du schéma
    def _physical_column_types(self, table: str = "fact_table") -> dict[str, str]:
        """Read the physical SQL type of every column of a table of this schema.

        The type is the one DuckDB reports for the stored column (``DESCRIBE``), with
        its parameters: ``DECIMAL(10,2)``, not ``DECIMAL``. This is the value recorded
        in ``metadata.sql_type``.

        Args:
            table: Bare table name. Defaults to ``'fact_table'``.

        Returns:
            dict[str, str]: Column name -> physical SQL type, in column order.

        Raises:
            duckdb.Error: If the table cannot be described.

        Examples:
            >>> manager._physical_column_types()["score"]
            'DECIMAL(10,2)'
        """
        rows = self.conn.execute(f"DESCRIBE {self._qualified(table)}").fetchall()
        return {str(row[0]): str(row[1]) for row in rows}


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
