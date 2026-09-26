# Importation des modules
import logging
import re
from datetime import UTC, datetime
from typing import Any

import narwhals as nw

# Champs d'UI de la table metadata renseignés par le producteur de métadonnées.
# Tous VARCHAR nullable (NULL par défaut) ; un update de données ne les écrase jamais.
# parent_name porte la hiérarchie de colonnes : la colonne
# parente dans un menu à group-options / arbre de sélection.
# label_for, porté par une colonne de libellés, pointe vers la colonne de code métier
# dont elle restitue le libellé ; plusieurs colonnes de libellés (langues,
# libellé court/long) peuvent pointer vers le même code.
UI_METADATA_FIELDS: tuple[str, ...] = (
    "parent_name",
    "label_for",
    "unit",
    "display_format",
    "family",
    "description",
    "default_aggregation",
)

# Clés acceptées dans un sous-dictionnaire de ``column_metadata`` : les champs d'UI
# ci-dessus, plus le libellé d'affichage.
COLUMN_METADATA_KEYS: frozenset[str] = frozenset({"label", *UI_METADATA_FIELDS})

# Colonnes de la table metadata (nom -> type SQL et contraintes), dans l'ordre du DDL.
# Source unique du CREATE TABLE (builder, managers, recovery), des colonnes requises
# de l'auditeur et du schéma du DataFrame de métadonnées vide.
METADATA_COLUMNS: dict[str, str] = {
    "name": "VARCHAR NOT NULL",
    "label": "VARCHAR NOT NULL",
    "sql_type": "VARCHAR NOT NULL",
    "is_categorical": "BOOLEAN NOT NULL DEFAULT FALSE",
    "is_primary_key": "BOOLEAN NOT NULL DEFAULT FALSE",
    **{field: "VARCHAR" for field in UI_METADATA_FIELDS},
}

# Convention de nommage des colonnes : snake_case ASCII. Les autres noms sont acceptés
# (les identifiants sont toujours quotés) mais signalés par un avertissement, car ils
# restent pénibles pour les clients SQL et les URL.
COLUMN_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-z_][a-z0-9_]*$")

# Agrégations acceptées pour ``metadata.default_aggregation``, validées à l'écriture.
ALLOWED_DEFAULT_AGGREGATIONS: frozenset[str] = frozenset(
    {"SUM", "AVG", "MAX", "MIN", "COUNT", "MEDIAN", "MODE"}
)


# Fonction de génération du DDL de la table metadata
def metadata_table_ddl(qualified_name: str, if_not_exists: bool = False) -> str:
    """Build the ``CREATE TABLE`` statement of a ``metadata`` table.

    Args:
        qualified_name: Already quoted and qualified table identifier (e.g. the
            result of ``_qualified('metadata')``).
        if_not_exists: Whether to emit ``CREATE TABLE IF NOT EXISTS``. Defaults to
            False.

    Returns:
        str: The DDL statement, columns taken from :data:`METADATA_COLUMNS`.

    Examples:
        >>> metadata_table_ddl('"main"."metadata"').splitlines()[0]
        'CREATE TABLE "main"."metadata" ('
    """
    # Clause optionnelle d'idempotence
    clause = "IF NOT EXISTS " if if_not_exists else ""
    columns = ",\n".join(f"    {name} {sql}" for name, sql in METADATA_COLUMNS.items())
    return f"CREATE TABLE {clause}{qualified_name} (\n{columns}\n)"


# Fonction de construction d'un DataFrame de métadonnées vide et typé
def empty_metadata_frame() -> nw.DataFrame[Any]:
    """Return an empty, typed metadata DataFrame (pyarrow backend).

    Explicit dtypes keep the callers' boolean filters valid on an empty frame.

    Returns:
        nw.DataFrame: Zero-row frame with the :data:`METADATA_COLUMNS` columns.

    Examples:
        >>> empty_metadata_frame().columns[:3]
        ['name', 'label', 'sql_type']
    """
    # Correspondance des types SQL vers les types narwhals
    schema = {
        name: nw.Boolean() if sql.startswith("BOOLEAN") else nw.String()
        for name, sql in METADATA_COLUMNS.items()
    }
    return nw.from_dict({name: [] for name in schema}, schema=schema, backend="pyarrow")


# Fonction de normalisation et de validation de ``metadata.default_aggregation``
def normalize_default_aggregation(value: str | None) -> str | None:
    """
    Normalize and validate a ``default_aggregation`` value.

    ``None`` passes through unchanged (the field is nullable). A string is
    upper-cased and checked against :data:`ALLOWED_DEFAULT_AGGREGATIONS`.

    Args:
        value (str | None): Raw aggregation label supplied by the metadata
            producer.

    Returns:
        str | None: The upper-cased aggregation label, or ``None``.

    Raises:
        ValueError: If ``value`` is neither ``None`` nor one of the allowed
            aggregations (case-insensitive).

    Examples:
        >>> normalize_default_aggregation('sum')
        'SUM'
        >>> normalize_default_aggregation(None) is None
        True
        >>> normalize_default_aggregation('total')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: Invalid default_aggregation 'total'; ...
    """
    # Champ nullable : absence de valeur acceptée telle quelle
    if value is None:
        return None

    # Normalisation en majuscules puis contrôle d'appartenance à la liste blanche
    normalized = str(value).upper()
    if normalized not in ALLOWED_DEFAULT_AGGREGATIONS:
        raise ValueError(
            f"Invalid default_aggregation {value!r}; expected one of "
            f"{sorted(ALLOWED_DEFAULT_AGGREGATIONS)} or None"
        )
    return normalized


# Hiérarchie des types SQL : (niveau, largeur en bits).
# Le niveau ordonne les familles (booléen < entier < flottant < texte) ; la largeur
# ordonne les types d'un même niveau, garantissant qu'un BIGINT enregistré n'est
# jamais rétrogradé par un lot d'Int32.
# Les types non ordonnés (DECIMAL, DATE, TIMESTAMP, TIME, INTERVAL, BLOB) sont
# volontairement absents : aucun élargissement n'a de sens entre eux et les types
# ci-dessous, et les y rattacher promouvrait à tort une colonne temporelle en VARCHAR.
SQL_TYPE_RANK: dict[str, tuple[int, int]] = {
    # Niveau 1 : booléen
    "BOOLEAN": (1, 1),
    # Niveau 2 : entiers, ordonnés par largeur
    "TINYINT": (2, 8),
    "UTINYINT": (2, 8),
    "SMALLINT": (2, 16),
    "USMALLINT": (2, 16),
    "INTEGER": (2, 32),
    "UINTEGER": (2, 32),
    "BIGINT": (2, 64),
    "UBIGINT": (2, 64),
    "HUGEINT": (2, 128),
    "UHUGEINT": (2, 128),
    # Niveau 3 : virgule flottante
    "FLOAT": (3, 32),
    "DOUBLE": (3, 64),
    # Niveau 4 : texte
    "VARCHAR": (4, 0),
}

# Types entiers non signés : à largeur égale, ils ne contiennent pas leur homologue
# signé (INTEGER et UINTEGER se recouvrent partiellement seulement).
UNSIGNED_SQL_TYPES: frozenset[str] = frozenset(
    {"UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"}
)

# Promotion en cas de conflit de signe : plus petit type signé contenant à la fois
# le type signé et le type non signé de la largeur indiquée.
SIGNED_WIDENING: dict[int, str] = {
    8: "SMALLINT",
    16: "INTEGER",
    32: "BIGINT",
    64: "HUGEINT",
}


# Fonction d'horodatage UTC naïf
def utc_now() -> datetime:
    """Return the current UTC time as a naive datetime.

    ``dataset_metadata.updated_at`` is a ``TIMESTAMP`` (no time zone): the value is
    written in UTC, and the tzinfo is dropped so DuckDB stores it as is.

    Returns:
        datetime: Current UTC time, without ``tzinfo``.

    Examples:
        >>> utc_now().tzinfo is None
        True
    """
    return datetime.now(UTC).replace(tzinfo=None)


# Fonction de refus des types composites
def check_supported_dtype(column: str, dtype: nw.dtypes.DType) -> None:
    """Refuse the composite types (``List``, ``Array``, ``Struct``).

    The metadata contract records one flat physical SQL type per column, on which the
    API relies to filter, type and serialize. A nested column has no such type, and
    a silent fallback to ``VARCHAR`` would misdescribe it. The producer must convert
    the column explicitly (e.g. to a JSON string) before writing it.

    Args:
        column: Name of the column, used in the error message.
        dtype: The Narwhals data type of the column.

    Raises:
        ValueError: If ``dtype`` is a ``List``, ``Array`` or ``Struct``.

    Examples:
        >>> import narwhals as nw
        >>> check_supported_dtype('score', nw.Float64())
        >>> tags = nw.List(nw.String())
        >>> check_supported_dtype('tags', tags)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: Column 'tags' has the composite type List(String); ...
    """
    if isinstance(dtype, nw.Array | nw.List | nw.Struct):
        raise ValueError(
            f"Column {column!r} has the composite type {dtype}; List, Array and"
            " Struct columns are not supported. Convert it explicitly (e.g. to a"
            " JSON string) or drop it before writing"
        )


# Fonction d'avertissement sur les noms de colonnes hors snake_case
def warn_nonstandard_column_name(column: str, logger: logging.Logger) -> None:
    """Log a warning when a column name is not ``snake_case`` ASCII.

    The name is accepted anyway: identifiers are always quoted by the package. The
    warning only flags names that remain awkward for SQL clients and URLs.

    Args:
        column: Column name to check.
        logger: Logger receiving the warning.

    Examples:
        >>> import logging
        >>> warn_nonstandard_column_name('valeur_totale', logging.getLogger('doc'))
    """
    if not COLUMN_NAME_PATTERN.fullmatch(column):
        logger.warning(
            f"Column name {column!r} does not match {COLUMN_NAME_PATTERN.pattern};"
            " it is accepted but may be awkward for SQL clients and URLs"
        )


# Fonction associant les types narwhals à leur équivalent SQL
def map_python_to_sql_type(dtype: nw.dtypes.DType) -> str:
    """
    Map Narwhals data types to SQL-compatible data types.

    This is the *inferred* type, used to declare columns. The type recorded in
    ``metadata.sql_type`` is the physical one, read back from the table after the
    write (e.g. ``DECIMAL(10,2)`` where this function returns ``DECIMAL``).

    Integer and float widths are preserved: the package never silently narrows a
    column. It is up to the producer to supply a ``Float32`` (or a narrower integer)
    when 32 bits are deemed sufficient.

    Args:
        dtype (nw.DType): The Narwhals data type.

    Returns:
        str: The corresponding SQL data type.

    Examples:
        >>> import narwhals as nw
        >>> import polars as pl
        >>> df = pl.DataFrame({'col': ['a', 'b']})
        >>> map_python_to_sql_type(df.schema['col'])
        'VARCHAR'
        >>> df = pl.DataFrame({'col': [1, 2]})  # polars infère Int64
        >>> map_python_to_sql_type(df.schema['col'])
        'BIGINT'
        >>> df = pl.DataFrame({'col': [1.0, 2.0]})  # polars infère Float64
        >>> map_python_to_sql_type(df.schema['col'])
        'DOUBLE'
    """
    # Types textuels
    # String, Categorical et Enum sont tous stockés sous forme VARCHAR en SQL
    if isinstance(dtype, nw.String | nw.Categorical | nw.Enum):
        return "VARCHAR"

    # Entiers signés
    # Préservation de la largeur : chaque type narwhals conserve son type SQL dédié.
    # Int128 est mappé vers HUGEINT, le type entier 128 bits natif de DuckDB.
    elif isinstance(dtype, nw.Int8):
        return "TINYINT"
    elif isinstance(dtype, nw.Int16):
        return "SMALLINT"
    elif isinstance(dtype, nw.Int32):
        return "INTEGER"
    elif isinstance(dtype, nw.Int64):
        return "BIGINT"
    elif isinstance(dtype, nw.Int128):
        return "HUGEINT"

    # Entiers non signés
    # Chaque largeur de bit possède un type UNSIGNED dédié dans DuckDB
    elif isinstance(dtype, nw.UInt8):
        return "UTINYINT"
    elif isinstance(dtype, nw.UInt16):
        return "USMALLINT"
    elif isinstance(dtype, nw.UInt32):
        return "UINTEGER"
    elif isinstance(dtype, nw.UInt64):
        return "UBIGINT"
    elif isinstance(dtype, nw.UInt128):
        return "UHUGEINT"

    # Types virgule flottante
    # Préservation de la largeur : Float32 → FLOAT (32 bits), Float64 → DOUBLE (64 bits)
    elif isinstance(dtype, nw.Float32):
        return "FLOAT"
    elif isinstance(dtype, nw.Float64):
        return "DOUBLE"

    # Type décimal à précision fixe
    # On retourne DECIMAL sans précision ni échelle car ces paramètres ne sont
    # pas toujours disponibles au moment de la construction du schéma SQL.
    elif isinstance(dtype, nw.Decimal):
        return "DECIMAL"

    # Types temporels
    elif isinstance(dtype, nw.Date):
        return "DATE"
    elif isinstance(dtype, nw.Datetime):
        return "TIMESTAMP"
    elif isinstance(dtype, nw.Duration):
        return "INTERVAL"
    elif isinstance(dtype, nw.Time):
        return "TIME"

    # Type booléen
    elif isinstance(dtype, nw.Boolean):
        return "BOOLEAN"

    # Type binaire
    # BLOB est le type DuckDB pour les données binaires brutes
    elif isinstance(dtype, nw.Binary):
        return "BLOB"

    # Types composites (Array, List, Struct)
    # Refusés à l'écriture par check_supported_dtype ; le repli vers VARCHAR ne sert
    # qu'aux appels internes de simple inférence (ex. validation de label_for).
    elif isinstance(dtype, nw.Array | nw.List | nw.Struct):
        return "VARCHAR"

    # Cas de repli : Object, Unknown, et tout type non reconnu
    else:
        return "VARCHAR"


# Fonction de validation du dictionnaire ``column_metadata``
def validate_column_metadata(
    column_metadata: dict[str, dict[str, str]] | None,
    columns: list[str],
) -> dict[str, dict[str, str | None]]:
    """
    Validate and normalize a ``column_metadata`` mapping.

    Shared by the schema builder and the column-management operations
    (``DatabaseUpdater.update_database``, ``DatabaseUpdater.add_columns``): both
    accept a per-column mapping of UI fields and must reject the same malformed
    input the same way.

    Args:
        column_metadata: Mapping of column name to a sub-dictionary of UI fields
            (``label``, ``parent_name``, ``label_for``, ``unit``, ``display_format``,
            ``family``, ``description``, ``default_aggregation``), all keys optional.
            ``None`` yields an empty mapping. Note that ``parent_name``/``label_for``
            existence and consistency validation happen later, in the caller.
        columns: Column names the mapping may reference (e.g. the columns of the
            source DataFrame, or the columns actually being added).

    Returns:
        dict[str, dict[str, str | None]]: The mapping with ``default_aggregation``
        upper-cased, ready to be consumed by the caller.

    Raises:
        ValueError: If a referenced column is absent from ``columns``, if a
            sub-dictionary carries an unknown key, or if ``default_aggregation`` is
            not an allowed value.

    Examples:
        >>> validate_column_metadata({'a': {'unit': '€'}}, ['a'])
        {'a': {'unit': '€'}}
    """
    # Absence de métadonnées d'UI : mapping vide
    if not column_metadata:
        return {}

    # Vérification de l'existence des colonnes référencées
    unknown_cols = set(column_metadata) - set(columns)
    if unknown_cols:
        raise ValueError(
            f"The following column_metadata columns do not exist in the DataFrame: "
            f"{sorted(unknown_cols)}"
        )

    # Contrôle des clés de chaque sous-dictionnaire et normalisation de l'agrégation
    normalized: dict[str, dict[str, str | None]] = {}
    for col, fields in column_metadata.items():
        unknown_keys = set(fields) - COLUMN_METADATA_KEYS
        if unknown_keys:
            raise ValueError(
                f"Unknown column_metadata key(s) for column {col!r}: "
                f"{sorted(unknown_keys)}; allowed keys are "
                f"{sorted(COLUMN_METADATA_KEYS)}"
            )
        col_fields: dict[str, str | None] = dict(fields)
        if "default_aggregation" in col_fields:
            col_fields["default_aggregation"] = normalize_default_aggregation(
                col_fields["default_aggregation"]
            )
        normalized[col] = col_fields

    return normalized


# Fonction de résolution d'un conflit de types SQL entre la base et un lot entrant
def resolve_sql_type_conflict(current: str, new: str) -> str | None:
    """
    Return the SQL type to store when a batch type differs from the stored one.

    Widening only: the recorded type is never narrowed. Within a level the greater
    width wins, so a stored ``BIGINT`` survives a batch of ``Int32``. At equal
    width, a signed/unsigned clash is promoted to the smallest signed type holding
    both. Non-ordered types (``DECIMAL``, ``DATE``, ``TIMESTAMP``, ``TIME``,
    ``INTERVAL``, ``BLOB``) have no ordering and always keep the stored type.

    Args:
        current (str): SQL type currently recorded in ``metadata.sql_type``.
        new (str): SQL type inferred from the incoming batch.

    Returns:
        str | None: The SQL type to write, or ``None`` when the stored type must
        be kept.

    Examples:
        >>> resolve_sql_type_conflict('BIGINT', 'INTEGER') is None
        True
        >>> resolve_sql_type_conflict('INTEGER', 'BIGINT')
        'BIGINT'
        >>> resolve_sql_type_conflict('INTEGER', 'UINTEGER')
        'BIGINT'
        >>> resolve_sql_type_conflict('INTEGER', 'VARCHAR')
        'VARCHAR'
        >>> resolve_sql_type_conflict('TIMESTAMP', 'BIGINT') is None
        True
    """
    # Types identiques : aucun conflit à résoudre
    if current == new:
        return None

    # Rangs respectifs des deux types
    current_rank = SQL_TYPE_RANK.get(current)
    new_rank = SQL_TYPE_RANK.get(new)

    # Type non ordonné d'un côté ou de l'autre : conservation du type enregistré
    if current_rank is None or new_rank is None:
        return None

    # Comparaison lexicographique du couple (niveau, largeur)
    if new_rank > current_rank:
        return new
    if new_rank < current_rank:
        return None

    # Niveau et largeur identiques : conflit de signe, promotion au type signé
    # immédiatement supérieur (None en 128 bits, aucun entier plus large n'existant)
    if (current in UNSIGNED_SQL_TYPES) != (new in UNSIGNED_SQL_TYPES):
        return SIGNED_WIDENING.get(current_rank[1])

    return None
