# Importation des modules
# Modules de base
import logging
from datetime import UTC, datetime

import narwhals as nw
import polars as pl

# Module de tests
import pytest
from narwhals.dtypes import DType

# Module du package à tester
from dt_ducklake_manager.utils.types import (
    ALLOWED_DEFAULT_AGGREGATIONS,
    METADATA_COLUMNS,
    check_supported_dtype,
    map_python_to_sql_type,
    metadata_table_ddl,
    normalize_default_aggregation,
    utc_now,
    warn_nonstandard_column_name,
)

# ---------------------------------------------------------------------------
# Tests des types textuels
# ---------------------------------------------------------------------------


# Test de l'association des types textuels vers VARCHAR
def test_string_maps_to_varchar() -> None:
    """Test that String type maps to VARCHAR.

    Examples:
        >>> map_python_to_sql_type(nw.String())
        'VARCHAR'
    """
    assert map_python_to_sql_type(nw.String()) == "VARCHAR"


# Test de l'association des types catégoriels et enum vers VARCHAR
def test_categorical_and_enum_map_to_varchar() -> None:
    """Test that Categorical and Enum types map to VARCHAR.

    Examples:
        >>> map_python_to_sql_type(nw.Categorical())
        'VARCHAR'
    """
    assert map_python_to_sql_type(nw.Categorical()) == "VARCHAR"


# ---------------------------------------------------------------------------
# Tests des types entiers signés
# ---------------------------------------------------------------------------


# Test de l'association des entiers signés vers leurs types SQL de largeur préservée
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (nw.Int8(), "TINYINT"),
        (nw.Int16(), "SMALLINT"),
        (nw.Int32(), "INTEGER"),
        (nw.Int64(), "BIGINT"),
    ],
)
def test_signed_int_maps_to_width_preserving_type(dtype: DType, expected: str) -> None:
    """Test that signed integer types map to their width-preserving SQL types.

    Args:
        dtype: Narwhals signed integer type.
        expected: Expected SQL type string.

    Examples:
        >>> map_python_to_sql_type(nw.Int32())
        'INTEGER'
    """
    assert map_python_to_sql_type(dtype) == expected


# Test de l'association de l'entier signé 128 bits vers HUGEINT
def test_int128_maps_to_hugeint() -> None:
    """Test that Int128 maps to HUGEINT.

    Examples:
        >>> map_python_to_sql_type(nw.Int128())
        'HUGEINT'
    """
    assert map_python_to_sql_type(nw.Int128()) == "HUGEINT"


# ---------------------------------------------------------------------------
# Tests des types entiers non signés
# ---------------------------------------------------------------------------


# Test de l'association des entiers non signés vers leurs types DuckDB dédiés
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (nw.UInt8(), "UTINYINT"),
        (nw.UInt16(), "USMALLINT"),
        (nw.UInt32(), "UINTEGER"),
        (nw.UInt64(), "UBIGINT"),
    ],
)
def test_unsigned_int_maps_to_correct_type(dtype: DType, expected: str) -> None:
    """Test that unsigned integer types map to their dedicated DuckDB types.

    Args:
        dtype: Narwhals unsigned integer type.
        expected: Expected SQL type string.

    Examples:
        >>> map_python_to_sql_type(nw.UInt8())
        'UTINYINT'
    """
    assert map_python_to_sql_type(dtype) == expected


# Test de l'association de l'entier non signé 128 bits vers UHUGEINT
def test_uint128_maps_to_uhugeint() -> None:
    """Test that UInt128 maps to UHUGEINT.

    Examples:
        >>> map_python_to_sql_type(nw.UInt128())
        'UHUGEINT'
    """
    assert map_python_to_sql_type(nw.UInt128()) == "UHUGEINT"


# ---------------------------------------------------------------------------
# Tests des types virgule flottante
# ---------------------------------------------------------------------------


# Test de l'association des types flottants vers leurs types SQL de largeur préservée
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (nw.Float32(), "FLOAT"),
        (nw.Float64(), "DOUBLE"),
    ],
)
def test_float_maps_to_width_preserving_type(dtype: DType, expected: str) -> None:
    """Test that Float32 maps to FLOAT and Float64 maps to DOUBLE.

    Args:
        dtype: Narwhals float type.
        expected: Expected SQL type string.

    Examples:
        >>> map_python_to_sql_type(nw.Float64())
        'DOUBLE'
    """
    assert map_python_to_sql_type(dtype) == expected


# Test de l'association du type décimal vers DECIMAL
def test_decimal_maps_to_decimal() -> None:
    """Test that Decimal maps to DECIMAL.

    Examples:
        >>> map_python_to_sql_type(nw.Decimal())
        'DECIMAL'
    """
    assert map_python_to_sql_type(nw.Decimal()) == "DECIMAL"


# ---------------------------------------------------------------------------
# Tests des types temporels
# ---------------------------------------------------------------------------


# Test de l'association des types temporels vers leurs équivalents SQL
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (nw.Date(), "DATE"),
        (nw.Datetime(), "TIMESTAMP"),
        (nw.Duration(), "INTERVAL"),
        (nw.Time(), "TIME"),
    ],
)
def test_temporal_types_mapping(dtype: DType, expected: str) -> None:
    """Test that temporal types map to their SQL equivalents.

    Args:
        dtype: Narwhals temporal type.
        expected: Expected SQL type string.

    Examples:
        >>> map_python_to_sql_type(nw.Date())
        'DATE'
    """
    assert map_python_to_sql_type(dtype) == expected


# ---------------------------------------------------------------------------
# Tests des types booléens et binaires
# ---------------------------------------------------------------------------


# Test de l'association du type booléen vers BOOLEAN
def test_boolean_maps_to_boolean() -> None:
    """Test that Boolean maps to BOOLEAN.

    Examples:
        >>> map_python_to_sql_type(nw.Boolean())
        'BOOLEAN'
    """
    assert map_python_to_sql_type(nw.Boolean()) == "BOOLEAN"


# Test de l'association du type binaire vers BLOB
def test_binary_maps_to_blob() -> None:
    """Test that Binary maps to BLOB.

    Examples:
        >>> map_python_to_sql_type(nw.Binary())
        'BLOB'
    """
    assert map_python_to_sql_type(nw.Binary()) == "BLOB"


# ---------------------------------------------------------------------------
# Tests des types composites
# ---------------------------------------------------------------------------


# Test de l'association des types composites vers VARCHAR (repli)
def test_list_maps_to_varchar() -> None:
    """Test that List type falls back to VARCHAR.

    Examples:
        >>> map_python_to_sql_type(nw.List(nw.String()))
        'VARCHAR'
    """
    assert map_python_to_sql_type(nw.List(nw.String())) == "VARCHAR"


def test_array_maps_to_varchar() -> None:
    """Test that Array type falls back to VARCHAR.

    Examples:
        >>> map_python_to_sql_type(nw.Array(nw.Int32(), 3))
        'VARCHAR'
    """
    assert map_python_to_sql_type(nw.Array(nw.Int32(), 3)) == "VARCHAR"


def test_struct_maps_to_varchar() -> None:
    """Test that Struct type falls back to VARCHAR.

    Examples:
        >>> map_python_to_sql_type(nw.Struct([]))
        'VARCHAR'
    """
    assert map_python_to_sql_type(nw.Struct([])) == "VARCHAR"


# ---------------------------------------------------------------------------
# Tests via inférence polars (cas d'utilisation réels)
# ---------------------------------------------------------------------------


# Test de l'inférence de type à partir d'un schéma polars réel
def test_map_via_polars_integer_schema() -> None:
    """Test type mapping via a real polars schema for integer columns.

    Examples:
        >>> df = pl.DataFrame({'col': [1, 2, 3]})  # polars infère Int64
        >>> map_python_to_sql_type(nw.from_native(df, eager_only=True).schema['col'])
        'BIGINT'
    """
    df = pl.DataFrame({"col": [1, 2, 3]})
    nw_df = nw.from_native(df, eager_only=True)
    assert map_python_to_sql_type(nw_df.schema["col"]) == "BIGINT"


# Test de l'inférence de type à partir d'un schéma polars pour les chaînes
def test_map_via_polars_string_schema() -> None:
    """Test type mapping via a real polars schema for string columns.

    Examples:
        >>> df = pl.DataFrame({'col': ['a', 'b']})
        >>> map_python_to_sql_type(nw.from_native(df, eager_only=True).schema['col'])
        'VARCHAR'
    """
    df = pl.DataFrame({"col": ["a", "b"]})
    nw_df = nw.from_native(df, eager_only=True)
    assert map_python_to_sql_type(nw_df.schema["col"]) == "VARCHAR"


# Test de l'inférence de type à partir d'un schéma polars pour les flottants
def test_map_via_polars_float_schema() -> None:
    """Test type mapping via a real polars schema for float columns.

    Examples:
        >>> df = pl.DataFrame({'col': [1.0, 2.0]})
        >>> map_python_to_sql_type(nw.from_native(df, eager_only=True).schema['col'])
        'DOUBLE'
    """
    df = pl.DataFrame({"col": [1.0, 2.0]})
    nw_df = nw.from_native(df, eager_only=True)
    assert map_python_to_sql_type(nw_df.schema["col"]) == "DOUBLE"


# ---------------------------------------------------------------------------
# Tests de normalize_default_aggregation()
# ---------------------------------------------------------------------------


# Test de la normalisation en majuscules d'une agrégation valide
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sum", "SUM"),
        ("Avg", "AVG"),
        ("MEDIAN", "MEDIAN"),
        ("mode", "MODE"),
        ("count", "COUNT"),
        ("min", "MIN"),
        ("max", "MAX"),
    ],
)
def test_normalize_default_aggregation_valid(raw: str, expected: str) -> None:
    """Test that a valid aggregation is upper-cased and returned.

    Args:
        raw: Case-insensitive aggregation label supplied by the producer.
        expected: Canonical upper-cased label.
    """
    assert normalize_default_aggregation(raw) == expected
    assert expected in ALLOWED_DEFAULT_AGGREGATIONS


# Test que None traverse la fonction sans validation (champ nullable)
def test_normalize_default_aggregation_none_passes_through() -> None:
    """Test that None is accepted unchanged since the field is nullable."""
    assert normalize_default_aggregation(None) is None


# Test qu'une agrégation inconnue lève une ValueError explicite
def test_normalize_default_aggregation_invalid_raises() -> None:
    """Test that an unsupported aggregation raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="Invalid default_aggregation 'TOTAL'"):
        normalize_default_aggregation("TOTAL")


# ---------------------------------------------------------------------------
# Tests de utc_now
# ---------------------------------------------------------------------------


# Test que l'horodatage est naïf et exprimé en UTC
def test_utc_now_is_naive_and_utc() -> None:
    """Test that utc_now returns a naive datetime in UTC, to the second.

    Examples:
        >>> utc_now().tzinfo is None
        True
    """
    before = datetime.now(UTC).replace(tzinfo=None)
    stamp = utc_now()
    after = datetime.now(UTC).replace(tzinfo=None)

    assert stamp.tzinfo is None
    assert before <= stamp <= after


# ---------------------------------------------------------------------------
# Tests de check_supported_dtype
# ---------------------------------------------------------------------------


# Test du refus de chaque type composite, avec le nom de la colonne dans le message
@pytest.mark.parametrize(
    "dtype",
    [
        nw.List(nw.Int64()),
        nw.List(nw.List(nw.String())),
        nw.Array(nw.Int64(), 2),
        nw.Struct({"a": nw.Int64()}),
    ],
)
def test_check_supported_dtype_refuses_composites(dtype: DType) -> None:
    """Test that List, Array and Struct types are refused, whatever the nesting.

    Args:
        dtype: Composite Narwhals type.
    """
    with pytest.raises(ValueError, match="'nested'.*composite type"):
        check_supported_dtype("nested", dtype)


# Test que les types plats sont acceptés
@pytest.mark.parametrize(
    "dtype",
    [
        nw.String(),
        nw.Int64(),
        nw.Float64(),
        nw.Decimal(),
        nw.Boolean(),
        nw.Datetime(),
        nw.Date(),
        nw.Binary(),
        nw.Categorical(),
    ],
)
def test_check_supported_dtype_accepts_flat_types(dtype: DType) -> None:
    """Test that flat types pass the composite check.

    Args:
        dtype: Flat Narwhals type.
    """
    # Aucune exception levée : la fonction ne renvoie rien
    check_supported_dtype("flat", dtype)


# Test du refus à partir d'un vrai schéma polars
def test_check_supported_dtype_on_polars_schema() -> None:
    """Test the refusal on dtypes coming from a real polars DataFrame."""
    df = nw.from_native(pl.DataFrame({"ok": [1], "tags": [["a"]]}), eager_only=True)
    check_supported_dtype("ok", df.schema["ok"])
    with pytest.raises(ValueError, match="tags"):
        check_supported_dtype("tags", df.schema["tags"])


# ---------------------------------------------------------------------------
# Tests de warn_nonstandard_column_name
# ---------------------------------------------------------------------------


# Test qu'un nom snake_case ne déclenche aucun avertissement
@pytest.mark.parametrize("name", ["a", "_a", "a_1", "commune_2024", "__", "x1_y2"])
def test_warn_nonstandard_column_name_silent_for_snake_case(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that a snake_case ASCII name is not flagged.

    Args:
        name: Valid column name.
        caplog: Log capture fixture.
    """
    warn_nonstandard_column_name(name, logging.getLogger("test_naming"))
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# Test qu'un nom hors convention déclenche un avertissement (sans refus)
@pytest.mark.parametrize(
    "name", ["Valeur", "année", "prix€", "1abc", "a b", "a-b", "a.b", "", "aB"]
)
def test_warn_nonstandard_column_name_warns(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that a name outside ``^[a-z_][a-z0-9_]*$`` is flagged, not refused.

    Args:
        name: Non-conforming column name.
        caplog: Log capture fixture.
    """
    warn_nonstandard_column_name(name, logging.getLogger("test_naming"))
    warnings_logged = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_logged) == 1
    assert repr(name) in warnings_logged[0].getMessage()


# ---------------------------------------------------------------------------
# Tests de la déclaration NOT NULL du contrat metadata
# ---------------------------------------------------------------------------


# Test que les cinq colonnes du contrat sont déclarées NOT NULL
@pytest.mark.parametrize(
    "column", ["name", "label", "sql_type", "is_categorical", "is_primary_key"]
)
def test_metadata_contract_columns_are_not_null(column: str) -> None:
    """Test that the five contract columns are declared NOT NULL, in the DDL too.

    Args:
        column: Contract column name.
    """
    assert "NOT NULL" in METADATA_COLUMNS[column]
    assert f"{column} " in metadata_table_ddl('"main"."metadata"')
    assert f"{column} {METADATA_COLUMNS[column]}" in metadata_table_ddl("t")


# Test que les champs d'UI restent nullables
def test_metadata_ui_columns_stay_nullable() -> None:
    """Test that the producer-owned UI fields carry no NOT NULL constraint."""
    ui = [
        c
        for c in METADATA_COLUMNS
        if c not in {"name", "label", "sql_type", "is_categorical", "is_primary_key"}
    ]
    assert ui
    assert all("NOT NULL" not in METADATA_COLUMNS[c] for c in ui)
