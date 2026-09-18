# Importation des modules
# Modules de base
import narwhals as nw
import polars as pl

# Module de tests
import pytest
from narwhals.dtypes import DType

# Module du package à tester
from dt_ducklake_manager.utils.types import (
    ALLOWED_DEFAULT_AGGREGATIONS,
    map_python_to_sql_type,
    normalize_default_aggregation,
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
