# Importation des modules
# Modules de base
import logging
import warnings
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

# DuckDB
import duckdb
import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.operations import DatabaseUpdater
from dt_ducklake_manager.operations._base import BaseSchemaManager
from dt_ducklake_manager.schema import DuckLakeTablesBuilder

# ---------------------------------------------------------------------------
# Fixture locale
# ---------------------------------------------------------------------------


# Initialisation d'une instance de BaseSchemaManager pour les tests
@pytest.fixture
def manager(built_ducklake_schema: duckdb.DuckDBPyConnection) -> BaseSchemaManager:
    """Create a BaseSchemaManager on the built in-memory schema.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.

    Returns:
        BaseSchemaManager: initialized with categorical_threshold=4.
    """
    return BaseSchemaManager(connection=built_ducklake_schema, categorical_threshold=4)


# ---------------------------------------------------------------------------
# Outils locaux
# ---------------------------------------------------------------------------


# Lecture des types enregistrés et physiques
def _types(conn: duckdb.DuckDBPyConnection) -> tuple[dict[str, str], dict[str, str]]:
    """Return the recorded (``metadata``) and physical (``DESCRIBE``) column types.

    Args:
        conn: Connection holding a built ``main`` schema.

    Returns:
        tuple: ``(recorded, physical)``, each mapping column name to SQL type.
    """
    recorded = dict(conn.execute("SELECT name, sql_type FROM metadata").fetchall())
    physical = {r[0]: r[1] for r in conn.execute("DESCRIBE fact_table").fetchall()}
    return recorded, physical


# Construction d'une base à clé primaire simple à partir d'un DataFrame
def _built_updater(df: pl.DataFrame) -> DatabaseUpdater:
    """Build a schema keyed on ``id`` from ``df`` and return an updater on it.

    Args:
        df: Source DataFrame with an ``id`` column.

    Returns:
        DatabaseUpdater: updater bound to the freshly built in-memory schema.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(df, primary_keys=["id"])
    builder.build_schema()
    return DatabaseUpdater(connection=builder.conn, categorical_threshold=4)


# Instantané de l'état (colonnes de la table, lignes de metadata)
def _snapshot(conn: duckdb.DuckDBPyConnection) -> tuple[list[Any], list[Any]]:
    """Return the fact table columns and the metadata rows, for before/after checks.

    Args:
        conn: Connection holding a built ``main`` schema.

    Returns:
        tuple: ``(DESCRIBE rows, metadata rows)``.
    """
    return (
        conn.execute("DESCRIBE fact_table").fetchall(),
        conn.execute("SELECT * FROM metadata ORDER BY name").fetchall(),
    )


# Instances de colonnes composites pour les tests de refus
COMPOSITES = [
    pytest.param(pl.Series("nested", [["a"], ["b"]]), id="list"),
    pytest.param(
        pl.Series("nested", [[1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2)), id="array"
    ),
    pytest.param(pl.Series("nested", [{"x": 1}, {"x": 2}]), id="struct"),
]


# ---------------------------------------------------------------------------
# Tests de add_columns : type physique
# ---------------------------------------------------------------------------


# Test que add_columns enregistre le type relu dans la table
def test_add_columns_records_physical_decimal_type(
    updater: DatabaseUpdater, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that add_columns records the stored DECIMAL type, parameters included.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
    """
    df = pl.DataFrame(
        {"id": [1, 2], "price": [Decimal("1.25"), Decimal("2.50")]},
        schema={"id": pl.Int64, "price": pl.Decimal(10, 2)},
    )
    updater.add_columns(df)

    recorded, physical = _types(built_ducklake_schema)
    assert recorded["price"] == physical["price"]
    assert recorded["price"].startswith("DECIMAL(")


# Test que les types non paramétrés restent inchangés
def test_add_columns_keeps_plain_type_names(
    updater: DatabaseUpdater, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that a plain type (DOUBLE, VARCHAR) is recorded as is.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
    """
    df = pl.DataFrame({"id": [1], "score": [1.5], "tag": ["x"]})
    updater.add_columns(df)

    recorded, _ = _types(built_ducklake_schema)
    assert recorded["score"] == "DOUBLE"
    assert recorded["tag"] == "VARCHAR"


# Test avec un nom accentué : accepté, typé et signalé
def test_add_columns_accented_name_is_accepted_and_flagged(
    updater: DatabaseUpdater,
    built_ducklake_schema: duckdb.DuckDBPyConnection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a non-snake_case new column is added with a warning.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
        caplog: Log capture fixture.
    """
    df = pl.DataFrame({"id": [1, 2], "Prix Total €": [1.5, 2.5]})
    with caplog.at_level(logging.WARNING):
        updater.add_columns(df)

    recorded, physical = _types(built_ducklake_schema)
    assert recorded["Prix Total €"] == physical["Prix Total €"] == "DOUBLE"
    flagged = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Prix Total €" in m for m in flagged)


# Test qu'un nom conforme ne déclenche aucun avertissement de nommage
def test_add_columns_snake_case_name_does_not_warn(
    updater: DatabaseUpdater, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that a snake_case new column raises no naming warning.

    Args:
        updater: DatabaseUpdater fixture.
        caplog: Log capture fixture.
    """
    with caplog.at_level(logging.WARNING):
        updater.add_columns(pl.DataFrame({"id": [1], "score_2024": [1.0]}))

    assert not [r for r in caplog.records if "does not match" in r.getMessage()]


# ---------------------------------------------------------------------------
# Tests de update_database : nouvelles colonnes et élargissement
# ---------------------------------------------------------------------------


# Test du type physique d'une colonne ajoutée par update_database
def test_update_database_new_column_records_physical_type(
    updater: DatabaseUpdater, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that a column added through allow_new_columns records its stored type.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
    """
    df = pl.DataFrame(
        {"id": [10], "category": ["A"], "price": [Decimal("3.75")]},
        schema={"id": pl.Int64, "category": pl.String, "price": pl.Decimal(12, 3)},
    )
    assert updater.update_database(df, allow_new_columns=True) is True

    recorded, physical = _types(built_ducklake_schema)
    assert recorded["price"] == physical["price"]
    assert recorded["price"].startswith("DECIMAL(")


# Test de l'élargissement de type : le type enregistré est le type physique
def test_update_database_widening_records_physical_type() -> None:
    """Test that a widened integer column records the widened stored type."""
    df = pl.DataFrame(
        {"id": [1, 2], "n": [1, 2]}, schema={"id": pl.Int64, "n": pl.Int16}
    )
    updater = _built_updater(df)
    assert _types(updater.conn)[0]["n"] == "SMALLINT"

    batch = pl.DataFrame(
        {"id": [3], "n": [70_000]}, schema={"id": pl.Int64, "n": pl.Int32}
    )
    assert updater.update_database(batch) is True

    recorded, physical = _types(updater.conn)
    assert recorded["n"] == physical["n"] == "INTEGER"


# Test de l'élargissement vers VARCHAR
def test_update_database_widening_to_varchar_records_physical_type() -> None:
    """Test that widening a numeric column to text records VARCHAR."""
    updater = _built_updater(pl.DataFrame({"id": [1, 2], "v": [1, 2]}))

    batch = pl.DataFrame({"id": [3], "v": ["abc"]})
    assert updater.update_database(batch) is True

    recorded, physical = _types(updater.conn)
    assert recorded["v"] == physical["v"] == "VARCHAR"


# Test qu'un lot DECIMAL sur une colonne DECIMAL(p,s) conserve le type stocké
def test_update_database_decimal_batch_keeps_stored_parameters() -> None:
    """Test that a DECIMAL batch never rewrites the stored DECIMAL(p,s) type."""
    df = pl.DataFrame(
        {"id": [1], "price": [Decimal("1.25")]},
        schema={"id": pl.Int64, "price": pl.Decimal(10, 2)},
    )
    updater = _built_updater(df)
    before = _types(updater.conn)[0]["price"]

    batch = pl.DataFrame(
        {"id": [2], "price": [Decimal("9.99")]},
        schema={"id": pl.Int64, "price": pl.Decimal(10, 2)},
    )
    assert updater.update_database(batch) is True

    recorded, physical = _types(updater.conn)
    assert recorded["price"] == before == physical["price"]


# ---------------------------------------------------------------------------
# Tests du refus des types composites
# ---------------------------------------------------------------------------


# Test du refus dans add_columns, sans effet de bord
@pytest.mark.parametrize("series", COMPOSITES)
def test_add_columns_refuses_composite(
    updater: DatabaseUpdater,
    built_ducklake_schema: duckdb.DuckDBPyConnection,
    series: pl.Series,
) -> None:
    """Test that add_columns refuses a composite column and leaves no trace.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
        series: Composite series to add.
    """
    before = _snapshot(built_ducklake_schema)
    df = pl.DataFrame({"id": [1, 2]}).with_columns(series)

    with pytest.raises(ValueError, match="composite type"):
        updater.add_columns(df)

    assert _snapshot(built_ducklake_schema) == before


# Test du refus dans add_columns pour une colonne existante écrasée
def test_add_columns_refuses_composite_on_overwrite(
    updater: DatabaseUpdater, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that overwriting an existing column with a list is refused.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
    """
    before = _snapshot(built_ducklake_schema)
    df = pl.DataFrame({"id": [1, 2], "category": [["a"], ["b"]]})

    with pytest.raises(ValueError, match="composite type"):
        updater.add_columns(df, overwrite=True)

    assert _snapshot(built_ducklake_schema) == before


# Test du refus dans update_database pour une nouvelle colonne
@pytest.mark.parametrize("series", COMPOSITES)
def test_update_database_refuses_composite_new_column(
    updater: DatabaseUpdater,
    built_ducklake_schema: duckdb.DuckDBPyConnection,
    series: pl.Series,
) -> None:
    """Test that a composite new column is refused and the update rolled back.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
        series: Composite series carried by the batch.
    """
    before = _snapshot(built_ducklake_schema)
    rows_before = built_ducklake_schema.execute(
        "SELECT count(*) FROM fact_table"
    ).fetchone()
    df = pl.DataFrame({"id": [10, 11]}).with_columns(series)

    with pytest.raises(ValueError, match="composite type"):
        updater.update_database(df, allow_new_columns=True)

    assert _snapshot(built_ducklake_schema) == before
    assert (
        built_ducklake_schema.execute("SELECT count(*) FROM fact_table").fetchone()
        == rows_before
    )


# Test du refus dans update_database pour une colonne existante
def test_update_database_refuses_composite_on_existing_column(
    updater: DatabaseUpdater, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that a list batch for an existing scalar column is refused.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
    """
    before = _snapshot(built_ducklake_schema)
    df = pl.DataFrame({"id": [10], "category": [["A"]]})

    with pytest.raises(ValueError, match="composite type"):
        updater.update_database(df)

    assert _snapshot(built_ducklake_schema) == before


# Test du refus direct dans _add_column_to_metadata
def test_add_column_to_metadata_refuses_composite(manager: BaseSchemaManager) -> None:
    """Test that the metadata writer itself refuses a composite column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="composite type"):
        manager._add_column_to_metadata("tags", pl.DataFrame({"tags": [["a"]]}))


# ---------------------------------------------------------------------------
# Tests de _add_column_to_metadata : repli et type physique
# ---------------------------------------------------------------------------


# Test du repli sur le type inféré pour une colonne absente de la table des faits
def test_add_column_to_metadata_falls_back_to_inferred_type(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that a column absent from the fact table records the inferred type.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    manager._add_column_to_metadata("ghost", pl.DataFrame({"ghost": [1.5]}))

    assert _types(built_ducklake_schema)[0]["ghost"] == "DOUBLE"


# Test que le type physique prime sur le type inféré du lot
def test_add_column_to_metadata_prefers_physical_type(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that the stored type wins over the batch's inferred type.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    # 'value' est un DOUBLE dans la table ; le lot le présente en texte
    manager._add_column_to_metadata("value", pl.DataFrame({"value": ["a"]}))

    assert _types(built_ducklake_schema)[0]["value"] == "DOUBLE"


# ---------------------------------------------------------------------------
# Tests de updated_at en UTC
# ---------------------------------------------------------------------------


# Test que l'horodatage d'une écriture est en UTC, à la seconde près
def test_update_database_stamps_updated_at_in_utc(
    updater: DatabaseUpdater,
    built_ducklake_schema: duckdb.DuckDBPyConnection,
    update_df: pl.DataFrame,
) -> None:
    """Test that a write stamps ``updated_at`` with the UTC time, to the second.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
        update_df: DataFrame with two new rows.
    """
    built_ducklake_schema.execute(
        "UPDATE dataset_metadata SET updated_at = TIMESTAMP '2000-01-01'"
    )
    before = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    assert updater.update_database(update_df) is True
    after = datetime.now(UTC).replace(tzinfo=None)

    row = built_ducklake_schema.execute(
        "SELECT updated_at FROM dataset_metadata"
    ).fetchone()
    assert row is not None
    assert before <= row[0] <= after


# Test que _touch_dataset_metadata passe bien par utc_now
def test_touch_dataset_metadata_uses_utc_now(
    manager: BaseSchemaManager,
    built_ducklake_schema: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the wiring: ``updated_at`` is whatever ``utc_now`` returns.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection with the built schema.
        monkeypatch: Pytest monkeypatch fixture.
    """
    fixed = datetime(2031, 5, 6, 7, 8, 9)
    monkeypatch.setattr("dt_ducklake_manager.operations._base.utc_now", lambda: fixed)

    manager._touch_dataset_metadata()

    assert built_ducklake_schema.execute(
        "SELECT updated_at FROM dataset_metadata"
    ).fetchone() == (fixed,)


# ---------------------------------------------------------------------------
# Tests du NOT NULL : contrôle du libellé à l'écriture
# ---------------------------------------------------------------------------


# Test que le libellé ne peut pas être effacé
def test_update_column_metadata_refuses_null_label(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that ``label=None`` is refused and the recorded label is kept.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    before = built_ducklake_schema.execute(
        "SELECT label FROM metadata WHERE name = 'value'"
    ).fetchone()

    with pytest.raises(ValueError, match="label"):
        manager.update_column_metadata("value", label=None)

    assert (
        built_ducklake_schema.execute(
            "SELECT label FROM metadata WHERE name = 'value'"
        ).fetchone()
        == before
    )


# Test que les autres champs d'UI restent effaçables
def test_update_column_metadata_still_clears_optional_fields(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that a nullable UI field can still be cleared with ``None``.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    manager.update_column_metadata("value", unit="€")
    manager.update_column_metadata("value", unit=None)

    assert built_ducklake_schema.execute(
        "SELECT unit FROM metadata WHERE name = 'value'"
    ).fetchone() == (None,)
