# Importation des modules
# Modules de base
import logging
import warnings
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# DuckDB
import duckdb
import polars as pl

# Module de tests
import pytest

# Module à tester
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from tests.utils.ducklake import requires_ducklake

# ---------------------------------------------------------------------------
# Outils locaux
# ---------------------------------------------------------------------------


# Construction d'un builder sans le UserWarning lié à l'absence de clé primaire
def _builder(
    df: pl.DataFrame,
    conn: duckdb.DuckDBPyConnection | None = None,
    **kwargs: object,
) -> DuckLakeTablesBuilder:
    """Build a DuckLakeTablesBuilder, silencing the no-primary-key UserWarning.

    Args:
        df: Source DataFrame.
        conn: Optional connection (in-memory when None).
        **kwargs: Extra ``DuckLakeTablesBuilder`` arguments.

    Returns:
        DuckLakeTablesBuilder: the builder, not yet built.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return DuckLakeTablesBuilder(df, connection=conn, **kwargs)  # type: ignore[arg-type]


# Lecture de la paire (type enregistré, type physique) de chaque colonne
def _recorded_and_physical(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, tuple[str, str]]:
    """Return ``{column: (metadata.sql_type, DESCRIBE type)}`` of a built schema.

    Args:
        conn: Connection holding a built ``main`` schema.

    Returns:
        dict[str, tuple[str, str]]: recorded and physical type of each column.
    """
    recorded = dict(conn.execute("SELECT name, sql_type FROM metadata").fetchall())
    physical = {r[0]: r[1] for r in conn.execute("DESCRIBE fact_table").fetchall()}
    return {c: (recorded[c], physical[c]) for c in physical}


# Attachement d'un catalogue DuckLake réel dans un répertoire temporaire
def _attach_ducklake(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Attach a real DuckLake catalog stored in ``tmp_path`` and ``USE`` it.

    Args:
        tmp_path: Temporary directory.

    Returns:
        duckdb.DuckDBPyConnection: connection positioned on ``db.main``.
    """
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL ducklake; LOAD ducklake;")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    conn.execute(
        f"ATTACH 'ducklake:{(tmp_path / 'c.ducklake').as_posix()}' AS db"
        f" (DATA_PATH '{data_dir.as_posix()}')"
    )
    conn.execute("USE db.main")
    return conn


# ---------------------------------------------------------------------------
# Tests du type physique enregistré dans metadata.sql_type
# ---------------------------------------------------------------------------


# Test du type DECIMAL(10,2) sur le chemin CTAS (sans clé ni partition)
def test_build_records_physical_decimal_type_on_ctas_path() -> None:
    """Test that a Decimal(10,2) column is recorded as DECIMAL(10,2), not DECIMAL."""
    df = pl.DataFrame(
        {"price": [Decimal("1.25"), Decimal("2.50")], "n": [1, 2]},
        schema={"price": pl.Decimal(10, 2), "n": pl.Int64},
    )
    builder = _builder(df)
    builder.build_schema()

    pairs = _recorded_and_physical(builder.conn)
    assert pairs["price"] == ("DECIMAL(10,2)", "DECIMAL(10,2)")
    assert pairs["n"] == ("BIGINT", "BIGINT")


# Test que, sur le chemin DDL explicite, le type enregistré est le type stocké
def test_build_records_physical_types_on_explicit_ddl_path() -> None:
    """Test that every recorded sql_type equals the stored one with a primary key."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "price": [Decimal("1.25"), Decimal("2.50")],
            "small": [1, 2],
            "label": ["a", "b"],
        },
        schema={
            "id": pl.Int64,
            "price": pl.Decimal(10, 2),
            "small": pl.Int16,
            "label": pl.String,
        },
    )
    builder = _builder(df, primary_keys=["id"])
    builder.build_schema()

    pairs = _recorded_and_physical(builder.conn)
    assert all(recorded == physical for recorded, physical in pairs.values())
    # Le type paramétré est bien celui de la table, pas le nom nu DECIMAL
    assert pairs["price"][0].startswith("DECIMAL(")
    assert pairs["small"][0] == "SMALLINT"


# Test que l'alignement est sans effet quand types inférés et stockés coïncident
def test_sync_physical_sql_types_is_idempotent(sample_df: pl.DataFrame) -> None:
    """Test that a second synchronisation changes nothing.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = _builder(sample_df)
    builder.build_schema()
    query = "SELECT * FROM metadata ORDER BY name"
    before = builder.conn.execute(query).fetchall()

    builder._sync_physical_sql_types()

    assert builder.conn.execute(query).fetchall() == before


# Test de la correction d'un type enregistré qui diverge du type stocké
def test_sync_physical_sql_types_corrects_divergence(sample_df: pl.DataFrame) -> None:
    """Test that a recorded type differing from the stored one is realigned.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = _builder(sample_df)
    builder.build_schema()
    builder.conn.execute("UPDATE metadata SET sql_type = 'VARCHAR' WHERE name = 'id'")

    builder._sync_physical_sql_types()

    assert builder.conn.execute(
        "SELECT sql_type FROM metadata WHERE name = 'id'"
    ).fetchone() == ("BIGINT",)


# Test avec des noms accentués, sans clé primaire : construction et type physique
def test_build_with_accented_column_names_and_no_primary_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that non-ASCII column names build, are typed and are flagged.

    Args:
        caplog: Log capture fixture.
    """
    df = pl.DataFrame(
        {"Valeur Élevée": [Decimal("1.5")], "année": [2024]},
        schema={"Valeur Élevée": pl.Decimal(6, 1), "année": pl.Int32},
    )
    builder = _builder(df)
    with caplog.at_level(logging.WARNING):
        builder.build_schema()

    rows = dict(builder.conn.execute("SELECT name, sql_type FROM metadata").fetchall())
    assert rows == {"Valeur Élevée": "DECIMAL(6,1)", "année": "INTEGER"}
    flagged = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Valeur Élevée" in m for m in flagged)
    assert any("année" in m for m in flagged)


# Test qu'un nom conforme ne déclenche aucun avertissement de nommage
def test_build_snake_case_names_do_not_warn(
    sample_df: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that a snake_case dataset raises no naming warning.

    Args:
        sample_df: Sample polars DataFrame (snake_case columns).
        caplog: Log capture fixture.
    """
    builder = _builder(sample_df)
    with caplog.at_level(logging.WARNING):
        builder.build_schema()

    assert not [r for r in caplog.records if "does not match" in r.getMessage()]


# ---------------------------------------------------------------------------
# Tests du refus des types composites
# ---------------------------------------------------------------------------


# Test du refus, sans rien écrire, de chaque type composite
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame({"id": [1], "tags": [["a", "b"]]}),
        pl.DataFrame(
            {"id": [1], "tags": [[1, 2]]},
            schema={"id": pl.Int64, "tags": pl.Array(pl.Int64, 2)},
        ),
        pl.DataFrame({"id": [1], "point": [{"x": 1, "y": 2}]}),
    ],
    ids=["list", "array", "struct"],
)
def test_build_refuses_composite_columns(df: pl.DataFrame) -> None:
    """Test that a composite column is refused and that nothing is written.

    Args:
        df: DataFrame carrying one composite column.
    """
    conn = duckdb.connect(":memory:")
    builder = _builder(df, conn, primary_keys=["id"])

    with pytest.raises(ValueError, match="composite type"):
        builder.build_schema()

    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    assert tables == []


# Test du refus sur un catalogue DuckLake réel : le schéma cible reste vide
@requires_ducklake
def test_build_refuses_composite_on_real_ducklake(tmp_path: Path) -> None:
    """Test the composite refusal against a real DuckLake catalog.

    Args:
        tmp_path: Temporary directory.
    """
    conn = _attach_ducklake(tmp_path)
    builder = _builder(pl.DataFrame({"l": [[1], [2]]}), conn)

    with pytest.raises(ValueError, match="List"):
        builder.build_schema()

    assert conn.execute(
        "SELECT count(*) FROM information_schema.tables"
        " WHERE table_catalog = 'db' AND table_schema = 'main'"
    ).fetchone() == (0,)


# ---------------------------------------------------------------------------
# Tests de updated_at en UTC
# ---------------------------------------------------------------------------


# Test que updated_at est en UTC, à la seconde près
def test_build_stamps_updated_at_in_utc(sample_df: pl.DataFrame) -> None:
    """Test that ``updated_at`` is the UTC time of the build, to the second.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = _builder(sample_df)
    before = datetime.now(UTC).replace(tzinfo=None)
    builder.build_schema()
    after = datetime.now(UTC).replace(tzinfo=None)

    row = builder.conn.execute("SELECT updated_at FROM dataset_metadata").fetchone()
    assert row is not None
    assert before.replace(microsecond=0) <= row[0] <= after


# Test que le builder passe bien par utc_now (indépendant du fuseau de la machine)
def test_build_uses_utc_now(
    sample_df: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the wiring: the written ``updated_at`` is whatever ``utc_now`` returns.

    Args:
        sample_df: Sample polars DataFrame.
        monkeypatch: Pytest monkeypatch fixture.
    """
    fixed = datetime(2030, 1, 2, 3, 4, 5)
    monkeypatch.setattr("dt_ducklake_manager.schema.persistence.utc_now", lambda: fixed)
    builder = _builder(sample_df)
    builder.build_schema()

    row = builder.conn.execute("SELECT updated_at FROM dataset_metadata").fetchone()
    assert row == (fixed,)


# ---------------------------------------------------------------------------
# Tests du NOT NULL du contrat metadata
# ---------------------------------------------------------------------------


# Test que la base refuse un NULL dans les colonnes du contrat
@pytest.mark.parametrize(
    "insert",
    [
        "INSERT INTO metadata (name, label, sql_type) VALUES (NULL, 'l', 'VARCHAR')",
        "INSERT INTO metadata (name, label, sql_type) VALUES ('n', NULL, 'VARCHAR')",
        "INSERT INTO metadata (name, label, sql_type) VALUES ('n', 'l', NULL)",
        "INSERT INTO metadata (name, label, sql_type, is_categorical)"
        " VALUES ('n', 'l', 'VARCHAR', NULL)",
        "INSERT INTO metadata (name, label, sql_type, is_primary_key)"
        " VALUES ('n', 'l', 'VARCHAR', NULL)",
    ],
    ids=["name", "label", "sql_type", "is_categorical", "is_primary_key"],
)
def test_metadata_table_rejects_null_contract_values(
    sample_df: pl.DataFrame, insert: str
) -> None:
    """Test that NULL is rejected in each NOT NULL column of ``metadata``.

    Args:
        sample_df: Sample polars DataFrame.
        insert: INSERT statement carrying one NULL contract value.
    """
    builder = _builder(sample_df)
    builder.build_schema()

    with pytest.raises(duckdb.ConstraintException):
        builder.conn.execute(insert)


# Test que les booléens omis prennent FALSE (DEFAULT conservé avec NOT NULL)
def test_metadata_table_defaults_booleans_to_false(sample_df: pl.DataFrame) -> None:
    """Test that omitted booleans still default to FALSE.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = _builder(sample_df)
    builder.build_schema()
    builder.conn.execute(
        "INSERT INTO metadata (name, label, sql_type) VALUES ('n', 'l', 'VARCHAR')"
    )

    assert builder.conn.execute(
        "SELECT is_categorical, is_primary_key FROM metadata WHERE name = 'n'"
    ).fetchone() == (False, False)


# Test du NOT NULL et du type physique sur un vrai catalogue DuckLake
@requires_ducklake
def test_metadata_contract_on_real_ducklake(tmp_path: Path) -> None:
    """Test NOT NULL and the physical type against a real DuckLake catalog.

    Args:
        tmp_path: Temporary directory.
    """
    conn = _attach_ducklake(tmp_path)
    df = pl.DataFrame(
        {"id": [1, 2], "price": [Decimal("1.5")] * 2},
        schema={"id": pl.Int64, "price": pl.Decimal(10, 2)},
    )
    _builder(df, conn, primary_keys=["id"]).build_schema()

    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "INSERT INTO metadata (name, label, sql_type) VALUES (NULL, 'l', 'x')"
        )
    types = dict(conn.execute("SELECT name, sql_type FROM metadata").fetchall())
    physical = {r[0]: r[1] for r in conn.execute("DESCRIBE fact_table").fetchall()}
    assert types == physical
    assert types["price"].startswith("DECIMAL(")
