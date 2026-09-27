# Importation des modules
# Modules de base
import logging
import os
import warnings
from datetime import datetime
from typing import Any

import duckdb
import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.operations import DatabaseUpdater
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from tests.utils.ducklake import requires_ducklake

# ---------------------------------------------------------------------------
# Fonctions et classes auxiliaires
# ---------------------------------------------------------------------------


# Connexion interposée faisant échouer une instruction précise
class _FailingConnection:
    """Proxy of a DuckDB connection raising an I/O error on one statement.

    Every attribute is delegated to the wrapped connection, except ``execute``,
    which raises ``duckdb.IOException`` when the statement contains ``pattern``:
    this simulates a disk failure in the middle of an operation.

    Args:
        conn: Connection to wrap.
        pattern: Fragment of SQL identifying the statement to fail.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, pattern: str) -> None:
        self._conn = conn
        self._pattern = pattern

    def execute(self, query: str, *args: Any, **kwargs: Any) -> Any:
        """Run ``query``, unless it matches the pattern to fail."""
        if self._pattern in query:
            raise duckdb.IOException("simulated I/O error")
        return self._conn.execute(query, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


# Lecture triée du contenu de la table des faits
def _facts(conn: Any) -> list[dict[str, Any]]:
    """Read the fact table sorted by id.

    Args:
        conn: DuckDB connection holding a built schema.

    Returns:
        list[dict]: One dict per row, sorted by ``id``.
    """
    rows: list[dict[str, Any]] = (
        conn.execute("SELECT * FROM fact_table ORDER BY id").pl().to_dicts()
    )
    return rows


# Lot mêlant une ligne existante modifiée, une ligne existante inchangée et une
# ligne nouvelle
@pytest.fixture
def mixed_df(sample_df: pl.DataFrame) -> pl.DataFrame:
    """Build a batch with one changed, one unchanged and one new row.

    Args:
        sample_df: DataFrame the schema was built from (ids 1 to 5).

    Returns:
        pl.DataFrame: ids 1 (value changed), 2 (identical) and 10 (new).
    """
    existing = sample_df.filter(pl.col("id").is_in([1, 2])).with_columns(
        pl.when(pl.col("id") == 1)
        .then(pl.lit(99.0))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    new_row = pl.DataFrame(
        {
            "id": [10],
            "category": ["A"],
            "value": [1.5],
            "date": [datetime(2024, 3, 1).date()],
            "status": ["active"],
            "high_cardinality": ["val_300"],
        }
    )
    return pl.concat([existing, new_row.cast(existing.schema)])


# ---------------------------------------------------------------------------
# Écriture en une passe : mises à jour, insertions et lignes inchangées
# ---------------------------------------------------------------------------


# Test que seules les lignes réellement modifiées sont comptées comme mises à jour
def test_update_counts_only_changed_rows(
    updater: DatabaseUpdater, mixed_df: pl.DataFrame
) -> None:
    """Test that an unchanged existing row is neither rewritten nor counted.

    Args:
        updater: DatabaseUpdater fixture.
        mixed_df: Batch with one changed, one unchanged and one new row.
    """
    assert updater.update_database(mixed_df) is True
    report = updater.last_report
    assert report is not None
    assert report.rows_inserted == 1
    assert report.rows_updated == 1

    # Valeur modifiée, ligne inchangée conservée, ligne nouvelle présente
    facts = {row["id"]: row for row in _facts(updater.conn)}
    assert facts[1]["value"] == 99.0
    assert facts[2]["value"] == 0.2
    assert 10 in facts


# Test qu'un lot entièrement identique à la base ne produit aucune écriture
def test_update_identical_batch_writes_nothing(
    updater: DatabaseUpdater, sample_df: pl.DataFrame
) -> None:
    """Test that re-sending the stored rows updates and inserts nothing.

    Args:
        updater: DatabaseUpdater fixture.
        sample_df: DataFrame the schema was built from.
    """
    before = _facts(updater.conn)
    assert updater.update_database(sample_df) is True
    report = updater.last_report
    assert report is not None
    assert (report.rows_inserted, report.rows_updated) == (0, 0)
    assert _facts(updater.conn) == before


# Test que les lignes nouvelles sont écrites triées selon cluster_by
def test_update_sorts_new_rows_by_cluster_by(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that a batch sent out of order lands sorted by cluster_by.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
    """
    updater.update_cluster_by(["value"])
    out_of_order = pl.DataFrame(
        {
            "id": [90, 91, 92],
            "category": ["A", "B", "A"],
            "value": [9.0, 5.0, 7.0],
            "date": pl.date_range(
                datetime(2024, 7, 1), datetime(2024, 7, 3), "1d", eager=True
            ),
            "status": ["active", "active", "active"],
            "high_cardinality": ["val_900", "val_901", "val_902"],
        }
    )
    assert updater.update_database(out_of_order) is True

    # Ordre physique d'insertion, lu sans tri explicite : croissant sur value
    rows = built_ducklake_schema.execute(
        "SELECT value FROM fact_table WHERE id IN (90, 91, 92)"
    ).fetchall()
    values = [r[0] for r in rows]
    assert values == sorted(values)


# Test qu'un lot ne portant qu'une partie des colonnes laisse les autres intactes
def test_update_partial_columns_keeps_other_values(updater: DatabaseUpdater) -> None:
    """Test that columns absent from the batch keep their stored values.

    Args:
        updater: DatabaseUpdater fixture.
    """
    assert updater.update_database(pl.DataFrame({"id": [1], "value": [42.0]}))
    row = updater.conn.execute(
        "SELECT value, status FROM fact_table WHERE id = 1"
    ).fetchone()
    assert row == (42.0, "active")


# ---------------------------------------------------------------------------
# Atomicité : un échec au milieu de l'écriture annule tout
# ---------------------------------------------------------------------------


# Test qu'un échec de l'insertion annule aussi la mise à jour déjà exécutée
def test_update_failure_after_update_statement_rolls_back_everything(
    updater: DatabaseUpdater, mixed_df: pl.DataFrame
) -> None:
    """Test that an I/O error on the INSERT also undoes the preceding UPDATE.

    The batch updates id=1 then inserts id=10; the INSERT fails: update_database
    returns False and the fact table is exactly as before (neither the new value
    of id=1 nor the row id=10).

    Args:
        updater: DatabaseUpdater fixture.
        mixed_df: Batch with one changed, one unchanged and one new row.
    """
    before = _facts(updater.conn)
    updater.conn = _FailingConnection(  # type: ignore[assignment]
        updater.conn, 'INSERT INTO "main"."fact_table"'
    )

    assert updater.update_database(mixed_df) is False
    assert _facts(updater.conn) == before
    report = updater.last_report
    assert report is not None
    assert any("simulated I/O error" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Unicité de la clé primaire
# ---------------------------------------------------------------------------


# Test que la déduplication du lot porte sur la clé primaire, selon keep
@pytest.mark.parametrize(
    ("keep", "expected"),
    [("first", [7.0]), ("last", [8.0]), ("none", [])],
)
def test_update_deduplicates_batch_on_primary_keys(
    updater: DatabaseUpdater, keep: Any, expected: list[float]
) -> None:
    """Test that two rows sharing a key but not their values are deduplicated.

    Args:
        updater: DatabaseUpdater fixture.
        keep: Deduplication strategy.
        expected: Values left for id=50 after the update.
    """
    batch = pl.DataFrame({"id": [50, 50], "value": [7.0, 8.0]})
    assert updater.update_database(batch, keep=keep) is True

    rows = updater.conn.execute("SELECT value FROM fact_table WHERE id = 50").fetchall()
    assert [r[0] for r in rows] == expected


# Test qu'un lot non unique sur la clé est refusé sans déduplication
def test_update_duplicate_keys_without_dedup_raises(
    updater: DatabaseUpdater,
) -> None:
    """Test that duplicated keys raise ValueError when deduplication is off.

    Args:
        updater: DatabaseUpdater fixture.
    """
    batch = pl.DataFrame({"id": [50, 50], "value": [7.0, 8.0]})
    with pytest.raises(ValueError, match="not unique"):
        updater.update_database(batch, check_duplicates_update=False)
    count = updater.conn.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 50"
    ).fetchone()[0]
    assert count == 0


# Test qu'un doublon de clé préexistant fait échouer l'update sans rien supprimer
def test_update_refuses_preexisting_duplicate_key(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that a duplicated key already in the base fails the update.

    The check raises rather than deleting rows: the duplicates stay, and so does
    the rest of the base.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    updater.conn.execute("INSERT INTO fact_table SELECT * FROM fact_table WHERE id = 1")
    before = _facts(updater.conn)

    assert updater.update_database(update_df) is False
    assert _facts(updater.conn) == before
    report = updater.last_report
    assert report is not None
    assert any("not unique" in w for w in report.warnings)


# Test que le contrôle d'unicité en base peut être désactivé
def test_update_duplicate_check_can_be_disabled(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that check_duplicates_db=False lets the update through.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    updater.conn.execute("INSERT INTO fact_table SELECT * FROM fact_table WHERE id = 1")
    assert updater.update_database(update_df, check_duplicates_db=False) is True


# ---------------------------------------------------------------------------
# Erreurs de saisie : ValueError, rien d'écrit
# ---------------------------------------------------------------------------


# Test qu'une clé primaire manquante dans le lot lève une ValueError
def test_update_missing_primary_key_raises(updater: DatabaseUpdater) -> None:
    """Test that a batch without its primary key column raises ValueError.

    Args:
        updater: DatabaseUpdater fixture.
    """
    with pytest.raises(ValueError, match="missing primary key"):
        updater.update_database(pl.DataFrame({"value": [1.0]}))


# Test qu'une clé primaire nulle dans le lot lève une ValueError
def test_update_null_primary_key_raises(updater: DatabaseUpdater) -> None:
    """Test that a null key raises instead of being inserted at every update.

    Args:
        updater: DatabaseUpdater fixture.
    """
    batch = pl.DataFrame({"id": [None, 7], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="null value"):
        updater.update_database(batch)
    count = updater.conn.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id IS NULL OR id = 7"
    ).fetchone()[0]
    assert count == 0


# Test qu'une table sans clé primaire déclarée lève une ValueError
def test_update_without_declared_primary_key_raises() -> None:
    """Test that update_database refuses a fact table with no primary key."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            pl.DataFrame({"id": [1], "value": [1.0]}), categorical_threshold=4
        )
    builder.build_schema()
    updater = DatabaseUpdater(connection=builder.conn)
    with pytest.raises(ValueError, match="No primary key"):
        updater.update_database(pl.DataFrame({"id": [2], "value": [2.0]}))


# Test qu'un lot vide est un succès sans écriture
def test_update_empty_batch_is_noop(
    updater: DatabaseUpdater, sample_df: pl.DataFrame
) -> None:
    """Test that an empty batch returns True, writes nothing and says so.

    Args:
        updater: DatabaseUpdater fixture.
        sample_df: DataFrame the schema was built from.
    """
    before = _facts(updater.conn)
    assert updater.update_database(sample_df.head(0)) is True
    assert _facts(updater.conn) == before
    report = updater.last_report
    assert report is not None
    assert any("empty" in w for w in report.warnings)


# Test que les paramètres de découpage en lots sont obsolètes
def test_update_use_batch_processing_is_deprecated(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that use_batch_processing is ignored with a DeprecationWarning.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    with pytest.warns(DeprecationWarning, match="use_batch_processing"):
        assert updater.update_database(update_df, use_batch_processing=True)


# ---------------------------------------------------------------------------
# Colonnes nouvelles : dans la transaction de l'update
# ---------------------------------------------------------------------------


# Test qu'un update refusé ne laisse pas la colonne nouvelle derrière lui
def test_refused_update_leaves_no_new_column(updater: DatabaseUpdater) -> None:
    """Test that a new column is rolled back with the update that added it.

    The batch adds a label column whose values break the code -> label
    dependency on 'category': the update raises and the column is gone.

    Args:
        updater: DatabaseUpdater fixture.
    """
    batch = pl.DataFrame({"id": [1, 3], "category_libelle": ["Cat A", "Autre A"]})
    with pytest.raises(ValueError):
        updater.update_database(
            batch,
            allow_new_columns=True,
            column_metadata={"category_libelle": {"label_for": "category"}},
        )
    assert "category_libelle" not in updater._get_fact_table_columns()
    row = updater.conn.execute(
        "SELECT COUNT(*) FROM metadata WHERE name = 'category_libelle'"
    ).fetchone()
    assert row[0] == 0


# Test qu'un parent_name peut désigner une autre colonne nouvelle du même lot
def test_update_new_columns_hierarchy_between_new_columns(
    updater: DatabaseUpdater,
) -> None:
    """Test that a new column may declare another new column as its parent.

    Args:
        updater: DatabaseUpdater fixture.
    """
    batch = pl.DataFrame(
        {"id": [1, 2], "departement": ["75", "13"], "commune": ["Paris", "Aix"]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert updater.update_database(
            batch,
            allow_new_columns=True,
            column_metadata={"commune": {"parent_name": "departement"}},
        )
    row = updater.conn.execute(
        "SELECT parent_name FROM metadata WHERE name = 'commune'"
    ).fetchone()
    assert row[0] == "departement"


# ---------------------------------------------------------------------------
# Journalisation : une seule ligne INFO de synthèse
# ---------------------------------------------------------------------------


# Test qu'un update réussi ne journalise qu'une ligne INFO
def test_update_logs_a_single_info_line(
    updater: DatabaseUpdater, update_df: pl.DataFrame, caplog: Any
) -> None:
    """Test that a successful update logs its summary as its only INFO line.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
        caplog: pytest log capture.
    """
    with caplog.at_level(logging.INFO):
        caplog.clear()
        assert updater.update_database(update_df) is True
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_records) == 1
    assert info_records[0].getMessage().startswith("update main")


# ---------------------------------------------------------------------------
# Catalogue DuckLake réel : snapshots et comptages mesurés
# ---------------------------------------------------------------------------


# Catalogue DuckLake réel construit à partir d'un petit jeu de données
@pytest.fixture
def real_catalog(tmp_path: Any) -> Any:
    """Provide a real DuckLake catalog holding a 5-row schema keyed by id.

    Args:
        tmp_path: pytest temporary directory.

    Yields:
        duckdb.DuckDBPyConnection: connection attached to the catalog.
    """
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(
        str(tmp_path / "test.ducklake"), data_dir, data_inlining_row_limit=0
    ).connect()
    df = pl.DataFrame(
        {
            "id": list(range(1, 6)),
            "category": ["A", "B", "A", "C", "B"],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=["id"], connection=conn
        ).build_schema()
    yield conn
    conn.close()


# Test qu'un update ne produit qu'un snapshot d'écriture, porteur du run_id
@requires_ducklake
def test_update_single_snapshot_carries_run_id(real_catalog: Any) -> None:
    """Test that the data write and the updated_at stamp share one snapshot.

    Without compaction, the update produces exactly one snapshot, authored by
    the run, which modifies both the fact table and dataset_metadata.

    Args:
        real_catalog: Connection attached to a real DuckLake catalog.
    """
    before = real_catalog.execute(
        "SELECT max(snapshot_id) FROM ducklake_snapshots('db')"
    ).fetchone()[0]
    updater = DatabaseUpdater(connection=real_catalog, categorical_threshold=4)
    batch = pl.DataFrame({"id": [1, 10], "category": ["A", "C"], "value": [9.0, 1.0]})

    assert updater.update_database(batch, run_id="run-42", compact_after_update=False)

    snapshots = real_catalog.execute(
        "SELECT snapshot_id, author FROM ducklake_snapshots('db')"
        " WHERE snapshot_id > ?",
        [before],
    ).fetchall()
    assert len(snapshots) == 1
    snapshot_id, author = snapshots[0]
    assert author == "run-42"

    # L'horodatage a changé dans ce même snapshot, et dans aucun autre
    stamps = real_catalog.execute(
        f"SELECT (SELECT updated_at FROM dataset_metadata AT (VERSION => {before})),"
        f" (SELECT updated_at FROM dataset_metadata AT (VERSION => {snapshot_id}))"
    ).fetchone()
    assert stamps[0] != stamps[1]

    report = updater.last_report
    assert report is not None
    assert (report.rows_inserted, report.rows_updated) == (1, 1)


# Test que la compaction post-écriture fusionne réellement les petits fichiers
@requires_ducklake
def test_update_compaction_merges_small_files(real_catalog: Any) -> None:
    """Test that the post-write merge reduces the number of data files.

    Three updates without compaction leave four files; a fourth update with
    compaction merges them.

    Args:
        real_catalog: Connection attached to a real DuckLake catalog.
    """
    updater = DatabaseUpdater(connection=real_catalog, categorical_threshold=4)
    for i in range(3):
        batch = pl.DataFrame({"id": [100 + i], "category": ["A"], "value": [float(i)]})
        assert updater.update_database(batch, compact_after_update=False)

    files_before = real_catalog.execute(
        "SELECT file_count FROM ducklake_table_info('db')"
        " WHERE table_name = 'fact_table'"
    ).fetchone()[0]
    assert files_before >= 4

    batch = pl.DataFrame({"id": [200], "category": ["B"], "value": [2.0]})
    assert updater.update_database(batch)
    report = updater.last_report
    assert report is not None
    assert report.maintenance["merge_files_processed"] > 0
    assert report.files_after < files_before + 1
