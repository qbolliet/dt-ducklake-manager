# Importation des modules
# Modules de base
import os
import warnings
from typing import Any

import duckdb
import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.maintenance import (
    DatabaseRecoveryManager,
    RecoveryOperation,
    RecoveryStrategy,
)
from dt_ducklake_manager.maintenance.recovery import RecoveryResult
from dt_ducklake_manager.schema import DuckLakeTablesBuilder


def _ducklake_available() -> bool:
    """Vérifie si l'extension DuckLake est disponible dans l'environnement de test."""
    try:
        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        conn.close()
        return True
    except Exception:
        return False


# ===========================================================================
# Tests des dataclasses
# ===========================================================================


# Test de l'initialisation de RecoveryOperation
def test_recovery_operation_initialization() -> None:
    """Test that RecoveryOperation is correctly initialized with default values.

    Examples:
        >>> op = RecoveryOperation(strategy=RecoveryStrategy.VALIDATE_AND_FIX)
        >>> op.auto_validate
        True
    """
    op = RecoveryOperation(strategy=RecoveryStrategy.VALIDATE_AND_FIX)
    assert op.strategy == RecoveryStrategy.VALIDATE_AND_FIX
    assert op.auto_validate is True
    assert op.target_recovery_point is None
    assert isinstance(op.parameters, dict)


# Test de l'initialisation de RecoveryOperation avec des paramètres personnalisés
def test_recovery_operation_custom_parameters() -> None:
    """Test RecoveryOperation with custom parameters.

    ``target_recovery_point`` carries a DuckLake ``snapshot_id``, the only kind of
    restore point left since application backups were dropped (§1.5).

    Examples:
        >>> op = RecoveryOperation(
        ...     strategy=RecoveryStrategy.USE_SNAPSHOT_HISTORY,
        ...     target_recovery_point='17',
        ...     auto_validate=False,
        ... )
        >>> op.target_recovery_point
        '17'
    """
    op = RecoveryOperation(
        strategy=RecoveryStrategy.USE_SNAPSHOT_HISTORY,
        target_recovery_point="17",
        parameters={"snapshot_version": 5},
        auto_validate=False,
        description="Test recovery",
    )
    assert op.target_recovery_point == "17"
    assert op.parameters == {"snapshot_version": 5}
    assert op.auto_validate is False


# Test de l'initialisation de RecoveryResult
def test_recovery_result_initialization() -> None:
    """Test that RecoveryResult is correctly initialized.

    Examples:
        >>> from dt_ducklake_manager.maintenance.recovery import RecoveryResult
        >>> result = RecoveryResult(success=True,
        strategy_used=RecoveryStrategy.VALIDATE_AND_FIX)
        >>> result.success
        True
    """
    result = RecoveryResult(
        success=True,
        strategy_used=RecoveryStrategy.VALIDATE_AND_FIX,
        recovery_time=1.5,
        operations_performed=["validate", "fix"],
    )
    assert result.success is True
    assert result.strategy_used == RecoveryStrategy.VALIDATE_AND_FIX


# ===========================================================================
# Tests de DatabaseRecoveryManager
# ===========================================================================


# Initialisation d'un gestionnaire de récupération pour les tests
@pytest.fixture
def recovery_manager(built_ducklake_schema: Any) -> DatabaseRecoveryManager:
    """Create a DatabaseRecoveryManager for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.

    Returns:
        DatabaseRecoveryManager: initialized with the test connection.
    """
    return DatabaseRecoveryManager(
        connection=built_ducklake_schema,
    )


# Initialisation d'un gestionnaire branché sur un catalogue DuckLake réel
@pytest.fixture
def ducklake_recovery_manager(tmp_path: Any) -> DatabaseRecoveryManager:
    """Create a DatabaseRecoveryManager on a real, on-disk DuckLake catalog.

    The in-memory connection used elsewhere attaches no catalog, so the
    ``ducklake_snapshots()`` table function is unavailable there. An extra write
    is performed so the history holds several snapshots.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        DatabaseRecoveryManager: bound to a catalog with a non-trivial history.
    """
    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(catalog, data_dir).connect()

    df = pl.DataFrame({"id": [1, 2, 3], "category": ["A", "B", "A"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=["id"], connection=conn
        ).build_schema()

    # Écriture supplémentaire : l'historique compte alors plusieurs snapshots
    conn.execute("INSERT INTO fact_table (id, category) VALUES (4, 'C')")

    return DatabaseRecoveryManager(connection=conn)


# Test de l'initialisation du gestionnaire de récupération
def test_recovery_manager_initialization(recovery_manager: Any) -> None:
    """Test that DatabaseRecoveryManager initializes correctly.

    Args:
        recovery_manager: DatabaseRecoveryManager fixture.
    """
    assert recovery_manager is not None
    assert recovery_manager.schema == "main"
    assert recovery_manager.catalog_alias == "db"
    assert recovery_manager.auditor is not None


# Test que le gestionnaire n'expose plus de sauvegarde applicative
def test_recovery_manager_has_no_application_backup(recovery_manager: Any) -> None:
    """Test that no application-level backup machinery survives (§1.5).

    Recovery relies on DuckLake time travel only: no recovery point is created,
    listed or deleted, and no backup directory is held.

    Args:
        recovery_manager: DatabaseRecoveryManager fixture.
    """
    for attribute in (
        "create_recovery_point",
        "list_recovery_points",
        "delete_recovery_point",
        "backup_dir",
    ):
        assert not hasattr(recovery_manager, attribute)


# ===========================================================================
# Tests du time travel DuckLake : LE mécanisme de récupération (§1.5)
# ===========================================================================


# Test que list_ducklake_snapshots retourne l'historique d'un catalogue réel
@pytest.mark.skipif(
    not _ducklake_available(),
    reason="Extension ducklake non disponible dans cet environnement",
)
def test_list_ducklake_snapshots_returns_history(
    ducklake_recovery_manager: Any,
) -> None:
    """Test that list_ducklake_snapshots returns the catalog's snapshot history.

    Args:
        ducklake_recovery_manager: manager bound to a real DuckLake catalog.
    """
    snapshots = ducklake_recovery_manager.list_ducklake_snapshots()

    # Historique non vide, trié par identifiant décroissant
    assert snapshots is not None
    assert len(snapshots) >= 2
    assert "snapshot_id" in snapshots.columns
    ids = snapshots["snapshot_id"].to_list()
    assert ids == sorted(ids, reverse=True)


# Test que list_ducklake_snapshots retourne None hors catalogue DuckLake
def test_list_ducklake_snapshots_without_catalog(recovery_manager: Any) -> None:
    """Test that list_ducklake_snapshots returns None on a plain connection.

    Args:
        recovery_manager: DatabaseRecoveryManager fixture (in-memory connection,
            no attached DuckLake catalog).
    """
    assert recovery_manager.list_ducklake_snapshots() is None


# Test que USE_SNAPSHOT_HISTORY inventorie les snapshots et guide la restauration
@pytest.mark.skipif(
    not _ducklake_available(),
    reason="Extension ducklake non disponible dans cet environnement",
)
def test_use_snapshot_history_returns_restore_instructions(
    ducklake_recovery_manager: Any,
) -> None:
    """Test that USE_SNAPSHOT_HISTORY inventories snapshots and guides restoration.

    Args:
        ducklake_recovery_manager: manager bound to a real DuckLake catalog.
    """
    operation = RecoveryOperation(
        strategy=RecoveryStrategy.USE_SNAPSHOT_HISTORY,
        auto_validate=False,
        description="Inventaire des snapshots",
    )
    result = ducklake_recovery_manager.recover_database(operation)

    # Inventaire retourné, accompagné de la procédure de restauration
    assert result.success is True
    assert any("snapshot" in line for line in result.operations_performed)
    assert any("snapshot_version=" in line for line in result.recommendations)


# Test qu'un snapshot cible inexistant est signalé sans faire échouer l'inventaire
@pytest.mark.skipif(
    not _ducklake_available(),
    reason="Extension ducklake non disponible dans cet environnement",
)
def test_use_snapshot_history_unknown_target_is_flagged(
    ducklake_recovery_manager: Any,
) -> None:
    """Test that an unknown target snapshot is reported rather than silently used.

    Args:
        ducklake_recovery_manager: manager bound to a real DuckLake catalog.
    """
    operation = RecoveryOperation(
        strategy=RecoveryStrategy.USE_SNAPSHOT_HISTORY,
        target_recovery_point="999999",
        auto_validate=False,
    )
    result = ducklake_recovery_manager.recover_database(operation)

    assert result.success is True
    assert any("introuvable" in line for line in result.operations_performed)
