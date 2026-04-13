# Importation des modules
# Modules de base
# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.maintenance import (
    DatabaseRecoveryManager,
    RecoveryOperation,
    RecoveryStrategy,
)
from dt_ducklake_manager.maintenance.recovery import BackupType, RecoveryResult

# ===========================================================================
# Tests des dataclasses
# ===========================================================================


# Test de l'initialisation de RecoveryOperation
def test_recovery_operation_initialization():
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
def test_recovery_operation_custom_parameters():
    """Test RecoveryOperation with custom parameters.

    Examples:
        >>> op = RecoveryOperation(
        ...     strategy=RecoveryStrategy.USE_SNAPSHOT_HISTORY,
        ...     target_recovery_point='rp_001',
        ...     auto_validate=False,
        ... )
        >>> op.target_recovery_point
        'rp_001'
    """
    op = RecoveryOperation(
        strategy=RecoveryStrategy.USE_SNAPSHOT_HISTORY,
        target_recovery_point="rp_001",
        parameters={"snapshot_version": 5},
        auto_validate=False,
        description="Test recovery",
    )
    assert op.target_recovery_point == "rp_001"
    assert op.parameters == {"snapshot_version": 5}
    assert op.auto_validate is False


# Test de l'initialisation de RecoveryResult
def test_recovery_result_initialization():
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
def recovery_manager(built_ducklake_schema, tmp_path):
    """Create a DatabaseRecoveryManager for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
        tmp_path: pytest temporary directory for backup storage.

    Returns:
        DatabaseRecoveryManager: initialized with the test connection.
    """
    return DatabaseRecoveryManager(
        connection=built_ducklake_schema,
        backup_dir=str(tmp_path / "backups"),
        categorical_threshold=4,
        max_backup_age_days=30,
        auto_backup_on_changes=False,
    )


# Test de l'initialisation du gestionnaire de récupération
def test_recovery_manager_initialization(recovery_manager):
    """Test that DatabaseRecoveryManager initializes correctly.

    Args:
        recovery_manager: DatabaseRecoveryManager fixture.
    """
    assert recovery_manager is not None
    assert recovery_manager.max_backup_age_days == 30


# Test de la création d'un point de récupération
def test_create_recovery_point(recovery_manager):
    """Test that create_recovery_point returns a non-None recovery ID.

    Args:
        recovery_manager: DatabaseRecoveryManager fixture.
    """
    recovery_id = recovery_manager.create_recovery_point(
        backup_type=BackupType.METADATA_BACKUP,
        description="Test recovery point",
    )
    # Vérification que l'identifiant est bien retourné
    assert recovery_id is not None
    assert isinstance(recovery_id, str)
    assert len(recovery_id) > 0


# Test que la liste des points de récupération est non vide après création
def test_list_recovery_points_after_creation(recovery_manager):
    """Test that list_recovery_points returns a non-empty list after creation.

    Args:
        recovery_manager: DatabaseRecoveryManager fixture.
    """
    # Création d'un point de récupération
    recovery_manager.create_recovery_point(
        backup_type=BackupType.METADATA_BACKUP,
        description="Test point",
    )

    # Vérification que le point est bien listé
    points = recovery_manager.list_recovery_points()
    assert isinstance(points, list)
    assert len(points) >= 1


# Test de la suppression d'un point de récupération
def test_delete_recovery_point(recovery_manager):
    """Test that delete_recovery_point returns True and removes the point.

    Args:
        recovery_manager: DatabaseRecoveryManager fixture.
    """
    # Création d'un point de récupération à supprimer
    recovery_id = recovery_manager.create_recovery_point(
        backup_type=BackupType.METADATA_BACKUP,
        description="Point à supprimer",
    )
    assert recovery_id is not None

    # Suppression du point
    success = recovery_manager.delete_recovery_point(recovery_id)
    assert success is True

    # Vérification que le point n'est plus listé
    points = recovery_manager.list_recovery_points()
    point_ids = [p.recovery_id for p in points]
    assert recovery_id not in point_ids


# Test de la création d'un point de récupération de type SCHEMA_BACKUP
def test_create_schema_backup(recovery_manager):
    """Test that a METADATA_BACKUP recovery point can be created and listed by type.

    Args:
        recovery_manager: DatabaseRecoveryManager fixture.
    """
    recovery_id = recovery_manager.create_recovery_point(
        backup_type=BackupType.METADATA_BACKUP,
        description="Schema backup test",
    )
    assert recovery_id is not None

    # Vérification du type de backup
    points = recovery_manager.list_recovery_points(
        backup_type=BackupType.METADATA_BACKUP
    )
    assert len(points) >= 1
    assert points[0].backup_type == BackupType.METADATA_BACKUP
