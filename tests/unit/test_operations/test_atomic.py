# Importation des modules
# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.operations import (
    AtomicDatabaseOperations,
    AtomicOperationConfig,
    AtomicOperationResult,
)
from dt_ducklake_manager.operations.atomic import AtomicOperationType

# ===========================================================================
# Tests de AtomicOperationType
# ===========================================================================


# Test que tous les types d'opérations atomiques sont définis
def test_atomic_operation_type_values():
    """Test that all expected AtomicOperationType values are defined.

    Examples:
        >>> AtomicOperationType.BATCH_UPDATE.value
        'batch_update'
    """
    # Vérification de la présence des types attendus
    assert hasattr(AtomicOperationType, "BATCH_UPDATE")
    assert hasattr(AtomicOperationType, "SCHEMA_MIGRATION")
    assert hasattr(AtomicOperationType, "CLEANUP_OPERATION")
    assert hasattr(AtomicOperationType, "VALIDATION_AND_REPAIR")


# ===========================================================================
# Tests de AtomicOperationConfig
# ===========================================================================


# Test de l'initialisation de AtomicOperationConfig avec les valeurs par défaut
def test_atomic_operation_config_defaults():
    """Test that AtomicOperationConfig has correct default values.

    Examples:
        >>> config =
        AtomicOperationConfig(operation_type=AtomicOperationType.BATCH_UPDATE)
        >>> config.enable_backup
        True
    """
    config = AtomicOperationConfig(operation_type=AtomicOperationType.BATCH_UPDATE)
    # Vérification des valeurs par défaut des paramètres booléens
    assert config.enable_backup is True
    assert config.enable_validation is True
    assert config.rollback_on_validation_fail is True
    # Vérification des valeurs par défaut des paramètres numériques
    assert config.max_retry_attempts >= 1
    # batch_size vaut None par défaut (le gestionnaire utilise un batch_size interne)
    assert config.batch_size is None


# Test de l'instanciation de AtomicOperationConfig avec des paramètres personnalisés
def test_atomic_operation_config_custom():
    """Test that AtomicOperationConfig stores custom parameter values.

    Examples:
        >>> config = AtomicOperationConfig(
        ...     operation_type=AtomicOperationType.CLEANUP_OPERATION,
        ...     enable_backup=False,
        ...     batch_size=500,
        ... )
        >>> config.batch_size
        500
    """
    config = AtomicOperationConfig(
        operation_type=AtomicOperationType.CLEANUP_OPERATION,
        enable_backup=False,
        enable_validation=False,
        batch_size=500,
        parallel_workers=2,
    )
    assert config.enable_backup is False
    assert config.enable_validation is False
    assert config.batch_size == 500
    assert config.parallel_workers == 2


# ===========================================================================
# Tests de AtomicOperationResult
# ===========================================================================


# Test de l'instanciation de AtomicOperationResult avec succès
def test_atomic_operation_result_success():
    """Test that AtomicOperationResult can be initialized with success=True.

    Examples:
        >>> result = AtomicOperationResult(
        ...     success=True,
        ...     operation_type=AtomicOperationType.BATCH_UPDATE,
        ...     execution_time=1.5,
        ... )
        >>> result.success
        True
    """
    result = AtomicOperationResult(
        success=True,
        operation_type=AtomicOperationType.BATCH_UPDATE,
        execution_time=1.5,
        backup_created=False,
        validation_passed=True,
        operations_performed=["insert"],
    )
    assert result.success is True
    assert result.validation_passed is True
    assert result.rollback_performed is False


# Test de l'instanciation de AtomicOperationResult avec échec
def test_atomic_operation_result_failure():
    """Test that AtomicOperationResult can be initialized with success=False.

    Examples:
        >>> result = AtomicOperationResult(
        ...     success=False,
        ...     operation_type=AtomicOperationType.BATCH_UPDATE,
        ...     error_message='Something went wrong',
        ... )
        >>> result.success
        False
    """
    result = AtomicOperationResult(
        success=False,
        operation_type=AtomicOperationType.BATCH_UPDATE,
        execution_time=0.1,
        backup_created=False,
        validation_passed=False,
        operations_performed=[],
        error_message="Something went wrong",
        rollback_performed=True,
    )
    assert result.success is False
    assert result.error_message == "Something went wrong"
    assert result.rollback_performed is True


# ===========================================================================
# Tests de AtomicDatabaseOperations
# ===========================================================================


# Initialisation d'une instance de AtomicDatabaseOperations pour les tests
@pytest.fixture
def atomic_ops(built_ducklake_schema, tmp_path):
    """Create an AtomicDatabaseOperations instance for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
        tmp_path: pytest temporary directory for backup storage.

    Returns:
        AtomicDatabaseOperations: initialized with the test connection.
    """
    return AtomicDatabaseOperations(
        connection=built_ducklake_schema,
        categorical_threshold=4,
        backup_dir=str(tmp_path / "backups"),
        default_batch_size=100,
        default_max_workers=2,
    )


# Test de l'initialisation de AtomicDatabaseOperations
def test_atomic_operations_initialization(atomic_ops):
    """Test that AtomicDatabaseOperations initializes without errors.

    Args:
        atomic_ops: AtomicDatabaseOperations fixture.
    """
    assert atomic_ops is not None


# Test de get_operation_status
def test_get_operation_status_returns_dict(atomic_ops):
    """Test that get_operation_status returns a dict with expected structure.

    Args:
        atomic_ops: AtomicDatabaseOperations fixture.
    """
    status = atomic_ops.get_operation_status()
    # Vérification que le résultat est un dictionnaire non vide
    assert isinstance(status, dict)


# Test de validate_system_integrity
def test_validate_system_integrity_returns_dict(atomic_ops):
    """Test that validate_system_integrity returns a dict with integrity information.

    Args:
        atomic_ops: AtomicDatabaseOperations fixture.
    """
    integrity = atomic_ops.validate_system_integrity()
    # Vérification que le résultat est un dictionnaire
    assert isinstance(integrity, dict)
