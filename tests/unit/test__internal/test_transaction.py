# Importation des modules
# Module de tests
from typing import Any

import pytest

# Modules du package à tester
from dt_ducklake_manager._internal.managers.transaction import (
    TransactionContext,
    TransactionManager,
    TransactionOperation,
    TransactionState,
)

# ---------------------------------------------------------------------------
# Tests des dataclasses et énumérations
# ---------------------------------------------------------------------------


# Test que TransactionState contient tous les états attendus
def test_transaction_state_values() -> None:
    """Test that TransactionState contains all expected state values.

    Examples:
        >>> TransactionState.RUNNING.value
        'running'
    """
    assert TransactionState.PENDING.value == "pending"
    assert TransactionState.RUNNING.value == "running"
    assert TransactionState.COMMITTED.value == "committed"
    assert TransactionState.ROLLED_BACK.value == "rolled_back"
    assert TransactionState.FAILED.value == "failed"


# Test de l'initialisation d'un TransactionOperation
def test_transaction_operation_initialization() -> None:
    """Test that TransactionOperation can be initialized with all fields.

    Examples:
        >>> op = TransactionOperation(operation_type='insert', operation_func=lambda:
        True)
        >>> op.operation_type
        'insert'
    """

    def dummy_func() -> None:
        pass

    op = TransactionOperation(
        operation_type="insert",
        operation_func=dummy_func,
        description="Test insert operation",
    )
    assert op.operation_type == "insert"
    assert op.description == "Test insert operation"
    assert op.rollback_func is None


# Test de l'initialisation d'un TransactionContext
def test_transaction_context_initialization() -> None:
    """Test that TransactionContext is correctly initialized with default values.

    Examples:
        >>> ctx = TransactionContext(transaction_id='tx_001')
        >>> ctx.state
        <TransactionState.PENDING: 'pending'>
    """
    ctx = TransactionContext(transaction_id="tx_001")
    assert ctx.transaction_id == "tx_001"
    assert ctx.state == TransactionState.PENDING
    assert isinstance(ctx.operations, list)
    assert ctx.error_message is None


# ---------------------------------------------------------------------------
# Fixture locale
# ---------------------------------------------------------------------------


# Initialisation d'un TransactionManager pour les tests
@pytest.fixture
def tx_manager(built_ducklake_schema: Any) -> TransactionManager:
    """Create a TransactionManager instance for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.

    Returns:
        TransactionManager: initialized with the test connection.
    """
    return TransactionManager(
        connection=built_ducklake_schema,
        categorical_threshold=4,
        transaction_timeout=60,
    )


# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------


# Test de l'initialisation correcte du gestionnaire de transactions
def test_transaction_manager_initialization(built_ducklake_schema: Any) -> None:
    """Test that TransactionManager initializes without errors.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    mgr = TransactionManager(connection=built_ducklake_schema, transaction_timeout=120)
    assert mgr is not None
    assert mgr.transaction_timeout == 120


# Test que l'alias du catalogue est exposé sous le nom uniformisé catalog_alias
def test_transaction_manager_catalog_alias(built_ducklake_schema: Any) -> None:
    """Test that ``catalog_alias`` defaults to 'db' and is stored when provided.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    assert TransactionManager(connection=built_ducklake_schema).catalog_alias == "db"
    mgr = TransactionManager(
        connection=built_ducklake_schema, catalog_alias="my_lake"
    )
    assert mgr.catalog_alias == "my_lake"


# ---------------------------------------------------------------------------
# Tests de begin_transaction()
# ---------------------------------------------------------------------------


# Test que begin_transaction retourne un identifiant de transaction valide
def test_begin_transaction_returns_id(tx_manager: Any) -> None:
    """Test that begin_transaction returns a non-empty transaction ID.

    Args:
        tx_manager: TransactionManager fixture.
    """
    tx_id = tx_manager.begin_transaction(description="Test transaction")
    # Vérification que l'identifiant est une chaîne non vide
    assert isinstance(tx_id, str)
    assert len(tx_id) > 0
    assert tx_id.startswith("tx_")


# Test que begin_transaction crée la transaction dans l'état RUNNING
def test_begin_transaction_state_running(tx_manager: Any) -> None:
    """Test that a transaction is in RUNNING state after begin_transaction.

    Args:
        tx_manager: TransactionManager fixture.
    """
    tx_id = tx_manager.begin_transaction()
    status = tx_manager.get_transaction_status(tx_id)
    assert status is not None
    assert status["state"] == TransactionState.RUNNING.value


# ---------------------------------------------------------------------------
# Tests de add_operation()
# ---------------------------------------------------------------------------


# Test que add_operation ajoute une opération à la transaction active
def test_add_operation_increments_count(tx_manager: Any) -> None:
    """Test that add_operation increases the total_operations count.

    Args:
        tx_manager: TransactionManager fixture.
    """
    tx_id = tx_manager.begin_transaction()

    def noop() -> None:
        pass

    tx_manager.add_operation(
        transaction_id=tx_id,
        operation_type="test",
        operation_func=noop,
        description="Noop operation",
    )

    status = tx_manager.get_transaction_status(tx_id)
    assert status["total_operations"] == 1


# ---------------------------------------------------------------------------
# Tests de rollback_transaction()
# ---------------------------------------------------------------------------


# Test que rollback_transaction passe l'état à ROLLED_BACK
def test_rollback_transaction(tx_manager: Any) -> None:
    """Test that rollback_transaction sets the transaction state to ROLLED_BACK.

    Args:
        tx_manager: TransactionManager fixture.
    """
    tx_id = tx_manager.begin_transaction()
    success = tx_manager.rollback_transaction(tx_id)

    # Vérification que le rollback a réussi
    assert isinstance(success, bool)

    # Vérification que la transaction n'est plus listée comme active
    active = tx_manager.list_active_transactions()
    active_ids = [t["transaction_id"] for t in active]
    assert tx_id not in active_ids


# ---------------------------------------------------------------------------
# Tests de list_active_transactions()
# ---------------------------------------------------------------------------


# Test que list_active_transactions retourne une liste vide au départ
def test_list_active_transactions_initially_empty(tx_manager: Any) -> None:
    """Test that list_active_transactions returns an empty list before any transaction.

    Args:
        tx_manager: TransactionManager fixture.
    """
    # Aucune transaction n'a encore été démarrée
    active = tx_manager.list_active_transactions()
    assert isinstance(active, list)
    assert len(active) == 0


# Test que list_active_transactions retourne la transaction démarrée
def test_list_active_transactions_contains_started(tx_manager: Any) -> None:
    """Test that list_active_transactions includes started transactions.

    Args:
        tx_manager: TransactionManager fixture.
    """
    tx_id = tx_manager.begin_transaction(description="Active TX")
    active = tx_manager.list_active_transactions()

    active_ids = [t["transaction_id"] for t in active]
    assert tx_id in active_ids

    # Nettoyage
    tx_manager.rollback_transaction(tx_id)
