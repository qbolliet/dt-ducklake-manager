# Importation des modules
# Modules de base
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# DuckDB
import duckdb

# Import du gestionnaire de base
from .base import BaseSchemaManager

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Énumération des états possibles d'une transaction
class TransactionState(Enum):
    """Enumeration of possible transaction states."""

    PENDING = "pending"
    RUNNING = "running"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


# Opérations au seil d'une transaction
@dataclass
class TransactionOperation:
    """
    Represents a single operation within a transaction.

    Attributes:
        operation_type (str): Type of operation ('insert', 'update', 'delete', etc.)
        operation_func (Callable): Function to execute for this operation
        operation_args (tuple): Arguments for the operation function
        operation_kwargs (dict): Keyword arguments for the operation function
        rollback_func (Optional[Callable]): Function to rollback this operation
        rollback_args (tuple): Arguments for the rollback function
        rollback_kwargs (dict): Keyword arguments for the rollback function
        description (str): Human-readable description of the operation
    """

    operation_type: str
    operation_func: Callable
    operation_args: tuple = field(default_factory=tuple)
    operation_kwargs: dict = field(default_factory=dict)
    rollback_func: Callable | None = None
    rollback_args: tuple = field(default_factory=tuple)
    rollback_kwargs: dict = field(default_factory=dict)
    description: str = ""


# Contexte pour suivre les transition d'état et d'opération
@dataclass
class TransactionContext:
    """
    Context for tracking transaction state and operations.

    Attributes:
        transaction_id (str): Unique identifier for the transaction
        state (TransactionState): Current state of the transaction
        operations (List[TransactionOperation]): List of operations in this transaction
        executed_operations (List[int]): Indices of successfully executed operations
        start_time (float): Timestamp when transaction started
        error_message (Optional[str]): Error message if transaction failed
        savepoints (Dict[str, int]): Named savepoints and their operation indices
    """

    transaction_id: str
    state: TransactionState = TransactionState.PENDING
    operations: list[TransactionOperation] = field(default_factory=list)
    executed_operations: list[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    error_message: str | None = None
    savepoints: dict[str, int] = field(default_factory=dict)
    # Identifiant du snapshot DuckLake capturé juste avant le début des opérations.
    # Permet de guider une restauration par time-travel en cas d'échec non récupérable.
    pre_transaction_snapshot: int | None = None


# Classe de gestion des transactions
class TransactionManager(BaseSchemaManager):
    """
    Manages application-level operation batches with rollback support for DuckLake.

    DuckLake does not support multi-statement SQL transactions: each DML
    (INSERT / UPDATE / DELETE) immediately creates an atomic snapshot in the
    catalog. This driver therefore provides an application layer:

    - Each operation is registered with an optional ``rollback_func``.
    - In case of failure, operations already executed are rolled back in reverse order
      via their ``rollback_func``.
    - The DuckLake snapshot captured at startup (``pre_transaction_snapshot``) is
      available in the ``TransactionContext`` to guide a time-travel restoration
      if the ``rollback_func``s themselves fail.

    Attributes:
        transaction_timeout (int): Maximum batch duration in seconds before automatic
            rollback is triggered.
        ducklake_catalog_alias (str): Alias of the DuckLake catalog used to capture
            the pre-operation snapshot version.
        ducklake_schema (str): DuckLake schema name used to capture the snapshot.
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        categorical_threshold: int | None = 50,
        log_filename: os.PathLike | None = None,
        transaction_timeout: int = 300,
        ducklake_catalog_alias: str = "db",
        ducklake_schema: str = "main",
    ):
        """
        Initialize the transaction manager.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            categorical_threshold: Threshold for determining categorical variables.
            log_filename: Path to log file.
            transaction_timeout: Maximum batch duration in seconds before automatic
                rollback is triggered. Defaults to 300.
            ducklake_catalog_alias: Alias of the DuckLake catalog used to capture
                the pre-operation snapshot version. Defaults to ``'db'``.
            ducklake_schema: DuckLake schema name used to capture the snapshot.
                Defaults to ``'main'``.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> tx_mgr = TransactionManager(conn, transaction_timeout=600)
        """
        # Initialisation du parent
        super().__init__(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
        )

        # Configuration du gestionnaire de batches
        self.transaction_timeout = transaction_timeout
        self.ducklake_catalog_alias = ducklake_catalog_alias
        self.ducklake_schema = ducklake_schema

        # Gestion des contextes de transaction applicatifs
        self._active_transactions: dict[str, TransactionContext] = {}
        self._transaction_counter = 0
        self._transaction_lock = threading.RLock()

    # Méthode de validation de l'opération
    def validate_operation(self, operation_type: str, **kwargs) -> bool:
        """
        Validate transaction operations before execution.

        Args:
            operation_type: Type of operation ('begin', 'commit', 'rollback', 'execute')
            **kwargs: Operation-specific parameters

        Returns:
            True if operation is valid
        """
        # Distinction de la méthode de validation suivant le type
        if operation_type == "begin":
            return True  # Les opérations de démarrage sont toujours valides
        elif operation_type == "commit":
            return self._validate_commit(**kwargs)
        elif operation_type == "rollback":
            return self._validate_rollback(**kwargs)
        elif operation_type == "execute":
            return self._validate_execute(**kwargs)
        else:
            # Logging
            self.logger.warning(f"Unknown transaction operation type: {operation_type}")
            return False

    # Méthode de validation d'un commit de transaction
    def _validate_commit(self, transaction_id: str | None = None, **kwargs) -> bool:
        """Validate transaction commit parameters."""
        # Vérification que la transaction possède un identifiant et n'est plus active
        if transaction_id and transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False
        return True

    # Méthode de validation d'un rollback de transaction
    def _validate_rollback(self, transaction_id: str | None = None, **kwargs) -> bool:
        """Validate transaction rollback parameters."""
        # Vérification que la transaction possède un identifiant et n'est plus active
        if transaction_id and transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False
        return True

    # Méthode de validation de l'exécution d'une opération
    def _validate_execute(
        self, transaction_id: str, operation: TransactionOperation, **kwargs
    ) -> bool:
        """Validate operation execution parameters."""
        # Vérification que la transaction n'est pas active
        if transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False
        # Vérification que la transaction n'est pas en attente ou en cours d'exécution
        context = self._active_transactions[transaction_id]
        if context.state not in [TransactionState.PENDING, TransactionState.RUNNING]:
            # Logging
            self.logger.error(
                f"Cannot execute operation in transaction state: {context.state}"
            )
            return False

        return True

    # Méthodes principales de gestion des transactions
    # Méthode d'amorçage d'une nouvelle transaction
    def begin_transaction(self, description: str = "") -> str:
        """
        Begin a new transaction.

        Args:
            description: Optional description of the transaction

        Returns:
            Transaction ID

        Example:
            >>> tx_id = tx_mgr.begin_transaction("Update user data")
            >>> # ... add operations ...
            >>> tx_mgr.commit_transaction(tx_id)
        """
        # Validation de l'opération
        if not self.validate_operation("begin"):
            raise Exception("Cannot begin transaction")

        with self._transaction_lock:
            # Génération d'un ID unique
            self._transaction_counter += 1
            transaction_id = f"tx_{self._transaction_counter}_{int(time.time())}"

            # Capture du snapshot DuckLake courant avant toute opération.
            # Ce numéro de version permet une restauration par time-travel si le
            # rollback applicatif échoue : DuckLakeConnector(..., snapshot_version=N).
            pre_snapshot: int | None = None
            try:
                result = self.conn.execute(
                    f"SELECT MAX(snapshot_id) FROM {self.ducklake_catalog_alias}"
                    f".ducklake_snapshots('{self.ducklake_schema}')"
                ).fetchone()
                pre_snapshot = result[0] if result else None
            except Exception:
                # Capture du snapshot non bloquante : ignorée si le catalogue
                # n'est pas accessible (ex. connexion :memory: pour les tests).
                pass

            # Création du contexte applicatif
            context = TransactionContext(
                transaction_id=transaction_id,
                state=TransactionState.RUNNING,
                pre_transaction_snapshot=pre_snapshot,
            )

            self._active_transactions[transaction_id] = context

            # Logging
            snapshot_info = (
                f" (snapshot avant = {pre_snapshot})"
                if pre_snapshot is not None
                else ""
            )
            self.logger.info(
                f"Started batch {transaction_id}: {description}{snapshot_info}"
            )
            return transaction_id

    # Méthode d'ajout d'une opération sur la base de données à la transaction
    def add_operation(
        self,
        transaction_id: str,
        operation_type: str,
        operation_func: Callable,
        operation_args: tuple = (),
        operation_kwargs: dict = None,
        rollback_func: Callable | None = None,
        rollback_args: tuple = (),
        rollback_kwargs: dict = None,
        description: str = "",
    ) -> bool:
        """
        Add an operation to an existing transaction.

        Args:
            transaction_id: ID of the transaction
            operation_type: Type of operation
            operation_func: Function to execute
            operation_args: Arguments for the function
            operation_kwargs: Keyword arguments for the function
            rollback_func: Function to rollback this operation
            rollback_args: Arguments for rollback function
            rollback_kwargs: Keyword arguments for rollback function
            description: Description of the operation

        Returns:
            True if operation was added successfully

        Example:
            >>> def insert_data(df):
            ...     # Insert logic
            ...     pass
            >>>
            >>> def rollback_insert(table_name):
            ...     # Rollback logic
            ...     pass
            >>>
            >>> tx_mgr.add_operation(
            ...     tx_id, 'insert', insert_data, (df,),
            ...     rollback_func=rollback_insert, rollback_args=('temp_table',)
            ... )
        """
        # Vérification que la transaction n'est pas déjà active
        if transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False

        context = self._active_transactions[transaction_id]

        # Vérification que la transaction n'est pas déjà en cours d'exécution
        if context.state != TransactionState.RUNNING:
            # Logging
            self.logger.error(
                f"Cannot add operation to transaction in state: {context.state}"
            )
            return False

        # Création de l'opération
        operation = TransactionOperation(
            operation_type=operation_type,
            operation_func=operation_func,
            operation_args=operation_args,
            operation_kwargs=operation_kwargs or {},
            rollback_func=rollback_func,
            rollback_args=rollback_args,
            rollback_kwargs=rollback_kwargs or {},
            description=description,
        )

        context.operations.append(operation)

        # Logging
        self.logger.info(
            f"Added operation to {transaction_id}: {operation_type} - {description}"
        )

        return True

    # Méthode d'exécution d'une opération
    def execute_operation(
        self, transaction_id: str, operation_index: int | None = None
    ) -> bool:
        """
        Execute a specific operation or the next pending operation.

        Args:
            transaction_id: ID of the transaction
            operation_index: Index of operation to execute (None for next pending)

        Returns:
            True if operation was executed successfully

        Example:
            >>> # Execute next pending operation
            >>> success = tx_mgr.execute_operation(tx_id)
            >>>
            >>> # Execute specific operation
            >>> success = tx_mgr.execute_operation(tx_id, operation_index=2)
        """
        # Vérification que la transaction fait bien partie des transactions actives
        if transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False

        context = self._active_transactions[transaction_id]

        # Vérification du timeout
        if time.time() - context.start_time > self.transaction_timeout:
            # Logging
            self.logger.error(f"Transaction {transaction_id} timed out")
            # Si timout dépassé annulation de la transaction
            self.rollback_transaction(transaction_id)
            return False

        # Détermination de l'opération à exécuter
        if operation_index is None:
            # Recherche de la prochaine opération non exécutée
            operation_index = len(context.executed_operations)

        # Vérification que l'index est valide
        if operation_index >= len(context.operations):
            # Logging
            self.logger.warning(f"Operation index {operation_index} out of range")
            return False

        # Vérification que l'opération n'est pas déjà exécutée
        if operation_index in context.executed_operations:
            # Logging
            self.logger.warning(f"Operation {operation_index} already executed")
            return True

        operation = context.operations[operation_index]

        # Validation de l'opération
        if not self.validate_operation(
            "execute", transaction_id=transaction_id, operation=operation
        ):
            return False

        try:
            # Exécution de l'opération
            result = operation.operation_func(
                *operation.operation_args, **operation.operation_kwargs
            )

            # Marquage comme exécutée
            context.executed_operations.append(operation_index)

            # Logging
            self.logger.info(
                f"Executed operation {operation_index} in {transaction_id}:"
                f"{operation.description}"
            )
            return True

        except Exception as e:
            # Marquage comme échouée
            context.state = TransactionState.FAILED
            context.error_message = str(e)

            # Logging
            self.logger.error(
                f"Operation {operation_index} failed in {transaction_id}: {e}"
            )
            return False

    # Méthode de commit d'une transaction
    def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction.

        Args:
            transaction_id: ID of the transaction to commit

        Returns:
            True if transaction was committed successfully

        Example:
            >>> success = tx_mgr.commit_transaction(tx_id)
            >>> if not success:
            ...     print("Transaction failed and was rolled back")
        """
        # Validation de l'opération
        if not self.validate_operation("commit", transaction_id=transaction_id):
            return False

        context = self._active_transactions[transaction_id]

        try:
            # Exécution de toutes les opérations restantes
            for i in range(len(context.operations)):
                if i not in context.executed_operations:
                    if not self.execute_operation(transaction_id, i):
                        # En cas d'échec, rollback automatique
                        self.rollback_transaction(transaction_id)
                        return False

            # En DuckLake chaque DML a déjà créé son propre snapshot atomique ;
            # il n'y a pas de COMMIT global à émettre.
            # Mise à jour du statut
            context.state = TransactionState.COMMITTED

            # Logging
            self.logger.info(
                f"Committed transaction {transaction_id} with {len(context.operations)}"
                f" operations"
            )

            # Nettoyage
            self._cleanup_transaction(transaction_id)
            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Failed to commit transaction {transaction_id}: {e}")
            # Annulation de la transaction
            self.rollback_transaction(transaction_id)
            return False

    # Méthode d'annulation d'une transaction
    def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a transaction.

        Args:
            transaction_id: ID of the transaction to rollback

        Returns:
            True if transaction was rolled back successfully

        Example:
            >>> success = tx_mgr.rollback_transaction(tx_id)
        """
        # Validation de l'opération
        if not self.validate_operation("rollback", transaction_id=transaction_id):
            return False

        context = self._active_transactions[transaction_id]

        try:
            # Rollback des opérations individuelles (en ordre inverse)
            rollback_errors = []
            # Parcours des opérations
            for operation_index in reversed(context.executed_operations):
                operation = context.operations[operation_index]
                # Annulation
                if operation.rollback_func:
                    try:
                        # Eécution de l'annulation
                        operation.rollback_func(
                            *operation.rollback_args, **operation.rollback_kwargs
                        )
                        # Logging
                        self.logger.info(
                            f"Rolled back operation {operation_index}:"
                            f"{operation.description}"
                        )
                    except Exception as e:
                        # Ajout aux erreurs
                        rollback_errors.append(f"Operation {operation_index}: {e}")
                        # Logging
                        self.logger.error(
                            f"Failed to rollback operation {operation_index}: {e}"
                        )

            # DuckLake ne supporte pas de ROLLBACK SQL multi-instructions : chaque
            # DML crée un snapshot irrévocable. Le rollback applicatif via rollback_func
            # (ci-dessus) est le seul mécanisme disponible à ce niveau.
            # En cas d'échec des rollback_func, utiliser le time-travel :
            #   DuckLakeConnector(...,
            # snapshot_version=context.pre_transaction_snapshot)
            if context.pre_transaction_snapshot is not None:
                self.logger.info(
                    f"Snapshot de référence pour time-travel : "
                    f"snapshot_version={context.pre_transaction_snapshot}"
                )

            # Mise à jour du statut
            context.state = TransactionState.ROLLED_BACK

            # Logging
            if rollback_errors:
                self.logger.warning(
                    f"Transaction {transaction_id} rolled back with"
                    f" {len(rollback_errors)} rollback errors"
                )
                context.error_message = f"Rollback errors: {'; '.join(rollback_errors)}"
            else:
                self.logger.info(
                    f"Successfully rolled back transaction {transaction_id}"
                )

            # Nettoyage
            self._cleanup_transaction(transaction_id)
            return len(rollback_errors) == 0

        except Exception as e:
            # Logging
            self.logger.error(
                f"Critical error during rollback of {transaction_id}: {e}"
            )
            context.state = TransactionState.FAILED
            context.error_message = str(e)
            return False

    # Méthodes utilitaires
    # Méthode de création d'un savepoint au sein d'une transaction
    def create_savepoint(self, transaction_id: str, savepoint_name: str) -> bool:
        """
        Create a savepoint within a transaction.

        Args:
            transaction_id: ID of the transaction
            savepoint_name: Name of the savepoint

        Returns:
            True if savepoint was created successfully

        Example:
            >>> tx_mgr.create_savepoint(tx_id, "before_critical_operation")
            >>> # ... execute risky operations ...
            >>> tx_mgr.rollback_to_savepoint(tx_id, "before_critical_operation")
        """
        # Vérification que l'identifiant fait bien partie des transactions actives
        if transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False

        context = self._active_transactions[transaction_id]

        # Enregistrement applicatif du savepoint : index de la dernière opération
        # exécutée.
        # DuckLake ne supporte pas les SAVEPOINTs SQL ; le rollback partiel repose
        # exclusivement sur les rollback_func enregistrées par opération.
        context.savepoints[savepoint_name] = len(context.executed_operations)

        # Logging
        self.logger.info(
            f"Created savepoint '{savepoint_name}' in batch {transaction_id}"
        )
        return True

    # Méthode d'annulation jusqu'à un savepoint
    def rollback_to_savepoint(self, transaction_id: str, savepoint_name: str) -> bool:
        """
        Rollback to a specific savepoint.

        Args:
            transaction_id: ID of the transaction
            savepoint_name: Name of the savepoint

        Returns:
            True if rollback was successful
        """
        # Vérification que la transaction ne fait pas partie des transactions actives
        if transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False

        context = self._active_transactions[transaction_id]

        # Vérification que la savepoint est enregistré
        if savepoint_name not in context.savepoints:
            # Logging
            self.logger.error(f"Savepoint '{savepoint_name}' not found")
            return False

        try:
            # Rollback applicatif au savepoint : exécution inverse des rollback_func
            # pour les opérations postérieures au savepoint.
            # DuckLake ne supporte pas ROLLBACK TO SAVEPOINT SQL.
            savepoint_operation_index = context.savepoints[savepoint_name]
            for op_idx in reversed(
                [
                    i
                    for i in context.executed_operations
                    if i >= savepoint_operation_index
                ]
            ):
                operation = context.operations[op_idx]
                if operation.rollback_func:
                    try:
                        operation.rollback_func(
                            *operation.rollback_args, **operation.rollback_kwargs
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Rollback to savepoint: failed for operation {op_idx}: {e}"
                        )
            context.executed_operations = [
                op_idx
                for op_idx in context.executed_operations
                if op_idx < savepoint_operation_index
            ]

            # Suppression des savepoints postérieurs
            savepoints_to_remove = [
                sp_name
                for sp_name, sp_idx in context.savepoints.items()
                if sp_idx >= savepoint_operation_index and sp_name != savepoint_name
            ]
            for sp_name in savepoints_to_remove:
                del context.savepoints[sp_name]

            # Logging
            self.logger.info(
                f"Rolled back to savepoint '{savepoint_name}' in transaction"
                f" {transaction_id}"
            )
            return True

        except Exception as e:
            # Logging
            self.logger.error(
                f"Failed to rollback to savepoint '{savepoint_name}': {e}"
            )
            return False

    # Méthode d'extraction du statut de la transaction
    def get_transaction_status(self, transaction_id: str) -> dict[str, Any] | None:
        """
        Get the status of a transaction.

        Args:
            transaction_id: ID of the transaction

        Returns:
            Dictionary containing transaction status information

        Example:
            >>> status = tx_mgr.get_transaction_status(tx_id)
            >>> print(f"State: {status['state']}, Operations:
            {status['total_operations']}")
        """
        # Vérification que la transaction n'est pas active
        if transaction_id not in self._active_transactions:
            return None

        context = self._active_transactions[transaction_id]

        return {
            "transaction_id": transaction_id,
            "state": context.state.value,
            "total_operations": len(context.operations),
            "executed_operations": len(context.executed_operations),
            "pending_operations": len(context.operations)
            - len(context.executed_operations),
            "start_time": context.start_time,
            "duration": time.time() - context.start_time,
            "savepoints": list(context.savepoints.keys()),
            "error_message": context.error_message,
        }

    # Méthode d'énumération des transactions actives
    def list_active_transactions(self) -> list[dict[str, Any]]:
        """
        List all active transactions.

        Returns:
            List of transaction status dictionaries

        Example:
            >>> active_txs = tx_mgr.list_active_transactions()
            >>> for tx in active_txs:
            ...     print(f"Transaction {tx['transaction_id']}: {tx['state']}")
        """
        with self._transaction_lock:
            return [
                self.get_transaction_status(tx_id)
                for tx_id in list(self._active_transactions.keys())
            ]

    # Méthode de nettoyage des transactions passées
    def cleanup_old_transactions(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old completed transactions.

        Args:
            max_age_seconds: Maximum age for keeping transaction records

        Returns:
            Number of transactions cleaned up

        Example:
            >>> cleaned = tx_mgr.cleanup_old_transactions(max_age_seconds=1800)
            >>> print(f"Cleaned up {cleaned} old transactions")
        """
        # Intialisation du temps
        current_time = time.time()
        # Initialisation du compteur des transactions supprimées
        cleaned_count = 0

        with self._transaction_lock:
            # Identification des transactions à supprimer
            transactions_to_remove = []
            # parcours des transactions
            for tx_id, context in self._active_transactions.items():
                # Nettoyer les transactions terminées depuis longtemps
                if (
                    context.state
                    in [
                        TransactionState.COMMITTED,
                        TransactionState.ROLLED_BACK,
                        TransactionState.FAILED,
                    ]
                    and current_time - context.start_time > max_age_seconds
                ):
                    transactions_to_remove.append(tx_id)
            # Suppression des transactions
            for tx_id in transactions_to_remove:
                # Nettoyage de la transaction
                self._cleanup_transaction(tx_id)
                # Incrémentation du compteur
                cleaned_count += 1

        if cleaned_count > 0:
            # Logging
            self.logger.info(f"Cleaned up {cleaned_count} old transactions")

        return cleaned_count

    # Méthode de nettoyage d'une transaction
    def _cleanup_transaction(self, transaction_id: str) -> None:
        """Clean up the context of a completed transaction."""
        with self._transaction_lock:
            if transaction_id in self._active_transactions:
                del self._active_transactions[transaction_id]

    # Méthodes de convenance pour les opérations atomiques
    # Méthode d'exécution de plusieurs opérations de manière atomique
    def execute_atomic(
        self,
        operations: list[TransactionOperation],
        description: str = "Atomic operation",
    ) -> bool:
        """
        Execute multiple operations atomically.

        Args:
            operations: List of operations to execute
            description: Description of the atomic operation

        Returns:
            True if all operations succeeded

        Example:
            >>> ops = [
            ...     TransactionOperation('insert', insert_func, (df1,)),
            ...     TransactionOperation('update', update_func, (df2,))
            ... ]
            >>> success = tx_mgr.execute_atomic(ops, "Batch data update")
        """
        # Amorçage de la transaction
        tx_id = self.begin_transaction(description)

        try:
            # Ajout de toutes les opérations
            for operation in operations:
                # Ajout de l'opération et si echoue, annulation de cette drnière
                if not self.add_operation(
                    tx_id,
                    operation.operation_type,
                    operation.operation_func,
                    operation.operation_args,
                    operation.operation_kwargs,
                    operation.rollback_func,
                    operation.rollback_args,
                    operation.rollback_kwargs,
                    operation.description,
                ):
                    self.rollback_transaction(tx_id)
                    return False

            # Commit de la transaction
            return self.commit_transaction(tx_id)

        except Exception as e:
            # Logging
            self.logger.error(f"Atomic operation failed: {e}")
            # Annulation de la transaction
            self.rollback_transaction(tx_id)
            return False
