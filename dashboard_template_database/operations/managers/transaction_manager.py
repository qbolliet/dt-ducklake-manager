# Importation des modules
# Modules de base
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
# DuckDB
import duckdb

# Import du gestionnaire de base
from .base_manager import BaseSchemaManager

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Énumération des états possibles d'une transaction
class TransactionState(Enum):
    """Énumération des états possibles d'une transaction."""
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
    rollback_func: Optional[Callable] = None
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
    operations: List[TransactionOperation] = field(default_factory=list)
    executed_operations: List[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    savepoints: Dict[str, int] = field(default_factory=dict)

# Classe de gestion des transactions
class TransactionManager(BaseSchemaManager):
    """
    Manages database transactions with support for atomic operations, rollback,
    and error recovery. Provides high-level transaction management with automatic
    cleanup and state validation.
    
    Attributes:
        auto_commit (bool): Whether to auto-commit single operations
        transaction_timeout (int): Maximum transaction duration in seconds
    """

    # Initialisation
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None,
                 auto_commit: bool = True,
                 transaction_timeout: int = 300):
        """
        Initialize the transaction manager.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Threshold for determining categorical variables
            log_filename: Path to log file
            auto_commit: Whether to auto-commit single operations
            transaction_timeout: Maximum transaction duration in seconds
            
        Example:
            >>> conn = duckdb.connect('database.db')
            >>> tx_mgr = TransactionManager(conn, transaction_timeout=600)
        """
        # Initialisation du parent
        super().__init__(connection, categorical_threshold, log_filename)
        
        # Configuration des transactions
        self.auto_commit = auto_commit
        self.transaction_timeout = transaction_timeout
        
        # Gestion des contextes de transaction
        self._active_transactions: Dict[str, TransactionContext] = {}
        self._transaction_counter = 0
        self._transaction_lock = threading.RLock()
        
        # Configuration de l'auto-commit de DuckDB
        try:
            self.conn.execute("SET autocommit = ?", [auto_commit])
        except Exception as e:
            self.logger.warning(f"Could not set autocommit mode: {e}")
    
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
        if operation_type == 'begin':
            return True  # Les opérations de démarrage sont toujours valides
        elif operation_type == 'commit':
            return self._validate_commit(**kwargs)
        elif operation_type == 'rollback':
            return self._validate_rollback(**kwargs)
        elif operation_type == 'execute':
            return self._validate_execute(**kwargs)
        else:
            # Logging
            self.logger.warning(f"Unknown transaction operation type: {operation_type}")
            return False
    
    # Méthode de validation d'un commit de transaction
    def _validate_commit(self, transaction_id: Optional[str] = None, **kwargs) -> bool:
        """Valider un commit de transaction."""
        # Vérification que la transaction possède un identifiant et n'est plus active
        if transaction_id and transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False
        return True
    
    # Méthode de validation d'un rollback de transaction
    def _validate_rollback(self, transaction_id: Optional[str] = None, **kwargs) -> bool:
        """Valider un rollback de transaction."""
        # Vérification que la transaction possède un identifiant et n'est plus active
        if transaction_id and transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False
        return True
    
    # Méthode de validation de l'exécution d'une opération
    def _validate_execute(self, transaction_id: str, operation: TransactionOperation, **kwargs) -> bool:
        """Valider l'exécution d'une opération."""
        # Vérification que la transaction n'est pas active
        if transaction_id not in self._active_transactions:
            # Logging
            self.logger.error(f"Transaction {transaction_id} not found")
            return False
        # Vérification que la transaction n'est pas en attente ou en cours d'exécution
        context = self._active_transactions[transaction_id]
        if context.state not in [TransactionState.PENDING, TransactionState.RUNNING]:
            # Logging
            self.logger.error(f"Cannot execute operation in transaction state: {context.state}")
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
        if not self.validate_operation('begin'):
            raise Exception("Cannot begin transaction")
        
        with self._transaction_lock:
            # Génération d'un ID unique
            self._transaction_counter += 1
            transaction_id = f"tx_{self._transaction_counter}_{int(time.time())}"
            
            # Création du contexte
            context = TransactionContext(
                transaction_id=transaction_id,
                state=TransactionState.PENDING
            )
            
            self._active_transactions[transaction_id] = context
            
            try:
                # Début de la transaction DuckDB
                self.conn.execute("BEGIN TRANSACTION")
                context.state = TransactionState.RUNNING
                
                # Logging
                self.logger.info(f"Started transaction {transaction_id}: {description}")
                return transaction_id
                
            except Exception as e:
                # Nettoyage en cas d'erreur
                del self._active_transactions[transaction_id]
                # Logging
                self.logger.error(f"Failed to begin transaction: {e}")
                raise
    
    # Méthode d'ajout d'une opération sur la base de données à la transaction
    def add_operation(self, 
                     transaction_id: str,
                     operation_type: str,
                     operation_func: Callable,
                     operation_args: tuple = (),
                     operation_kwargs: dict = None,
                     rollback_func: Optional[Callable] = None,
                     rollback_args: tuple = (),
                     rollback_kwargs: dict = None,
                     description: str = "") -> bool:
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
            self.logger.error(f"Cannot add operation to transaction in state: {context.state}")
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
            description=description
        )
        
        context.operations.append(operation)
        
        # Logging
        self.logger.info(f"Added operation to {transaction_id}: {operation_type} - {description}")

        return True
    
    # Méthode d'exécution d'une opération
    def execute_operation(self, transaction_id: str, operation_index: Optional[int] = None) -> bool:
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
        if not self.validate_operation('execute', transaction_id=transaction_id, operation=operation):
            return False
        
        try:
            # Exécution de l'opération
            result = operation.operation_func(*operation.operation_args, **operation.operation_kwargs)
            
            # Marquage comme exécutée
            context.executed_operations.append(operation_index)
            
            # Logging
            self.logger.info(f"Executed operation {operation_index} in {transaction_id}: {operation.description}")
            return True
            
        except Exception as e:
            # Marquage comme échouée
            context.state = TransactionState.FAILED
            context.error_message = str(e)
            
            # Logging
            self.logger.error(f"Operation {operation_index} failed in {transaction_id}: {e}")
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
        if not self.validate_operation('commit', transaction_id=transaction_id):
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
            
            # Commit de la transaction DuckDB
            self.conn.execute("COMMIT")
            
            # Mise à jour du statut
            context.state = TransactionState.COMMITTED
            
            # Logging
            self.logger.info(f"Committed transaction {transaction_id} with {len(context.operations)} operations")
            
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
        if not self.validate_operation('rollback', transaction_id=transaction_id):
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
                        operation.rollback_func(*operation.rollback_args, **operation.rollback_kwargs)
                        # Logging
                        self.logger.info(f"Rolled back operation {operation_index}: {operation.description}")
                    except Exception as e:
                        # Ajout aux erreurs
                        rollback_errors.append(f"Operation {operation_index}: {e}")
                        # Logging
                        self.logger.error(f"Failed to rollback operation {operation_index}: {e}")
            
            # Rollback de la transaction DuckDB
            try:
                # Exécution de l'annulation
                self.conn.execute("ROLLBACK")
            except Exception as e:
                # Logging
                self.logger.warning(f"DuckDB rollback failed: {e}")
            
            # Mise à jour du statut
            context.state = TransactionState.ROLLED_BACK
            
            # Logging
            if rollback_errors:
                self.logger.warning(f"Transaction {transaction_id} rolled back with {len(rollback_errors)} rollback errors")
                context.error_message = f"Rollback errors: {'; '.join(rollback_errors)}"
            else:
                self.logger.info(f"Successfully rolled back transaction {transaction_id}")
            
            # Nettoyage
            self._cleanup_transaction(transaction_id)
            return len(rollback_errors) == 0
            
        except Exception as e:
            # Logging
            self.logger.error(f"Critical error during rollback of {transaction_id}: {e}")
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
        
        try:
            # Création du savepoint DuckDB
            self.conn.execute(f"SAVEPOINT {savepoint_name}")
            
            # Enregistrement du savepoint
            context.savepoints[savepoint_name] = len(context.executed_operations)
            
            # Logging
            self.logger.info(f"Created savepoint '{savepoint_name}' in transaction {transaction_id}")
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to create savepoint '{savepoint_name}': {e}")
            return False
    
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
            # Rollback DuckDB au savepoint
            self.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            
            # Mise à jour du contexte local
            savepoint_operation_index = context.savepoints[savepoint_name]
            context.executed_operations = [
                op_idx for op_idx in context.executed_operations 
                if op_idx < savepoint_operation_index
            ]
            
            # Suppression des savepoints postérieurs
            savepoints_to_remove = [
                sp_name for sp_name, sp_idx in context.savepoints.items()
                if sp_idx >= savepoint_operation_index and sp_name != savepoint_name
            ]
            for sp_name in savepoints_to_remove:
                del context.savepoints[sp_name]
            
            # Logging
            self.logger.info(f"Rolled back to savepoint '{savepoint_name}' in transaction {transaction_id}")
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to rollback to savepoint '{savepoint_name}': {e}")
            return False
    
    # Méthode d'extraction du statut de la transaction
    def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a transaction.
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            Dictionary containing transaction status information
            
        Example:
            >>> status = tx_mgr.get_transaction_status(tx_id)
            >>> print(f"State: {status['state']}, Operations: {status['total_operations']}")
        """
        # Vérification que la transaction n'est pas active
        if transaction_id not in self._active_transactions:
            return None
        
        context = self._active_transactions[transaction_id]
        
        return {
            'transaction_id': transaction_id,
            'state': context.state.value,
            'total_operations': len(context.operations),
            'executed_operations': len(context.executed_operations),
            'pending_operations': len(context.operations) - len(context.executed_operations),
            'start_time': context.start_time,
            'duration': time.time() - context.start_time,
            'savepoints': list(context.savepoints.keys()),
            'error_message': context.error_message
        }
    
    # Méthode d'énumération des transactions actives
    def list_active_transactions(self) -> List[Dict[str, Any]]:
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
                if (context.state in [TransactionState.COMMITTED, TransactionState.ROLLED_BACK, TransactionState.FAILED] and
                    current_time - context.start_time > max_age_seconds):
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
        """Nettoyer le contexte d'une transaction terminée."""
        with self._transaction_lock:
            if transaction_id in self._active_transactions:
                del self._active_transactions[transaction_id]
    
    # Méthodes de convenance pour les opérations atomiques
    # Méthode d'exécution de plusieurs opérations de manière atomique
    def execute_atomic(self, 
                      operations: List[TransactionOperation], 
                      description: str = "Atomic operation") -> bool:
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
                    operation.description
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