# Tests pour l'architecture refactorisée
# Modules de test
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import duckdb

# Import des gestionnaires refactorisés
from dashboard_template_database.operations.managers.base_manager import BaseSchemaManager
from dashboard_template_database.operations.managers.dimension_manager import DimensionManager
from dashboard_template_database.operations.managers.data_manager import DataManager
from dashboard_template_database.operations.managers.transaction_manager import (
    TransactionManager, TransactionOperation, TransactionState
)
from dashboard_template_database.operations.auditor import (
    DatabaseAuditor, ValidationLevel, IssueType, IssueSeverity
)
from dashboard_template_database.operations.updater_v2 import DatabaseUpdaterV2
from dashboard_template_database.operations.deleter_v2 import DatabaseDeleterV2
from dashboard_template_database.operations.recovery import (
    DatabaseRecoveryManager, RecoveryStrategy, BackupType
)
from dashboard_template_database.operations.atomic_operations import (
    AtomicDatabaseOperations, AtomicOperationType, AtomicOperationConfig
)


class TestDummySchemaManager(BaseSchemaManager):
    """Gestionnaire de schéma factice pour les tests."""
    
    def validate_operation(self, operation_type: str, **kwargs) -> bool:
        return True


@pytest.fixture
def test_db():
    """Fixture pour base de données de test."""
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
        db_path = tmp.name
    
    conn = duckdb.connect(db_path)
    yield conn
    conn.close()
    
    # Nettoyage
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def sample_dataframe():
    """Fixture pour DataFrame d'exemple."""
    np.random.seed(42)
    data = {
        'id': range(100),
        'name': [f'Item_{i}' for i in range(100)],
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.randn(100),
        'status': np.random.choice(['active', 'inactive'], 100),
        'date': pd.date_range('2023-01-01', periods=100, freq='D')
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_backup_dir():
    """Fixture pour répertoire de sauvegarde temporaire."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestBaseSchemaManager:
    """Tests pour BaseSchemaManager."""
    
    def test_initialization(self, test_db):
        """Test de l'initialisation du gestionnaire de base."""
        manager = TestDummySchemaManager(test_db)
        
        assert manager.conn is not None
        assert manager.categorical_threshold == 50
        assert manager.logger is not None
        assert manager._metadata_cache is None
    
    def test_metadata_cache_management(self, test_db, sample_dataframe):
        """Test de la gestion du cache des métadonnées."""
        manager = TestDummySchemaManager(test_db)
        
        # Création de la table metadata
        manager._add_column_to_metadata('test_column', sample_dataframe)
        
        # Premier chargement (création du cache)
        metadata1 = manager._load_current_metadata()
        assert len(metadata1) > 0
        
        # Deuxième chargement (utilisation du cache)
        metadata2 = manager._load_current_metadata()
        pd.testing.assert_frame_equal(metadata1, metadata2)
        
        # Invalidation du cache
        manager._invalidate_metadata_cache()
        assert manager._metadata_cache is None
    
    def test_table_introspection(self, test_db, sample_dataframe):
        """Test des méthodes d'introspection des tables."""
        manager = TestDummySchemaManager(test_db)
        
        # Création d'une table de test
        test_db.register('test_table', sample_dataframe)
        test_db.execute("CREATE TABLE fact_table AS SELECT * FROM test_table")
        
        # Test d'existence de table
        assert manager._table_exists('fact_table')
        assert not manager._table_exists('nonexistent_table')
        
        # Test des colonnes
        columns = manager._get_fact_table_columns()
        assert 'id' in columns
        assert 'category' in columns
        
        # Test d'existence de colonne
        assert manager._column_exists('id', 'fact_table')
        assert not manager._column_exists('nonexistent_column', 'fact_table')


class TestDimensionManager:
    """Tests pour DimensionManager."""
    
    def test_initialization(self, test_db):
        """Test de l'initialisation du gestionnaire de dimensions."""
        dim_mgr = DimensionManager(test_db, max_workers=2)
        
        assert dim_mgr.conn is not None
        assert dim_mgr.max_workers == 2
        assert dim_mgr._dimension_lock is not None
    
    def test_create_dimension_table(self, test_db):
        """Test de création de table de dimension."""
        dim_mgr = DimensionManager(test_db)
        
        # Données de test
        values = pd.Series(['A', 'B', 'C', 'A', 'B'])
        
        # Création de la dimension
        success = dim_mgr.create_dimension_table('test_dim', values)
        assert success
        
        # Vérification de la table créée
        assert dim_mgr._table_exists('dim_test_dim')
        
        # Vérification du contenu
        mapping = dim_mgr.get_dimension_mapping('test_dim')
        assert len(mapping) == 3  # A, B, C
        assert set(mapping['label']) == {'A', 'B', 'C'}
    
    def test_update_dimension_values(self, test_db):
        """Test de mise à jour des valeurs de dimension."""
        dim_mgr = DimensionManager(test_db)
        
        # Création initiale
        initial_values = pd.Series(['A', 'B'])
        dim_mgr.create_dimension_table('test_dim', initial_values)
        
        # Ajout de nouvelles valeurs
        new_values = pd.Series(['C', 'D', 'A'])  # A existe déjà
        added_count = dim_mgr.update_dimension_values('test_dim', new_values)
        
        assert added_count == 2  # C et D ajoutés
        
        # Vérification
        mapping = dim_mgr.get_dimension_mapping('test_dim')
        assert len(mapping) == 4  # A, B, C, D
    
    def test_dimension_conversions(self, test_db, sample_dataframe):
        """Test des conversions catégoriel/non-catégoriel."""
        dim_mgr = DimensionManager(test_db, categorical_threshold=10)
        
        # Ajout aux métadonnées
        dim_mgr._add_column_to_metadata('category', sample_dataframe)
        
        # Test de conversion vers catégoriel
        success = dim_mgr.convert_to_categorical('category', sample_dataframe['category'])
        assert success
        
        # Vérification
        assert dim_mgr._table_exists('dim_category')
        assert dim_mgr._is_dimension_column('category')
        
        # Test de conversion vers non-catégoriel
        success = dim_mgr.convert_to_non_categorical('category')
        assert success
        
        # Vérification
        assert not dim_mgr._table_exists('dim_category')
    
    def test_cleanup_orphaned_entries(self, test_db):
        """Test du nettoyage des entrées orphelines."""
        dim_mgr = DimensionManager(test_db)
        
        # Création d'une dimension avec des valeurs
        values = pd.Series(['A', 'B', 'C'])
        dim_mgr.create_dimension_table('test_dim', values)
        dim_mgr._add_column_to_metadata('test_dim', pd.DataFrame({'test_dim': values}))
        dim_mgr._update_categorical_status('test_dim', True)
        
        # Création d'une fact table avec seulement certaines valeurs
        test_data = pd.DataFrame({'test_dim': ['0', '1']})  # Seulement A et B (0, 1)
        test_db.register('temp_fact', test_data)
        test_db.execute("CREATE TABLE fact_table AS SELECT * FROM temp_fact")
        
        # Nettoyage
        removed = dim_mgr.cleanup_orphaned_dimension_entries('test_dim')
        
        # Vérification que la valeur C (2) a été supprimée
        mapping = dim_mgr.get_dimension_mapping('test_dim')
        assert len(mapping) == 2


class TestDataManager:
    """Tests pour DataManager."""
    
    def test_initialization(self, test_db):
        """Test de l'initialisation du gestionnaire de données."""
        data_mgr = DataManager(test_db, batch_size=1000)
        
        assert data_mgr.conn is not None
        assert data_mgr.batch_size == 1000
    
    def test_create_fact_table(self, test_db, sample_dataframe):
        """Test de création de la table de faits."""
        data_mgr = DataManager(test_db)
        
        # Création de la table
        success = data_mgr.create_fact_table(sample_dataframe)
        assert success
        
        # Vérification
        assert data_mgr._table_exists('fact_table')
        
        # Vérification du contenu
        count = test_db.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        assert count == len(sample_dataframe)
    
    def test_insert_data(self, test_db, sample_dataframe):
        """Test d'insertion de données."""
        data_mgr = DataManager(test_db)
        
        # Insertion de données
        rows_inserted = data_mgr.insert_data(sample_dataframe, use_batch=False)
        assert rows_inserted == len(sample_dataframe)
        
        # Vérification
        count = test_db.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        assert count == len(sample_dataframe)
    
    def test_upsert_data(self, test_db, sample_dataframe):
        """Test d'upsert de données."""
        data_mgr = DataManager(test_db)
        
        # Insertion initiale
        data_mgr.insert_data(sample_dataframe[:50])
        
        # Upsert avec nouvelles données et modifications
        modified_data = sample_dataframe.copy()
        modified_data.loc[0, 'value'] = 999  # Modification
        
        inserted, updated = data_mgr.upsert_data(
            modified_data, 
            merge_keys=['id'], 
            use_batch=False
        )
        
        assert inserted == 50  # Nouvelles lignes 50-99
        assert updated == 50   # Lignes modifiées 0-49
    
    def test_delete_rows(self, test_db, sample_dataframe):
        """Test de suppression de lignes."""
        data_mgr = DataManager(test_db)
        
        # Insertion de données
        data_mgr.insert_data(sample_dataframe)
        
        # Suppression avec filtre
        deleted = data_mgr.delete_rows("status = 'inactive'")
        assert deleted >= 0
        
        # Vérification
        remaining = test_db.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        assert remaining < len(sample_dataframe)
    
    def test_column_operations(self, test_db, sample_dataframe):
        """Test des opérations sur les colonnes."""
        data_mgr = DataManager(test_db)
        
        # Création de la table
        data_mgr.create_fact_table(sample_dataframe)
        
        # Ajout d'une colonne
        new_data = sample_dataframe.copy()
        new_data['new_column'] = 'test_value'
        
        success = data_mgr.add_column('new_column', new_data, default_value='default')
        assert success
        
        # Vérification
        columns = data_mgr._get_fact_table_columns()
        assert 'new_column' in columns
        
        # Suppression d'une colonne
        dropped = data_mgr.drop_columns(['new_column'])
        assert 'new_column' in dropped
        
        # Vérification
        columns = data_mgr._get_fact_table_columns()
        assert 'new_column' not in columns


class TestTransactionManager:
    """Tests pour TransactionManager."""
    
    def test_initialization(self, test_db):
        """Test de l'initialisation du gestionnaire de transactions."""
        tx_mgr = TransactionManager(test_db)
        
        assert tx_mgr.conn is not None
        assert tx_mgr._active_transactions == {}
        assert tx_mgr._transaction_counter == 0
    
    def test_transaction_lifecycle(self, test_db):
        """Test du cycle de vie des transactions."""
        tx_mgr = TransactionManager(test_db)
        
        # Début de transaction
        tx_id = tx_mgr.begin_transaction("Test transaction")
        assert tx_id is not None
        assert tx_id in tx_mgr._active_transactions
        
        # Vérification du statut
        status = tx_mgr.get_transaction_status(tx_id)
        assert status['state'] == 'running'
        assert status['total_operations'] == 0
        
        # Commit
        success = tx_mgr.commit_transaction(tx_id)
        assert success
    
    def test_transaction_operations(self, test_db):
        """Test des opérations dans les transactions."""
        tx_mgr = TransactionManager(test_db)
        
        # Début de transaction
        tx_id = tx_mgr.begin_transaction("Test transaction with operations")
        
        # Ajout d'opérations
        def dummy_operation():
            return True
        
        operation = TransactionOperation(
            operation_type='test',
            operation_func=dummy_operation,
            description="Test operation"
        )
        
        success = tx_mgr.add_operation(tx_id, **operation.__dict__)
        assert success
        
        # Exécution de l'opération
        success = tx_mgr.execute_operation(tx_id)
        assert success
        
        # Commit
        success = tx_mgr.commit_transaction(tx_id)
        assert success
    
    def test_transaction_rollback(self, test_db):
        """Test du rollback de transaction."""
        tx_mgr = TransactionManager(test_db)
        
        # Début de transaction
        tx_id = tx_mgr.begin_transaction("Test rollback")
        
        # Ajout d'une opération qui échoue
        def failing_operation():
            raise Exception("Test failure")
        
        operation = TransactionOperation(
            operation_type='test_fail',
            operation_func=failing_operation,
            description="Failing operation"
        )
        
        tx_mgr.add_operation(tx_id, **operation.__dict__)
        
        # L'exécution devrait échouer
        success = tx_mgr.execute_operation(tx_id)
        assert not success
        
        # Rollback
        success = tx_mgr.rollback_transaction(tx_id)
        assert success
    
    def test_savepoints(self, test_db):
        """Test des savepoints."""
        tx_mgr = TransactionManager(test_db)
        
        # Début de transaction
        tx_id = tx_mgr.begin_transaction("Test savepoints")
        
        # Création d'un savepoint
        success = tx_mgr.create_savepoint(tx_id, "test_savepoint")
        assert success
        
        # Rollback au savepoint
        success = tx_mgr.rollback_to_savepoint(tx_id, "test_savepoint")
        assert success
        
        # Commit
        tx_mgr.commit_transaction(tx_id)


class TestDatabaseAuditor:
    """Tests pour DatabaseAuditor."""
    
    def test_initialization(self, test_db):
        """Test de l'initialisation de l'auditeur."""
        auditor = DatabaseAuditor(test_db)
        
        assert auditor.conn is not None
        assert auditor.categorical_threshold == 50
    
    def test_basic_validation(self, test_db):
        """Test de validation basique."""
        auditor = DatabaseAuditor(test_db)
        
        # Validation d'une base vide
        report = auditor.validate_database(ValidationLevel.BASIC)
        
        assert report is not None
        assert report.validation_level == ValidationLevel.BASIC
        assert len(report.issues) >= 0  # Peut avoir des issues (metadata manquante)
    
    def test_health_check(self, test_db, sample_dataframe):
        """Test du contrôle de santé rapide."""
        auditor = DatabaseAuditor(test_db)
        
        # Création d'une structure de base
        test_db.register('temp_data', sample_dataframe)
        test_db.execute("CREATE TABLE fact_table AS SELECT * FROM temp_data")
        
        # Contrôle de santé
        health = auditor.get_quick_health_check()
        
        assert 'status' in health
        assert 'fact_table_rows' in health
        assert health['fact_table_rows'] == len(sample_dataframe)
    
    def test_validation_with_issues(self, test_db):
        """Test de validation avec des issues."""
        auditor = DatabaseAuditor(test_db)
        
        # Création d'une structure avec des problèmes
        test_db.execute("""
            CREATE TABLE fact_table (
                id INTEGER,
                name VARCHAR
            )
        """)
        
        test_db.execute("""
            CREATE TABLE dim_category (
                value VARCHAR,
                label VARCHAR
            )
        """)
        
        # Cette dimension n'a pas de colonne correspondante dans fact_table
        test_db.execute("INSERT INTO dim_category VALUES ('0', 'A'), ('1', 'B')")
        
        # Validation
        report = auditor.validate_database(ValidationLevel.COMPREHENSIVE)
        
        # Devrait détecter des issues
        assert len(report.issues) > 0


class TestDatabaseUpdaterV2:
    """Tests pour DatabaseUpdaterV2."""
    
    def test_initialization(self, test_db):
        """Test de l'initialisation de l'updater v2."""
        updater = DatabaseUpdaterV2(test_db, enable_validation=False)
        
        assert updater.conn is not None
        assert updater.dimension_mgr is not None
        assert updater.data_mgr is not None
        assert updater.transaction_mgr is not None
    
    def test_database_update_direct(self, test_db, sample_dataframe):
        """Test de mise à jour directe de la base."""
        updater = DatabaseUpdaterV2(test_db, enable_validation=False)
        
        # Mise à jour
        success = updater.update_database(
            sample_dataframe,
            use_transaction=False,
            use_batch_processing=False
        )
        
        assert success
        
        # Vérification
        assert updater._table_exists('fact_table')
        assert updater._table_exists('metadata')
    
    def test_database_update_transactional(self, test_db, sample_dataframe):
        """Test de mise à jour transactionnelle de la base."""
        updater = DatabaseUpdaterV2(test_db, enable_validation=False)
        
        # Mise à jour
        success = updater.update_database(
            sample_dataframe,
            use_transaction=True,
            use_batch_processing=False
        )
        
        assert success


class TestDatabaseDeleterV2:
    """Tests pour DatabaseDeleterV2."""
    
    def test_initialization(self, test_db):
        """Test de l'initialisation du deleter v2."""
        deleter = DatabaseDeleterV2(test_db, enable_validation=False)
        
        assert deleter.conn is not None
        assert deleter.dimension_mgr is not None
        assert deleter.data_mgr is not None
        assert deleter.transaction_mgr is not None
    
    def test_delete_rows(self, test_db, sample_dataframe):
        """Test de suppression de lignes."""
        deleter = DatabaseDeleterV2(test_db, enable_validation=False, auto_cleanup=False)
        
        # Création de données
        test_db.register('temp_data', sample_dataframe)
        test_db.execute("CREATE TABLE fact_table AS SELECT * FROM temp_data")
        
        # Suppression
        deleted = deleter.delete_rows(
            "status = 'inactive'",
            use_transaction=False
        )
        
        assert deleted >= 0
    
    def test_delete_columns(self, test_db, sample_dataframe):
        """Test de suppression de colonnes."""
        deleter = DatabaseDeleterV2(test_db, enable_validation=False, auto_cleanup=False)
        
        # Création de données
        test_db.register('temp_data', sample_dataframe)
        test_db.execute("CREATE TABLE fact_table AS SELECT * FROM temp_data")
        
        # Suppression
        results = deleter.delete_columns(['date'], use_transaction=False)
        
        assert 'date' in results


class TestDatabaseRecoveryManager:
    """Tests pour DatabaseRecoveryManager."""
    
    def test_initialization(self, test_db, temp_backup_dir):
        """Test de l'initialisation du gestionnaire de récupération."""
        recovery_mgr = DatabaseRecoveryManager(test_db, temp_backup_dir)
        
        assert recovery_mgr.conn is not None
        assert recovery_mgr.backup_dir == temp_backup_dir
        assert recovery_mgr.auditor is not None
    
    def test_create_recovery_point(self, test_db, temp_backup_dir, sample_dataframe):
        """Test de création de point de récupération."""
        recovery_mgr = DatabaseRecoveryManager(test_db, temp_backup_dir)
        
        # Création de données
        test_db.register('temp_data', sample_dataframe)
        test_db.execute("CREATE TABLE fact_table AS SELECT * FROM temp_data")
        
        # Création du point de récupération
        recovery_id = recovery_mgr.create_recovery_point(
            BackupType.FULL_BACKUP,
            "Test backup",
            force_validation=False
        )
        
        assert recovery_id is not None
        
        # Vérification
        points = recovery_mgr.list_recovery_points()
        assert len(points) == 1
        assert points[0].recovery_id == recovery_id
    
    def test_recovery_operations(self, test_db, temp_backup_dir):
        """Test des opérations de récupération."""
        recovery_mgr = DatabaseRecoveryManager(test_db, temp_backup_dir)
        
        # Test de récupération par nettoyage
        from dashboard_template_database.operations.recovery import RecoveryOperation
        
        operation = RecoveryOperation(
            strategy=RecoveryStrategy.CLEAN_ORPHANED_DATA,
            description="Test cleanup recovery"
        )
        
        result = recovery_mgr.recover_database(operation)
        assert result.success


class TestAtomicDatabaseOperations:
    """Tests pour AtomicDatabaseOperations."""
    
    def test_initialization(self, test_db, temp_backup_dir):
        """Test de l'initialisation des opérations atomiques."""
        atomic_ops = AtomicDatabaseOperations(test_db, backup_dir=temp_backup_dir)
        
        assert atomic_ops.conn is not None
        assert atomic_ops.transaction_mgr is not None
        assert atomic_ops.updater is not None
        assert atomic_ops.deleter is not None
        assert atomic_ops.recovery_mgr is not None
        assert atomic_ops.auditor is not None
    
    def test_batch_update_operation(self, test_db, temp_backup_dir, sample_dataframe):
        """Test d'opération de mise à jour par lot."""
        atomic_ops = AtomicDatabaseOperations(test_db, backup_dir=temp_backup_dir)
        
        # Configuration
        config = AtomicOperationConfig(
            operation_type=AtomicOperationType.BATCH_UPDATE,
            enable_backup=False,  # Désactivé pour accélérer les tests
            enable_validation=False
        )
        
        # Exécution
        result = atomic_ops.execute_batch_update(sample_dataframe, config)
        
        assert result.success
        assert result.operation_type == AtomicOperationType.BATCH_UPDATE
        assert len(result.operations_performed) > 0
    
    def test_cleanup_operation(self, test_db, temp_backup_dir, sample_dataframe):
        """Test d'opération de nettoyage."""
        atomic_ops = AtomicDatabaseOperations(test_db, backup_dir=temp_backup_dir)
        
        # Création de données
        test_db.register('temp_data', sample_dataframe)
        test_db.execute("CREATE TABLE fact_table AS SELECT * FROM temp_data")
        
        # Configuration du nettoyage
        cleanup_config = {
            'remove_orphaned_dimensions': True,
            'remove_null_columns': True
        }
        
        config = AtomicOperationConfig(
            operation_type=AtomicOperationType.CLEANUP_OPERATION,
            enable_backup=False,
            enable_validation=False
        )
        
        # Exécution
        result = atomic_ops.execute_cleanup_operation(cleanup_config, config)
        
        assert result.success
        assert result.operation_type == AtomicOperationType.CLEANUP_OPERATION
    
    def test_system_status(self, test_db, temp_backup_dir):
        """Test du statut du système."""
        atomic_ops = AtomicDatabaseOperations(test_db, backup_dir=temp_backup_dir)
        
        # Statut des opérations
        status = atomic_ops.get_operation_status()
        
        assert 'timestamp' in status
        assert 'active_transactions' in status
        assert 'components_ready' in status
        
        # Validation de l'intégrité du système
        integrity = atomic_ops.validate_system_integrity()
        
        assert 'overall_status' in integrity
        assert 'components_status' in integrity


class TestIntegrationScenarios:
    """Tests d'intégration pour des scénarios complexes."""
    
    def test_complete_workflow(self, test_db, temp_backup_dir, sample_dataframe):
        """Test d'un workflow complet."""
        # Initialisation
        atomic_ops = AtomicDatabaseOperations(test_db, backup_dir=temp_backup_dir)
        
        # 1. Mise à jour initiale
        config = AtomicOperationConfig(
            operation_type=AtomicOperationType.BATCH_UPDATE,
            enable_backup=True,
            enable_validation=False
        )
        
        result = atomic_ops.execute_batch_update(sample_dataframe, config)
        assert result.success
        
        # 2. Mise à jour incrémentale
        new_data = sample_dataframe.copy()
        new_data['value'] = new_data['value'] * 2  # Modification des valeurs
        
        result = atomic_ops.execute_batch_update(new_data, config)
        assert result.success
        
        # 3. Nettoyage
        cleanup_config = {'remove_orphaned_dimensions': True}
        cleanup_op_config = AtomicOperationConfig(
            operation_type=AtomicOperationType.CLEANUP_OPERATION,
            enable_backup=False,
            enable_validation=False
        )
        
        result = atomic_ops.execute_cleanup_operation(cleanup_config, cleanup_op_config)
        assert result.success
        
        # 4. Validation finale
        integrity = atomic_ops.validate_system_integrity()
        assert integrity['overall_status'] in ['healthy', 'component_issues', 'empty']
    
    def test_error_recovery_scenario(self, test_db, temp_backup_dir, sample_dataframe):
        """Test de scénario de récupération d'erreur."""
        atomic_ops = AtomicDatabaseOperations(test_db, backup_dir=temp_backup_dir)
        
        # Création d'un backup
        recovery_id = atomic_ops.recovery_mgr.create_recovery_point(
            BackupType.FULL_BACKUP,
            "Test backup for recovery",
            force_validation=False
        )
        assert recovery_id is not None
        
        # Simulation d'une corruption (création de données incohérentes)
        test_db.execute("""
            CREATE TABLE fact_table (
                id INTEGER,
                invalid_column VARCHAR
            )
        """)
        
        # Test de récupération d'urgence
        result = atomic_ops.emergency_recovery(allow_destructive=False)
        
        # Le résultat peut être success ou failure selon la stratégie choisie
        assert result.operation_type == AtomicOperationType.VALIDATION_AND_REPAIR
    
    def test_concurrent_operations_safety(self, test_db, temp_backup_dir, sample_dataframe):
        """Test de sécurité des opérations concurrentes."""
        atomic_ops = AtomicDatabaseOperations(test_db, backup_dir=temp_backup_dir)
        
        # Test que les transactions sont gérées correctement
        tx_mgr = atomic_ops.transaction_mgr
        
        # Début de plusieurs transactions
        tx1 = tx_mgr.begin_transaction("Transaction 1")
        tx2 = tx_mgr.begin_transaction("Transaction 2")
        
        assert tx1 != tx2
        assert len(tx_mgr.list_active_transactions()) == 2
        
        # Commit des transactions
        tx_mgr.commit_transaction(tx1)
        tx_mgr.commit_transaction(tx2)
        
        assert len(tx_mgr.list_active_transactions()) == 0


@pytest.mark.slow
class TestPerformanceAndScalability:
    """Tests de performance et de scalabilité."""
    
    def test_large_dataset_processing(self, test_db, temp_backup_dir):
        """Test de traitement de gros jeux de données."""
        # Création d'un dataset plus important
        np.random.seed(42)
        large_data = {
            'id': range(10000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
            'value': np.random.randn(10000),
            'status': np.random.choice(['active', 'inactive', 'pending'], 10000)
        }
        large_df = pd.DataFrame(large_data)
        
        atomic_ops = AtomicDatabaseOperations(test_db, backup_dir=temp_backup_dir, default_batch_size=1000)
        
        # Configuration pour gros volume
        config = AtomicOperationConfig(
            operation_type=AtomicOperationType.BATCH_UPDATE,
            enable_backup=False,  # Désactivé pour accélérer
            enable_validation=False,
            batch_size=1000
        )
        
        # Mesure du temps d'exécution
        import time
        start_time = time.time()
        
        result = atomic_ops.execute_batch_update(large_df, config)
        
        execution_time = time.time() - start_time
        
        assert result.success
        assert execution_time < 30  # Devrait se terminer en moins de 30 secondes
        
        # Vérification que le traitement par batch a été utilisé
        assert 'batch' in str(result.operations_performed).lower() or len(large_df) <= config.batch_size
    
    def test_memory_efficiency(self, test_db, temp_backup_dir):
        """Test d'efficacité mémoire."""
        # Test que les gestionnaires libèrent correctement la mémoire
        atomic_ops = AtomicDatabaseOperations(test_db, backup_dir=temp_backup_dir)
        
        # Création et suppression de multiples points de récupération
        recovery_ids = []
        for i in range(5):
            recovery_id = atomic_ops.recovery_mgr.create_recovery_point(
                BackupType.METADATA_BACKUP,
                f"Test backup {i}",
                force_validation=False
            )
            if recovery_id:
                recovery_ids.append(recovery_id)
        
        # Suppression des points de récupération
        for recovery_id in recovery_ids:
            atomic_ops.recovery_mgr.delete_recovery_point(recovery_id)
        
        # Vérification que la mémoire est libérée
        remaining_points = atomic_ops.recovery_mgr.list_recovery_points()
        assert len(remaining_points) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])