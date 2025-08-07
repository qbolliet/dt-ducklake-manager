#!/usr/bin/env python3
"""
Lightweight updater script specifically for model results.
Designed to be called by separate model containers.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml
import logging

# Ajout du chemin pour les imports
sys.path.insert(0, '/app')

from dashboard_template_database.manager import DatabaseManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_updater')

class ModelResultUpdater:
    """
    Specialized updater for model results.
    Handles common patterns for model outputs.
    """
    
    def __init__(self, database_name: str, database_path: str = None):
        """
        Initialize the updater.
        
        Args:
            database_name: Name of the target database
            database_path: Path to database directory
        """
        self.database_name = database_name
        self.database_path = database_path or os.environ.get('DATABASE_PATH', '/data/databases')
        self.manager = None
        
    def connect(self):
        """Establish database connection."""
        self.manager = DatabaseManager(
            database_path=self.database_path,
            database_name=self.database_name
        )
        
        if not self.manager.database_exists():
            raise ValueError(f"Database {self.database_name} does not exist")
        
        logger.info(f"Connected to database: {self.database_name}")
        
    def update_from_model_output(self, 
                                data: pd.DataFrame,
                                model_name: str,
                                model_version: str = None,
                                metadata: dict = None) -> dict:
        """
        Update database with model output.
        
        Args:
            data: DataFrame with model results
            model_name: Name of the model
            model_version: Version of the model
            metadata: Additional metadata (e.g., run parameters, metrics)
            
        Returns:
            Update statistics
        """
        # Ajout de métadonnées au DataFrame
        data = data.copy()
        data['model_name'] = model_name
        data['model_version'] = model_version or 'latest'
        data['update_timestamp'] = datetime.now()
        
        # Si des métadonnées sont fournies, les ajouter
        if metadata:
            for key, value in metadata.items():
                if key not in data.columns:
                    data[key] = value
        
        # Détermination des clés de fusion basées sur le type de modèle
        merge_keys = self._determine_merge_keys(data, model_name)
        
        # Mise à jour de la base
        stats = self.manager.update_database(
            df=data,
            update_mode='upsert',
            merge_keys=merge_keys
        )
        
        # Log des statistiques
        logger.info(f"Update completed for {model_name}: {stats}")
        
        # Enregistrement dans la table d'historique
        self._track_model_update(model_name, model_version, stats, metadata)
        
        return stats
    
    def update_forecast_results(self,
                               forecasts: pd.DataFrame,
                               model_name: str,
                               horizon: int,
                               confidence_level: float = 0.95) -> dict:
        """
        Specialized method for forecast model results.
        
        Args:
            forecasts: DataFrame with forecast values
            model_name: Name of the forecast model
            horizon: Forecast horizon in periods
            confidence_level: Confidence level for intervals
            
        Returns:
            Update statistics
        """
        # Enrichissement des prévisions
        forecasts = forecasts.copy()
        forecasts['forecast_model'] = model_name
        forecasts['forecast_horizon'] = horizon
        forecasts['confidence_level'] = confidence_level
        forecasts['is_forecast'] = True
        forecasts['forecast_generated_at'] = datetime.now()
        
        # Merge keys pour les prévisions (typiquement date + série)
        merge_keys = ['date', 'indicator_id', 'country_code'] if 'country_code' in forecasts.columns else ['date', 'indicator_id']
        
        return self.manager.update_database(
            df=forecasts,
            update_mode='upsert',
            merge_keys=merge_keys
        )
    
    def update_simulation_results(self,
                                 simulations: pd.DataFrame,
                                 scenario_name: str,
                                 parameters: dict) -> dict:
        """
        Specialized method for simulation model results.
        
        Args:
            simulations: DataFrame with simulation results
            scenario_name: Name of the scenario
            parameters: Simulation parameters
            
        Returns:
            Update statistics
        """
        # Enrichissement des simulations
        simulations = simulations.copy()
        simulations['scenario_name'] = scenario_name
        simulations['simulation_timestamp'] = datetime.now()
        simulations['is_simulation'] = True
        
        # Ajout des paramètres comme colonnes
        for param_name, param_value in parameters.items():
            simulations[f'param_{param_name}'] = param_value
        
        # Pas de merge pour les simulations - on garde toutes les runs
        return self.manager.update_database(
            df=simulations,
            update_mode='append'
        )
    
    def _determine_merge_keys(self, data: pd.DataFrame, model_name: str) -> list:
        """
        Determine merge keys based on data structure and model type.
        
        Args:
            data: Input DataFrame
            model_name: Name of the model
            
        Returns:
            List of column names to use as merge keys
        """
        # Clés communes pour les séries temporelles
        time_keys = ['date', 'timestamp', 'period', 'time']
        entity_keys = ['indicator_id', 'country_code', 'region', 'entity_id']
        
        merge_keys = []
        
        # Recherche de clés temporelles
        for key in time_keys:
            if key in data.columns:
                merge_keys.append(key)
                break
        
        # Recherche de clés d'entité
        for key in entity_keys:
            if key in data.columns:
                merge_keys.append(key)
        
        # Si modèle spécifique, ajouter model_name comme clé
        if 'model_name' in data.columns:
            merge_keys.append('model_name')
        
        # Fallback si aucune clé trouvée
        if not merge_keys:
            logger.warning("No merge keys found, will append data")
            return None
        
        return merge_keys
    
    def _track_model_update(self, model_name: str, version: str, stats: dict, metadata: dict = None):
        """Track model update in history table."""
        try:
            # Création d'une entrée d'historique
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'model_version': version,
                'rows_added': stats.get('rows_added', 0),
                'rows_updated': stats.get('rows_updated', 0),
                'metadata': metadata
            }
            
            # Sauvegarde dans un fichier de log
            log_file = Path(self.database_path) / f"{self.database_name}_model_updates.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(history_entry) + '\n')
                
        except Exception as e:
            logger.warning(f"Could not track update history: {e}")
    
    def close(self):
        """Close database connection."""
        if self.manager:
            self.manager.close()


def main():
    """Main entry point for model updater."""
    parser = argparse.ArgumentParser(description='Update database with model results')
    
    # Arguments principaux
    parser.add_argument('--database-name', 
                       default=os.environ.get('DATABASE_NAME'),
                       required=True,
                       help='Target database name')
    
    parser.add_argument('--input-file',
                       default=os.environ.get('INPUT_FILE', '/data/input/model_output.csv'),
                       help='Input file with model results')
    
    parser.add_argument('--model-name',
                       default=os.environ.get('MODEL_NAME'),
                       required=True,
                       help='Name of the model')
    
    parser.add_argument('--model-version',
                       default=os.environ.get('MODEL_VERSION', 'latest'),
                       help='Model version')
    
    parser.add_argument('--model-type',
                       choices=['forecast', 'simulation', 'optimization', 'general'],
                       default='general',
                       help='Type of model')
    
    parser.add_argument('--metadata-file',
                       help='JSON file with additional metadata')
    
    # Arguments spécifiques aux prévisions
    parser.add_argument('--forecast-horizon',
                       type=int,
                       help='Forecast horizon (for forecast models)')
    
    parser.add_argument('--confidence-level',
                       type=float,
                       default=0.95,
                       help='Confidence level for forecast intervals')
    
    # Arguments spécifiques aux simulations
    parser.add_argument('--scenario-name',
                       help='Scenario name (for simulations)')
    
    parser.add_argument('--parameters-file',
                       help='JSON file with simulation parameters')
    
    args = parser.parse_args()
    
    try:
        # Chargement des données
        logger.info(f"Loading data from {args.input_file}")
        
        if args.input_file.endswith('.csv'):
            data = pd.read_csv(args.input_file)
        elif args.input_file.endswith('.parquet'):
            data = pd.read_parquet(args.input_file)
        elif args.input_file.endswith('.json'):
            data = pd.read_json(args.input_file)
        else:
            raise ValueError(f"Unsupported file format: {args.input_file}")
        
        logger.info(f"Loaded {len(data)} rows")
        
        # Chargement des métadonnées si fournies
        metadata = {}
        if args.metadata_file:
            with open(args.metadata_file) as f:
                metadata = json.load(f)
        
        # Initialisation de l'updater
        updater = ModelResultUpdater(
            database_name=args.database_name,
            database_path=os.environ.get('DATABASE_PATH', '/data/databases')
        )
        
        updater.connect()
        
        # Mise à jour selon le type de modèle
        if args.model_type == 'forecast':
            stats = updater.update_forecast_results(
                forecasts=data,
                model_name=args.model_name,
                horizon=args.forecast_horizon or 12,
                confidence_level=args.confidence_level
            )
            
        elif args.model_type == 'simulation':
            # Chargement des paramètres de simulation
            parameters = {}
            if args.parameters_file:
                with open(args.parameters_file) as f:
                    parameters = json.load(f)
            
            stats = updater.update_simulation_results(
                simulations=data,
                scenario_name=args.scenario_name or 'default',
                parameters=parameters
            )
            
        else:
            # Mise à jour générale
            stats = updater.update_from_model_output(
                data=data,
                model_name=args.model_name,
                model_version=args.model_version,
                metadata=metadata
            )
        
        # Affichage des résultats
        print(f"Update successful: {json.dumps(stats, indent=2)}")
        
        # Fermeture
        updater.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"Update failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())