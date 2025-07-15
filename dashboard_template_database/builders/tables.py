# Importation des modules
# Modules de base
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
# Duckdb
import duckdb

# Modules ad hoc
from .schema import SchemaBuilder

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe créant les tables correspondant au schéma
class DuckdbTablesBuilder(SchemaBuilder):
    """
    A class to build and manage schema tables in DuckDB.

    This class extends `SchemaBuilder` and provides functionality to create 
    metadata, dimension, and fact tables in DuckDB, while logging all operations.

    Attributes:
        conn (duckdb.DuckDBPyConnection): DuckDB connection object.
    """

    # Initialisation
    def __init__(self, df: pd.DataFrame, categorical_threshold: Optional[int] = 50, connection: Optional[duckdb.DuckDBPyConnection] = None, path : Optional[Union[os.PathLike, None]]=None, log_filename: Optional[os.PathLike] = os.path.join(FILE_PATH.parents[2], "logs/duckdb_schema_builder.log")):
        """
        Initialize the DuckdbTablesBuilder class.

        Args:
            df (pd.DataFrame): The input dataset.
            categorical_threshold (Optional[int]): Threshold for determining categorical variables.
            connection (Optional[duckdb.DuckDBPyConnection]): Existing DuckDB connection. Defaults to None.
            path (Optional[Union[os.PathLike, None]]): Path to the DuckDB database file. Defaults to None.
            log_filename (Optional[os.PathLike]): Path to the log file. Defaults to a pre-defined path.

        """
        # Initialisation du schéma
        super().__init__(df=df, categorical_threshold=categorical_threshold, log_filename=log_filename)
        
        # Initialisation de la connection
        if (connection is None) & (path is None) :
            self.conn = duckdb.connect(':memory:')
        elif (connection is None) :
            self.conn = duckdb.connect(path)
        else :
            self.conn = connection
    
    # Méthode de création de la table des méta-données
    def create_duckdb_metadata_table(self, table_name: Optional[str] = 'metadata', column_labels: Optional[Dict[str, str]] = None) -> None:
        """
        Create a metadata table in DuckDB.

        Args:
            table_name (Optional[str]): Name of the metadata table in DuckDB. Defaults to 'metadata'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names to labels.

        """
        # Création de la table des méta-données si elle n'existe pas déjà
        if not hasattr(self, 'df_metadata'):
            _ = self.create_metadata_table(column_labels)
        
        # Conversion DataFrame en table DuckDB
        self.conn.register('temp_metadata', self.df_metadata)
        self.conn.execute(f"""
            CREATE TABLE {table_name} AS 
            SELECT * FROM temp_metadata
        """)
        self.conn.execute('DROP VIEW temp_metadata')

        # Logging
        self.logger.info("Successfully registered duckdb meta-data table")
    
    # Méthode de création des tables de dimensions
    def create_duckdb_dimension_tables(self, table_prefix: Optional[str] = 'dim_', column_labels: Optional[Dict[str, str]] = None) -> None:
        """
        Create dimension tables in DuckDB for categorical variables.

        Args:
            table_prefix (Optional[str]): Prefix for dimension table names. Defaults to 'dim_'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names to labels.

        """
        # Création du dictionnaire des tables de dimensions si elles n'existent pas déjà
        if not hasattr(self, 'dimension_tables'):
            _ = self.create_dimension_tables(column_labels)
        
        # Création de chaque table de dimension dans DuckDB
        for dim_name, dim_df in self.dimension_tables.items():
            # Initialisation du nom de la table
            table_name = f"{table_prefix}{dim_name}"
            # Enregistrement d'une vue temporaire
            self.conn.register('temp_dim', dim_df)
            
            # Création d'une table avec "value" comme clé primaire
            self.conn.execute(f"""
                CREATE TABLE {table_name} AS 
                SELECT
                    value,
                    label 
                FROM temp_dim
            """)
            
            # Ajout de la vue correspondante
            self.conn.execute('DROP VIEW temp_dim')

            # Logging
            self.logger.info(f"Successfully registered duckdb dimension table for {dim_name}")
    
    # Méthode de création de la table d'informations
    def create_duckdb_fact_table(self, table_name: Optional[str] = 'fact_table', table_prefix: Optional[str] = 'dim_', column_labels: Optional[Dict[str, str]] = None) -> None:
        """
        Create a fact table in DuckDB with foreign key relationships to dimension tables.

        Args:
            table_name (Optional[str]): Name of the fact table in DuckDB. Defaults to 'fact_table'.
            table_prefix (Optional[str]): Prefix for dimension table names. Defaults to 'dim_'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names to labels.

        """
        # Création de la table d'informations si elle n'existe pas déjà
        if not hasattr(self, 'df_fact'):
            _ = self.create_fact_table(column_labels)
        
        # Enregistrement de la table comme vue temporaire
        self.conn.register('temp_fact', self.df_fact)
        
        # Création de la table d'information avec seulement les colonnes originales
        query = f"""
        CREATE TABLE {table_name} AS 
        SELECT {', '.join(self.df_fact.columns)}
        FROM temp_fact
        """
        
        self.conn.execute(query)

        # Suppression de la vue temporaire
        self.conn.execute('DROP VIEW temp_fact')

        # Logging des clés étrangères créées (pour information)
        for dim_name in self.dimension_tables.keys():
            self.logger.info(f"Foreign key created for dimension '{dim_name}' - can be joined on {table_name}.{dim_name} = {table_prefix}{dim_name}.value")

        # Logging
        self.logger.info(f"Successfully registered duckdb fact table")
    
    # Méthode de construction du schéma
    def build_duckdb_schema(self, metadata_table: Optional[str] = 'metadata', fact_table: Optional[str] = 'fact_table', dim_table_prefix: Optional[str] = 'dim_', column_labels: Optional[Dict[str, str]] = None) -> None:
        """
        Build the entire schema in DuckDB, including metadata, dimension, and fact tables.

        Args:
            metadata_table (Optional[str]): Name of the metadata table. Defaults to 'metadata'.
            fact_table (Optional[str]): Name of the fact table. Defaults to 'fact_table'.
            dim_table_prefix (Optional[str]): Prefix for dimension tables. Defaults to 'dim_'.
            column_labels (Optional[Dict[str, str]]): Optional mapping of column names to labels.

        """
        # Création de la table des méta-données
        self.create_duckdb_metadata_table(
            table_name=metadata_table, 
            column_labels=column_labels
        )
        
        # Création de la table de dimensions
        self.create_duckdb_dimension_tables(
            table_prefix=dim_table_prefix, 
            column_labels=column_labels
        )
        
        # Création de la table d'informations avec les clés étrangères
        self.create_duckdb_fact_table(
            table_name=fact_table, 
            table_prefix=dim_table_prefix, 
            column_labels=column_labels
        )
    
    # Méthode d'affichage du schéma
    def display_schema(self) -> None:
        """
        Display the structure of all tables in the DuckDB schema.
        """
        # Extraction des tables
        tables = self.conn.execute("SHOW TABLES").fetchall()
        # Logging
        self.logger.info("\n Created Tables:")
        # Parcours des tables
        for table in tables:
            # Affichage de la structure
            self.logger.info(f"\n {table[0]} Structure:")
            # Extraction des informations relatives à la table
            table_info = self.conn.execute(f"DESCRIBE {table[0]}").fetchall()
            # Affichage de chaque information
            for col in table_info:
                self.logger.info(f"  {col[0]}: {col[1]}")