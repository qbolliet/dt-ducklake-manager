# Importation des modules
# Modules de base
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
# Module d'initialisation du logger
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Classe de création d'une base de données DuckDB avec :
# - Une "Fact table" : Contenant les données
# - Une "Label table" : Contenant les labels associés à chaque indicateur
# - Une "Metadata table" : Contenant les caractéristiques des variables de la factable (type, modalités etc ...)
class SchemaBuilder:
    """
    A class to automate the creation of metadata, dimension tables, and fact tables 
    from a given pandas DataFrame, primarily for use in data warehousing.

    Attributes:
        df (pd.DataFrame): The input dataset.
        categorical_threshold (int): Threshold for determining if a column is categorical 
                                     based on the number of unique modalities.
        logger (logging.Logger): Logger instance for tracking processing steps.
    """

    # Initialisation
    def __init__(self, df: pd.DataFrame, categorical_threshold: Optional[int] = 50, log_filename: Optional[os.PathLike] = None) -> None:
        """
        Initialize the SchemaBuilder with a DataFrame and optional parameters.

        Args:
            df (pd.DataFrame): The input dataset.
            categorical_threshold (int, optional): Maximum number of unique values 
                                                   for a column to be considered categorical. Defaults to 50.
            log_filename (os.PathLike, optional): Path to the log file. Defaults to 
                                                  a file named `schema_builder.log` in a logs directory.
        """
        # Initialisation des arguments
        # Jeu de données
        self.df = df
        # Initialisation du seuil au deçà duquel les modalités d'une variable catégorielle ne sont plus exportées dans 
        self.categorical_threshold = categorical_threshold
        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/schema_builder.log")
        self.logger = _init_logger(filename=log_filename)
    
    # Méthode inférant le type des colonnes du jeu de données 
    def create_metadata_table(self, column_labels : Optional[Union[Dict[str, str], None]]= None) -> Dict:
        """
        Automatically infer metadata for the DataFrame's columns, including types, labels, 
        and categorical attributes.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing metadata for each column in the input dataset.
        """
        # Initialisation de la liste des méta-données
        list_metadata = []
        # Parcours des colonnes du jeu de données
        for col in self.df.columns:
            # Extraction du type de la colonne
            dtype = str(self.df[col].dtype)
            # Initialisation des méta-données associées à la colonne
            if column_labels is not None :
                metadata = {
                    'name': col,
                    'label': column_labels[col] if col in column_labels.keys() else col.replace('_', ' ').title(),
                    'python_type': dtype,
                    'sql_type': self._map_python_to_sql_type(dtype),
                    'is_categorical': False,
                    # 'modalities': None
                }
            else :
                metadata = {
                    'name': col,
                    'label': col.replace('_', ' ').title(),
                    'python_type': dtype,
                    'sql_type': self._map_python_to_sql_type(dtype),
                    'is_categorical': False,
                    # 'modalities': None
                }
            
            # Logging
            self.logger.info(f"Successfully extracted meta-data from column '{col}'")

            # Si la colonne est object
            if dtype == 'object' :
                # Calcul du nombre de modalités
                n_modalities = self.df[col].nunique()
                # Si le nombre de modalités dans la colonne est inférieur au seuil, la variable est catégorielle
                if  n_modalities <= self.categorical_threshold:
                    # Mise à jour du type de la variable
                    metadata['is_categorical'] = True
                    # Logging
                    self.logger.info(f"The column '{col}' is of type 'object' and the number of modalities {n_modalities} satisfies the categorical threshold criteria {self.categorical_threshold}")
                    # Mise à jour des modalités
                    # metadata['modalities'] = str(self.df[col].dropna().unique().tolist())
                else :
                    # Logging
                    self.logger.warning(f"The column '{col}' is  of type 'object' but the number of modalities {n_modalities} exceeds the categorical threshold criteria {self.categorical_threshold}")
            
            # Ajout au dictionnaire
            list_metadata.append(metadata)            
        
        # Création d'un DataFrame
        self.df_metadata = pd.DataFrame.from_dict(list_metadata).sort_values(by='label', ascending=True, ignore_index=True)

        # Logging
        self.logger.info("Successfully built the meta-data DataFrame")
        
        return self.df_metadata

    # Correspondance entre les types "python" et "SQL"
    @staticmethod
    def _map_python_to_sql_type(dtype: str) -> str:
        """
        Map Python data types to SQL-compatible data types.

        Args:
            dtype (str): The Python data type as a string.

        Returns:
            str: The corresponding SQL data type.
        """
        # /!\ Peut être mis dans un json de paramètres
        # Dictionnaire des correspondance entre les types Python et SQL
        type_mapping = {
            'object': 'VARCHAR',
            'int64': 'INTEGER',
            'float64': 'DOUBLE',
            'datetime64[ns]': 'TIMESTAMP',
            'bool': 'BOOLEAN'
        }
        return type_mapping.get(dtype, 'VARCHAR')
    
    # Méthode créant la dimension table
    def create_dimension_tables(self, column_labels : Optional[Union[Dict[str, str], None]]= None) -> Dict[str, pd.DataFrame] :
        """
        Generate dimension tables for categorical columns in the dataset.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.

        Returns:
            dict: A dictionary of DataFrames, where keys are column names and values are 
                  the corresponding dimension tables.
        """
        # Création d'une table de méta-données si cette-dernière n'existe pas déjà
        if not hasattr(self, 'metadata') :
            _ = self.create_metadata_table(column_labels=column_labels)

        # Initialisation du dictionnaire des tables de dimension
        self.dimension_tables = {}

        # Parcours des tables de dimensions
        for categorical_dimension in self.df_metadata.loc[self.df_metadata['is_categorical'], 'name'] :
            # Extraction des modalités
            self.dimension_tables[categorical_dimension] = pd.Series(self.df[categorical_dimension].unique(), name='label').to_frame().reset_index(names='value').sort_values(by='label', ascending=True, ignore_index=True)
            # Logging
            self.logger.info(f"Successfully built dimension table for '{categorical_dimension}'")

        # Logging
        self.logger.info("Successfully built dimension tables")

        return self.dimension_tables

    # Méthode créant la table des informations
    def create_fact_table(self, column_labels : Optional[Union[Dict[str, str], None]]= None) -> pd.DataFrame:
        """
        Generate a fact table by replacing categorical values with corresponding IDs.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.

        Returns:
            pd.DataFrame: The fact table with categorical values replaced by IDs.
        """
        # Création des tables de dimensions si ces-dernières n'existent pas
        if not hasattr(self, 'dimension_tables') :
            _ = self.create_dimension_tables(column_labels=column_labels)
        
        # Initialisation de la table des informations
        self.df_fact = self.df.copy()

        # Remplacement des labels par leur valeur
        for column in self.dimension_tables.keys() :
            # Construction du dictionnaire de passage
            dict_label_value = {d['label'] : d['value'] for d in self.dimension_tables[column].to_dict(orient='records')}
            # Remplacement des valeurs
            self.df_fact[column] = self.df_fact[column].replace(dict_label_value)
            # Logging
            self.logger.info(f"Successfully replace modalities by ids in column '{column}'")
        
        # Logging
        self.logger.info("Successfully built fact table")
        
        return self.df_fact
    
    # Méthode créant les différentes tables
    def build(self, column_labels : Optional[Union[Dict[str, str], None]]= None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame] :
        """
        Execute the full pipeline to create metadata, dimension tables, and a fact table.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.

        Returns:
            tuple: A tuple containing:
                   - Metadata DataFrame.
                   - Dictionary of dimension tables.
                   - Fact table DataFrame.
        """
        # Création de la table des méta-données
        _ = self.create_metadata_table(column_labels=column_labels)
        # Création des tables de dimension
        _ = self.create_dimension_tables(column_labels=column_labels)
        # Création des tables d'informations
        _ = self.create_fact_table(column_labels=column_labels)

        return self.df_metadata, self.dimension_tables, self.df_fact