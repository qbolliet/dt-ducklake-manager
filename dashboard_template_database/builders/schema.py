# Importation des modules
# Modules de base
import os
import warnings
import narwhals as nw
import polars as pl
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
# Module d'initialisation du logger
from ..utils.logger import _init_logger
# Utilitaires de traitement des données
from ..utils.data_processing import map_python_to_sql_type

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Classe de création d'une base de données DuckDB avec :
# - Une "Fact table" : Contenant les données
# - Une "Label table" : Contenant les labels associés à chaque indicateur
# - Une "Metadata table" : Contenant les caractéristiques des variables de la factable (type, modalités etc ...)
class SchemaBuilder:
    """
    A class to automate the creation of metadata, dimension tables, and fact tables
    from a given DataFrame (pandas, polars, or narwhals-compatible), primarily for use in data warehousing.

    Attributes:
        df (nw.DataFrame): The input dataset (converted to narwhals).
        categorical_threshold (int): Threshold for determining if a column is categorical
                                     based on the number of unique modalities.
        logger (logging.Logger): Logger instance for tracking processing steps.
    """

    # Initialisation
    def __init__(self, df, categorical_threshold: Optional[int] = None, primary_keys: Optional[List[str]] = None, log_filename: Optional[os.PathLike] = None) -> None:
        """
        Initialize the SchemaBuilder with a DataFrame and optional parameters.

        Args:
            df: The input dataset (pandas, polars, or narwhals-compatible).
            categorical_threshold (Optional[int]): Maximum number of unique values
                for a column to be considered categorical. When None (default), no
                dimension tables are created and all object columns are marked as
                non-categorical. Pass an integer (e.g. 50) to restore the previous
                behaviour.
            primary_keys (List[str], optional): List of column names to use as primary key.
                Can be a single column or composite key. Defaults to None (no primary key).
                When None, deduplication will use all columns and a UserWarning is raised.
            log_filename (os.PathLike, optional): Path to the log file. Defaults to
                a file named `schema_builder.log` in a logs directory.

        Examples:
            >>> # Single primary key (with polars)
            >>> import polars as pl
            >>> df = pl.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
            >>> builder = SchemaBuilder(df, primary_keys=['id'])

            >>> # Composite primary key
            >>> builder = SchemaBuilder(df, primary_keys=['date', 'country', 'indicator'])

            >>> # No primary key — raises UserWarning
            >>> builder = SchemaBuilder(df)

            >>> # Threshold disabled — no dimension tables are created
            >>> builder = SchemaBuilder(df, categorical_threshold=None, primary_keys=['id'])

            >>> # Threshold enabled — behaves like the original default
            >>> builder = SchemaBuilder(df, categorical_threshold=50, primary_keys=['id'])
        """
        # Conversion vers narwhals
        self.df = nw.from_native(df, eager_only=True)
        # Initialisation du seuil au deçà duquel les modalités d'une variable catégorielle ne sont plus exportées dans
        self.categorical_threshold = categorical_threshold

        # Validation des clés primaires si spécifiées
        if primary_keys is not None and len(primary_keys) > 0:
            # Vérification de l'existence des colonnes
            missing_cols = set(primary_keys) - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"The following primary key columns do not exist in the DataFrame: {missing_cols}"
                )

            self.primary_keys = primary_keys
        else:
            self.primary_keys = []

        # Avertissement en l'absence de clés primaires : la déduplication portera sur
        # l'ensemble des colonnes, ce qui peut produire des résultats inattendus lorsque
        # plusieurs colonnes de valeurs (value, lower_bound, upper_bound…) diffèrent
        # entre deux lignes qui représentent pourtant le même enregistrement.
        if not self.primary_keys:
            warnings.warn(
                "No primary key specified. The deduplication will apply to "
                "all columns. Pass primary_keys=['col1', ...] to "
                "avoid this warning.",
                UserWarning,
                stacklevel=2,
            )

        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/schema_builder.log")
        self.logger = _init_logger(filename=log_filename)

        # Logging de la validation des clés primaires (après initialisation du logger)
        if len(self.primary_keys) > 0:
            self.logger.info(f"Primary keys validated successfully: {self.primary_keys}")
    
    # Méthode inférant le type des colonnes du jeu de données
    def create_metadata_table(self, column_labels : Optional[Union[Dict[str, str], None]]= None) -> nw.DataFrame:
        """
        Automatically infer metadata for the DataFrame's columns, including types, labels,
        and categorical attributes.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.

        Returns:
            nw.DataFrame: A DataFrame containing metadata for each column in the input dataset.
        """
        # Initialisation de la liste des méta-données
        list_metadata = []
        # Parcours des colonnes du jeu de données
        for col in self.df.columns:
            # Extraction du type de la colonne (narwhals DType)
            dtype_obj = self.df.schema[col]
            dtype_str = str(dtype_obj)
            # Initialisation des méta-données associées à la colonne
            if column_labels is not None :
                metadata = {
                    'name': col,
                    'label': column_labels[col] if col in column_labels.keys() else col.replace('_', ' ').title(),
                    'python_type': dtype_str,
                    'sql_type': map_python_to_sql_type(dtype_obj),
                    'is_categorical': False,
                    'is_primary_key': col in self.primary_keys,
                    # 'modalities': None
                }
            else :
                metadata = {
                    'name': col,
                    'label': col.replace('_', ' ').title(),
                    'python_type': dtype_str,
                    'sql_type': map_python_to_sql_type(dtype_obj),
                    'is_categorical': False,
                    'is_primary_key': col in self.primary_keys,
                    # 'modalities': None
                }

            # Logging
            self.logger.info(f"Successfully extracted meta-data from column '{col}'")

            # Vérification si la colonne est de type String (équivalent narwhals de 'object')
            if isinstance(dtype_obj, nw.String):
                # Calcul du nombre de modalités
                n_modalities = self.df[col].n_unique()
                # Vérification du seuil : si categorical_threshold vaut None, aucune colonne
                # n'est traitée comme catégorielle, quel que soit son nombre de modalités.
                if self.categorical_threshold is not None :
                    if n_modalities <= self.categorical_threshold:
                        # Mise à jour du type de la variable
                        metadata['is_categorical'] = True
                        # Logging
                        self.logger.info(f"The column '{col}' is of type 'String' and the number of modalities {n_modalities} satisfies the categorical threshold criteria {self.categorical_threshold}")
                        # Mise à jour des modalités
                        # metadata['modalities'] = str(self.df[col].dropna().unique().tolist())
                    else :
                        # Logging
                        self.logger.warning(f"The column '{col}' is of type 'String' but the number of modalities {n_modalities} exceeds the categorical threshold criteria {self.categorical_threshold}")

            # Ajout au dictionnaire
            list_metadata.append(metadata)

        # Création d'un DataFrame avec polars
        df_metadata_pl = pl.DataFrame(list_metadata).sort('label')
        # Conversion vers narwhals
        self.df_metadata = nw.from_native(df_metadata_pl, eager_only=True)

        # Logging
        self.logger.info("Successfully built the meta-data DataFrame")

        return self.df_metadata



    # Méthode créant la dimension table
    def create_dimension_tables(self, column_labels : Optional[Union[Dict[str, str], None]]= None) -> Dict[str, nw.DataFrame] :
        """
        Generate dimension tables for categorical columns in the dataset.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.

        Returns:
            dict: A dictionary of DataFrames, where keys are column names and values are
                  the corresponding dimension tables (narwhals DataFrames).
        """
        # Création d'une table de méta-données si cette-dernière n'existe pas déjà
        if not hasattr(self, 'df_metadata') :
            _ = self.create_metadata_table(column_labels=column_labels)

        # Initialisation du dictionnaire des tables de dimension
        self.dimension_tables = {}

        # Parcours des tables de dimensions
        categorical_cols = self.df_metadata.filter(nw.col('is_categorical'))['name'].to_list()
        for categorical_dimension in categorical_cols :
            # Extraction des modalités uniques et construction de la table de dimension
            unique_vals = self.df[categorical_dimension].unique().sort()
            unique_list = unique_vals.to_list()

            # Création de la table de dimension avec polars
            dim_df_pl = pl.DataFrame({
                'value': list(range(len(unique_list))),
                'label': unique_list
            }).sort('label')

            # Conversion vers narwhals
            self.dimension_tables[categorical_dimension] = nw.from_native(dim_df_pl, eager_only=True)
            # Logging
            self.logger.info(f"Successfully built dimension table for '{categorical_dimension}'")

        # Logging
        self.logger.info("Successfully built dimension tables")

        return self.dimension_tables

    # Méthode créant la table des informations
    def create_fact_table(self, column_labels : Optional[Union[Dict[str, str], None]]= None) -> nw.DataFrame:
        """
        Generate a fact table by replacing categorical values with corresponding IDs.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.

        Returns:
            nw.DataFrame: The fact table with categorical values replaced by IDs.
        """
        # Création des tables de dimensions si ces-dernières n'existent pas
        if not hasattr(self, 'dimension_tables') :
            _ = self.create_dimension_tables(column_labels=column_labels)

        # Initialisation de la table des informations
        self.df_fact = self.df.clone()

        # Remplacement des labels par leur valeur
        for column in self.dimension_tables.keys() :
            # Conversion de la table de dimension en polars natif
            dim_df_pl = nw.to_native(self.dimension_tables[column])
            fact_df_pl = nw.to_native(self.df_fact)

            # Créer un dictionnaire de mapping label -> value
            mapping_dict = dict(zip(dim_df_pl['label'].to_list(), dim_df_pl['value'].to_list()))

            # Utiliser replace de polars pour remplacer les labels par les IDs
            fact_df_pl = fact_df_pl.with_columns(
                pl.col(column).replace(mapping_dict)
            )

            # Reconversion vers narwhals
            self.df_fact = nw.from_native(fact_df_pl, eager_only=True)
            # Logging
            self.logger.info(f"Successfully replace modalities by ids in column '{column}'")

        # Logging
        self.logger.info("Successfully built fact table")

        return self.df_fact


    # Méthode créant les différentes tables
    def build(self, column_labels : Optional[Union[Dict[str, str], None]]= None) -> Tuple[nw.DataFrame, Dict[str, nw.DataFrame], nw.DataFrame] :
        """
        Execute the full pipeline to create metadata, dimension tables, and a fact table.

        Args:
            column_labels (dict, optional): A dictionary mapping column names to labels.
                                            Defaults to None.

        Returns:
            tuple: A tuple containing:
                   - Metadata DataFrame (narwhals).
                   - Dictionary of dimension tables (narwhals DataFrames).
                   - Fact table DataFrame (narwhals).
        """
        # Création de la table des méta-données
        _ = self.create_metadata_table(column_labels=column_labels)
        # Création des tables de dimension
        _ = self.create_dimension_tables(column_labels=column_labels)
        # Création des tables d'informations
        _ = self.create_fact_table(column_labels=column_labels)

        return self.df_metadata, self.dimension_tables, self.df_fact