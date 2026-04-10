# Importation des modules
# Modules de base
import os
import warnings
import polars as pl
import logging
from io import BytesIO
from datetime import datetime, timedelta
# Module de connexion à S3
import boto3
import s3fs
#from moto import mock_aws
# Module de tests
import pytest
# Modules du package
from dashboard_template_database.schema import DuckLakeTablesBuilder

# Initialisation d'un jeu de données d'exemple
@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pl.DataFrame({
        'id': list(range(1, 6)),
        'category': ['A', 'B', 'A', 'C', 'B'],
        'value': [0.1, 0.2, 0.3, 0.4, 0.5],  # Valeurs fixes pour la reproductibilité
        'date': pl.date_range(datetime(2024, 1, 1), datetime(2024, 1, 5), '1d', eager=True),
        'status': ['active', 'inactive', 'active', 'active', 'inactive'],
        'high_cardinality': [f'val_{i}' for i in range(100, 105)]
    })

# Initialisation du dictionnaire de labels pour les colonnes
@pytest.fixture
def column_labels():
    """Fixture for DataFrame column labels."""
    return {
        'id': 'Identifier',
        'category': 'Category Name',
        'value': 'Numeric Value',
        'date': 'Date Field',
        'status': 'Status Field'
    }

# Initialisation du nom du bucket
@pytest.fixture
def s3_bucket():
    """Fixture for S3 bucket name."""
    return 'test-bucket'

# Initialisation des credentials de connexion
@pytest.fixture
def aws_credentials():
    """Set mock AWS credentials and endpoint for tests."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    os.environ['AWS_S3_ENDPOINT'] = 'localhost:5000'

# Initialisation d'une connexion s3
# @pytest.fixture
# def s3_client(aws_credentials):
#     """Set mock AWS boto3 client and endpoint for tests."""
#     with mock_aws():
#         yield boto3.client('s3', region_name='us-east-1')

# Initialisation d'un bucket avec des fichiers
@pytest.fixture
def setup_test_bucket(s3_client, s3_bucket, sample_df):
    """Set mock AWS bucket and files for tests."""
    # Création du bucket
    s3_client.create_bucket(Bucket=s3_bucket)
    
    # Création des fichiers de tests
    test_files = {
        'test.csv': sample_df.to_csv(index=False).encode(),
        'test.parquet': sample_df.to_parquet(),
        'test.xlsx': _create_excel_file(sample_df)
    }
    
    # Upload des fichiers
    for key, data in test_files.items():
        s3_client.put_object(Bucket=s3_bucket, Key=key, Body=data)
    
    return s3_bucket

# Fonction d'initialisation d'un client s3fs
# @pytest.fixture(autouse=True)
# def mock_s3fs(aws_credentials):
#     """Set mock AWS s3fsclient and endpoint for tests."""
#     with mock_aws():
#         fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'http://localhost:5000'})
#         yield fs

# Fonction auxliaire d'exportation d'un fichier excel
def _create_excel_file(df):
    """Create an in-memory Excel file from a DataFrame."""
    # Convertir polars en pandas pour l'export Excel
    import pandas as pd
    df_pd = df.to_pandas()
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_pd.to_excel(writer, index=False)
    output.seek(0)
    return output.getvalue()

# Fonction d'initalisation de logging
@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """Set logging for tests."""
    caplog.set_level(logging.INFO)

# Créer une nouvelle fixture pour les tests de fichiers temporaires
@pytest.fixture
def temp_files(sample_df, tmp_path):
    """Initialization of the files to load and save."""
    # Création de fichiers temporaires de différents formats
    files = {}

    # CSV
    csv_path = tmp_path / "test.csv"
    sample_df.write_csv(csv_path)
    files['csv'] = csv_path

    # Excel (convertir en pandas pour l'export)
    import pandas as pd
    xlsx_path = tmp_path / "test.xlsx"
    sample_df.to_pandas().to_excel(xlsx_path, index=False, engine='openpyxl')
    files['xlsx'] = xlsx_path

    # Parquet
    parquet_path = tmp_path / "test.parquet"
    sample_df.write_parquet(parquet_path)
    files['parquet'] = parquet_path

    return files

# Fixture pour DataFrame avec doublons
@pytest.fixture
def sample_df_with_duplicates():
    """Create a sample DataFrame with duplicates for testing.

    Rows 0&3 and rows 1&4 are fully identical (including 'value'), so they are
    detected as duplicates regardless of the column subset used for deduplication.
    """
    return pl.DataFrame({
        'id': [1, 2, 3, 1, 2],  # Doublons sur id 1 et 2
        'category': ['A', 'B', 'A', 'A', 'B'],
        'status': ['active', 'inactive', 'active', 'active', 'inactive'],
        'value': [10.0, 20.0, 30.0, 10.0, 20.0]  # Même valeur pour les lignes dupliquées
    })

# Fixture pour DataFrame avec colonne 'value'
@pytest.fixture
def sample_df_with_value_column():
    """Create a sample DataFrame with a 'value' column for testing."""
    return pl.DataFrame({
        'id': [1, 2, 3, 1],  # Doublons sur id 1
        'category': ['A', 'B', 'A', 'A'],
        'status': ['active', 'inactive', 'active', 'active'],
        'value': [10.0, 20.0, 30.0, 40.0]  # Valeurs différentes pour les doublons
    })

# Fixture pour DataFrame avec clés primaires et plusieurs colonnes de valeurs
@pytest.fixture
def sample_df_with_primary_keys():
    """Create a sample DataFrame with primary keys and multiple value columns for testing.

    Rows 0 and 3 are duplicates on 'id' but differ on the 'value' column,
    which allows testing that deduplication based on primary_keys correctly
    identifies them as duplicates regardless of the 'value' difference.
    """
    return pl.DataFrame({
        'id': [1, 2, 3, 1],
        'category': ['A', 'B', 'C', 'A'],
        'value': [10.0, 20.0, 30.0, 99.0],
        'lower_bound': [1.0, 2.0, 3.0, 1.0],
        'upper_bound': [5.0, 6.0, 7.0, 5.0],
    })

# Fixture fournissant une connexion DuckLake (in-memory) avec un schéma déjà construit
@pytest.fixture
def built_ducklake_schema(sample_df):
    """Provide an in-memory DuckDB connection with a fully built schema.

    The schema is built from sample_df with categorical_threshold=4 and
    primary_keys=['id'], ready to be used with operation managers.
    Uses an in-memory connection (no DuckLake catalog file) for test isolation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=['id'],
        )
    builder.build_schema()
    return builder.conn