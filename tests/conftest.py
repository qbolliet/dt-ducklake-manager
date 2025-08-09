# Importation des modules
# Modules de base
import os
import numpy as np
import pandas as pd
import logging
from io import BytesIO
# Module de connexion à S3
import boto3
import s3fs
#from moto import mock_aws
# Module de tests
import pytest

# Initialisation d'un jeu de données d'exemple
@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': range(1, 6),
        'category': ['A', 'B', 'A', 'C', 'B'],
        'value': np.random.rand(5),
        'date': pd.date_range('2024-01-01', periods=5),
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
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
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
    sample_df.to_csv(csv_path, index=False)
    files['csv'] = csv_path
    
    # Excel
    xlsx_path = tmp_path / "test.xlsx"
    sample_df.to_excel(xlsx_path, index=False, engine='openpyxl')
    files['xlsx'] = xlsx_path
    
    # Pickle
    pkl_path = tmp_path / "test.pkl"
    sample_df.to_pickle(pkl_path)
    files['pkl'] = pkl_path
    
    # Parquet
    parquet_path = tmp_path / "test.parquet"
    sample_df.to_parquet(parquet_path)
    files['parquet'] = parquet_path
    
    return files

# Fixture pour DataFrame avec doublons
@pytest.fixture
def sample_df_with_duplicates():
    """Create a sample DataFrame with duplicates for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 1, 2],  # Doublons sur id 1 et 2
        'category': ['A', 'B', 'A', 'A', 'B'],
        'status': ['active', 'inactive', 'active', 'active', 'inactive'],
        'value': [10.0, 20.0, 30.0, 40.0, 50.0]
    })

# Fixture pour DataFrame avec colonne 'value'
@pytest.fixture
def sample_df_with_value_column():
    """Create a sample DataFrame with a 'value' column for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 1],  # Doublons sur id 1
        'category': ['A', 'B', 'A', 'A'],
        'status': ['active', 'inactive', 'active', 'active'],
        'value': [10.0, 20.0, 30.0, 40.0]  # Valeurs différentes pour les doublons
    })