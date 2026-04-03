# Importation des modules
# Modules de base
import os
from pathlib import Path
from typing import Optional

# DuckDB
import duckdb

# Module d'initialisation du logger
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe de connexion à un catalogue DuckLake
class DuckLakeConnector:
    """
    Creates and configures a DuckDB connection attached to a DuckLake catalog.

    DuckLake stores all table metadata in a catalog file (a small DuckDB or SQLite
    database) while row data lives in immutable Parquet files under ``data_path``.
    This class handles installing the ``ducklake`` extension, attaching the catalog,
    and routing the session to the correct schema.

    After calling ``connect()``, the returned ``duckdb.DuckDBPyConnection`` can be
    passed directly to ``DuckdbTablesBuilder``, ``DatabaseUpdater``, or any other
    class that accepts a ``connection`` parameter.

    Attributes:
        catalog_path (str): Path to the ``.ducklake`` catalog file.
        data_path (str): Directory where Parquet data files are stored.
        read_only (bool): Whether the connection is read-only.
        catalog_alias (str): Alias used in the ``ATTACH`` statement (default ``'db'``).
        schema (str): DuckLake schema to activate with ``USE`` (default ``'main'``).
        logger (logging.Logger): Logger instance.

    Examples:
        >>> # Connexion en lecture-écriture
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
        >>> # Connexion en lecture seule (API GraphQL, dashboard)
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/', read_only=True).connect()
        >>> # Time travel vers un snapshot spécifique (audit d'un run ML)
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/', snapshot_version=3).connect()
    """

    # Initialisation
    def __init__(
        self,
        catalog_path: str,
        data_path: str,
        read_only: bool = False,
        snapshot_version: Optional[int] = None,
        snapshot_time: Optional[str] = None,
        catalog_alias: str = 'db',
        schema: str = 'main',
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[2], "logs/ducklake_connector.log"
        ),
    ) -> None:
        """
        Initialize the DuckLakeConnector.

        Args:
            catalog_path (str): Path to the DuckLake catalog file (e.g.
                ``'data/catalog.ducklake'``).
            data_path (str): Directory where Parquet data files are stored
                (e.g. ``'data/files/'``).
            read_only (bool): Open the catalog in read-only mode. Useful for
                dashboard/API layers that must not write. Defaults to False.
            snapshot_version (Optional[int]): Open a historical snapshot by
                version number. Implies ``READ_ONLY``. Defaults to None.
            snapshot_time (Optional[str]): Open a historical snapshot by
                timestamp (ISO-8601 string, e.g. ``'2025-01-01 00:00:00'``).
                Implies ``READ_ONLY``. Defaults to None.
            catalog_alias (str): Alias for the attached DuckLake catalog in SQL
                (e.g. ``USE {alias}.{schema}``). Defaults to ``'db'``.
            schema (str): DuckLake schema to activate. Defaults to ``'main'``.
            log_filename (Optional[os.PathLike]): Path to the log file.

        Examples:
            >>> connector = DuckLakeConnector('catalog.ducklake', 'data/')
            >>> conn = connector.connect()
        """
        # Stockage des paramètres de connexion
        self.catalog_path = str(catalog_path)
        self.data_path = str(data_path)
        # snapshot_version et snapshot_time impliquent un accès en lecture seule :
        # DuckLake ouvre automatiquement le catalogue en READ_ONLY dans ce cas.
        # L'attribut self.read_only reflète cet état effectif pour cohérence.
        self.read_only = read_only or snapshot_version is not None or snapshot_time is not None
        self.snapshot_version = snapshot_version
        self.snapshot_time = snapshot_time
        self.catalog_alias = catalog_alias
        self.schema = schema

        # Initialisation du logger
        self.logger = _init_logger(filename=log_filename)

    # ---------------------------------------------------------------------------
    # Méthodes publiques de connexion
    # ---------------------------------------------------------------------------

    # Création d'une nouvelle connexion DuckDB attachée au catalogue DuckLake
    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Create and return a DuckDB connection attached to the DuckLake catalog.

        Steps performed:
        1. Open an in-memory DuckDB connection.
        2. Install and load the ``ducklake`` extension.
        3. Build and execute the ``ATTACH`` statement with appropriate options.
        4. Activate the target schema with ``USE``.

        Returns:
            duckdb.DuckDBPyConnection: Configured DuckDB connection ready for use.

        Raises:
            duckdb.Error: If the extension cannot be loaded or the catalog cannot
                be attached.

        Examples:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> conn = DuckLakeConnector(
            ...     'catalog.ducklake', 'data/',
            ...     read_only=True,
            ... ).connect()
        """
        # Ouverture d'une connexion DuckDB en mémoire
        # DuckLake utilise toujours :memory: comme connexion de base car le catalogue
        # est géré séparément dans le fichier .ducklake.
        conn = duckdb.connect(':memory:')

        # Installation et chargement de l'extension DuckLake
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        self.logger.info("DuckLake extension loaded")

        # Construction de la chaîne d'options ATTACH
        attach_sql = self._build_attach_sql()
        conn.execute(attach_sql)
        self.logger.info(
            f"DuckLake catalog attached : '{self.catalog_path}' "
            f"(alias={self.catalog_alias}, read_only={self.read_only})"
        )

        # Activation du schéma cible pour que les requêtes non qualifiées fonctionnent
        conn.execute(f"USE {self.catalog_alias}.{self.schema}")
        self.logger.info(f"Activated scheme : {self.catalog_alias}.{self.schema}")

        return conn

    # Attachement d'un catalogue DuckLake à une connexion DuckDB existante
    def attach(self, conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
        """
        Attach the DuckLake catalog to an already-open DuckDB connection.

        Useful when the extension is already loaded or when sharing a connection
        across multiple catalogs. The ``USE`` statement is also executed to
        activate the configured schema.

        Args:
            conn (duckdb.DuckDBPyConnection): Existing DuckDB connection.

        Returns:
            duckdb.DuckDBPyConnection: The same connection, now with the catalog
            attached and the schema activated.

        Examples:
            >>> import duckdb
            >>> conn = duckdb.connect(':memory:')
            >>> conn.execute("LOAD ducklake;")
            >>> connector = DuckLakeConnector('catalog.ducklake', 'data/')
            >>> connector.attach(conn)
        """
        # Attachement du catalogue sur une connexion existante (sans réinstaller l'extension)
        attach_sql = self._build_attach_sql()
        conn.execute(attach_sql)
        conn.execute(f"USE {self.catalog_alias}.{self.schema}")
        self.logger.info(
            f"DuckLake catalog attached to the existing connection: '{self.catalog_path}'"
        )
        return conn

    # ---------------------------------------------------------------------------
    # Méthode privée de construction de la requête ATTACH
    # ---------------------------------------------------------------------------

    # Construction de la clause SQL ATTACH avec les options appropriées
    def _build_attach_sql(self) -> str:
        """
        Build the ``ATTACH`` SQL statement from the connector configuration.

        The option string is built incrementally:
        - ``DATA_PATH`` is always included.
        - ``READ_ONLY`` is appended for read-only or time-travel connections.
        - ``SNAPSHOT_VERSION`` or ``SNAPSHOT_TIME`` is appended for time travel.

        Returns:
            str: The complete ``ATTACH`` SQL statement.
        """
        # Liste des options ATTACH à construire
        options = [f"DATA_PATH '{self.data_path}'"]

        # Ajout de l'option SNAPSHOT_VERSION si une version est spécifiée
        # (implique READ_ONLY automatiquement selon la spec DuckLake)
        if self.snapshot_version is not None:
            options.append(f"SNAPSHOT_VERSION {self.snapshot_version}")
        # Ajout de l'option SNAPSHOT_TIME si un timestamp est spécifié
        elif self.snapshot_time is not None:
            options.append(f"SNAPSHOT_TIME '{self.snapshot_time}'")
        # Ajout de READ_ONLY si demandé explicitement (hors time travel)
        elif self.read_only:
            options.append("READ_ONLY")

        # Assemblage de la requête ATTACH finale
        options_str = ", ".join(options)
        return (
            f"ATTACH 'ducklake:{self.catalog_path}' AS {self.catalog_alias} ({options_str})"
        )
