# Importation des modules
# Modules de base
import os
from enum import StrEnum
from pathlib import Path

# DuckDB
import duckdb

# Module d'initialisation du logger
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Énumération des backends de catalogue supportés par DuckLake
class CatalogType(StrEnum):
    """Supported DuckLake catalog backends.

    DuckLake stores its catalog metadata in a backing database whose type is
    selected by the prefix of the ``ATTACH`` target string:

    - ``DUCKDB``: a local ``.ducklake`` file (``ducklake:<path>``). Default,
      single-process; the file is locked at the process level, so it cannot be
      read by one program while another writes it.
    - ``SQLITE``: a local SQLite file (``ducklake:sqlite:<path>``).
    - ``POSTGRES``: a PostgreSQL server (``ducklake:postgres:<conn>``). A true
      multi-client backend: a read-only consumer (e.g. a GraphQL API) can query
      the catalog while another process updates it, without lock conflicts.

    A ``StrEnum`` so the members compare equal to their plain string values
    (e.g. ``CatalogType.POSTGRES == "postgres"``).
    """

    DUCKDB = "duckdb"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


# Échappement d'une valeur destinée à un littéral de chaîne SQL
def _quote_literal(value: str) -> str:
    """Escape single quotes for safe inclusion inside a SQL string literal.

    The value is meant to be wrapped by single quotes by the caller; this helper
    only doubles any embedded single quote, following the SQL standard.

    Args:
        value (str): Raw value to escape (e.g. a password or hostname).

    Returns:
        str: The escaped value, without surrounding quotes.

    Examples:
        >>> _quote_literal("pa'ss")
        "pa''ss"
        >>> f"PASSWORD '{_quote_literal(\"pa'ss\")}'"
        "PASSWORD 'pa''ss'"
    """
    # Doublage des apostrophes : seul caractère à neutraliser dans un littéral SQL
    return value.replace("'", "''")


# Classe de connexion à un catalogue DuckLake
class DuckLakeConnector:
    """
    Creates and configures a DuckDB connection attached to a DuckLake catalog.

    DuckLake stores all table metadata in a catalog backend while row data lives in
    immutable Parquet files under ``data_path``. The catalog backend can be a local
    file (DuckDB ``.ducklake`` or SQLite) or a PostgreSQL server, selected via
    ``catalog_type``. This class handles installing the required extensions,
    attaching the catalog, and routing the session to the correct schema.

    The PostgreSQL backend is the recommended choice for **concurrent read/write**
    deployments: a read-only consumer (e.g. a GraphQL API) can query the catalog
    while another process updates it, which the file-based DuckDB catalog cannot
    support (it is locked at the process level). Use :meth:`from_postgres` to build
    such a connector ergonomically; PostgreSQL credentials are supplied through a
    DuckDB secret rather than embedded in the connection string.

    After calling ``connect()``, the returned ``duckdb.DuckDBPyConnection`` can be
    passed directly to ``DuckLakeTablesBuilder``, ``DatabaseUpdater``, or any other
    class that accepts a ``connection`` parameter.

    Attributes:
        catalog_path (str): Locator of the catalog. A file path for the DuckDB
            backend, or a backend-prefixed connection string for the others
            (e.g. ``'postgres:dbname=ducklake'``, ``'sqlite:catalog.sqlite'``).
        data_path (str): Directory where Parquet data files are stored.
        catalog_type (CatalogType): Catalog backend (DuckDB, SQLite or PostgreSQL).
        meta_secret (Optional[str]): Name of the DuckDB secret holding the catalog
            backend credentials, referenced via ``META_SECRET`` (PostgreSQL only).
        read_only (bool): Whether the connection is read-only.
        catalog_alias (str): Alias used in the ``ATTACH`` statement (default ``'db'``).
        schema (str): DuckLake schema to activate with ``USE`` (default ``'main'``).
        logger (logging.Logger): Logger instance.

    Examples:
        >>> # Read-write connection (local DuckDB catalog file)
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
        >>> # Read-only connection (GraphQL API, dashboard)
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/',
        read_only=True).connect()
        >>> # Time-travel to a specific snapshot (ML run audit)
        >>> conn = DuckLakeConnector('catalog.ducklake', 'data/',
        snapshot_version=3).connect()
        >>> # PostgreSQL catalog for concurrent read/write (production)
        >>> conn = DuckLakeConnector.from_postgres(
        ...     'data/', dbname='ducklake', host='localhost',
        ...     user='app', password='***',
        ... ).connect()
    """

    # Initialisation
    def __init__(
        self,
        catalog_path: str,
        data_path: str,
        read_only: bool = False,
        snapshot_version: int | None = None,
        snapshot_time: str | None = None,
        catalog_alias: str = "db",
        schema: str = "main",
        catalog_type: CatalogType | str = CatalogType.DUCKDB,
        meta_secret: str | None = None,
        log_filename: str | os.PathLike[str] | None = os.path.join(
            FILE_PATH.parents[2], "logs/ducklake_connector.log"
        ),
    ) -> None:
        """
        Initialize the DuckLakeConnector.

        For the PostgreSQL backend, prefer the :meth:`from_postgres` class method,
        which assembles the connection string and the credential secret for you.

        Args:
            catalog_path (str): Locator of the catalog. For the DuckDB backend, a
                file path (e.g. ``'data/catalog.ducklake'``). For other backends, a
                backend-prefixed connection string (e.g.
                ``'postgres:dbname=ducklake'`` or ``'sqlite:catalog.sqlite'``).
            data_path (str): Directory where Parquet data files are stored
                (e.g. ``'data/files/'``).
            read_only (bool): Open the catalog in read-only mode. Useful for
                dashboard/API layers that must not write. Defaults to False.
            snapshot_version (Optional[int]): Open a historical snapshot by
                version number. Implies ``READ_ONLY``. Defaults to None.
                **Important**: with the file-based DuckDB backend, DuckLake locks
                the catalog file at the process level, so a snapshot connection
                cannot coexist with another connection to the same catalog. When an
                active connection already exists, use :meth:`at_clause` instead to
                build an ``AT (VERSION => n)`` clause.
            snapshot_time (Optional[str]): Open a historical snapshot by
                timestamp (ISO-8601 string, e.g. ``'2025-01-01 00:00:00'``).
                Implies ``READ_ONLY``. Defaults to None.
                Same restriction as ``snapshot_version`` above; prefer
                :meth:`at_clause` when a connection is already open.
            catalog_alias (str): Alias for the attached DuckLake catalog in SQL
                (e.g. ``USE {alias}.{schema}``). Defaults to ``'db'``.
            schema (str): DuckLake schema to activate. Defaults to ``'main'``.
            catalog_type (CatalogType | str): Catalog backend to use. Defaults to
                ``CatalogType.DUCKDB`` (local file). Drives which DuckDB extensions
                are loaded by :meth:`connect`.
            meta_secret (Optional[str]): Name of a DuckDB secret carrying the
                catalog backend credentials, referenced via ``META_SECRET`` in the
                ``ATTACH`` statement (PostgreSQL only). Defaults to None.
            log_filename (Optional[os.PathLike]): Path to the log file.

        Examples:
            >>> connector = DuckLakeConnector('catalog.ducklake', 'data/')
            >>> conn = connector.connect()
        """
        # Stockage des paramètres de connexion
        self.catalog_path = str(catalog_path)
        self.data_path = str(data_path)
        # Normalisation du backend de catalogue (accepte une chaîne ou un CatalogType).
        # CatalogType(...) valide la valeur et lève ValueError si elle est inconnue.
        self.catalog_type = CatalogType(catalog_type)
        self.meta_secret = meta_secret
        # snapshot_version et snapshot_time impliquent un accès en lecture seule :
        # DuckLake ouvre automatiquement le catalogue en READ_ONLY dans ce cas.
        # L'attribut self.read_only reflète cet état effectif pour cohérence.
        self.read_only = (
            read_only or snapshot_version is not None or snapshot_time is not None
        )
        self.snapshot_version = snapshot_version
        self.snapshot_time = snapshot_time
        self.catalog_alias = catalog_alias
        self.schema = schema
        # SQL de création du secret de session, renseigné par from_postgres lorsque
        # des identifiants bruts sont fournis ; exécuté avant ATTACH par connect/attach.
        # Conservé à part pour ne jamais être journalisé (il contient le mot de passe).
        self._secret_sql: str | None = None

        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(
                FILE_PATH.parents[2], "logs/ducklake_connector.log"
            )
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
        2. Install and load the required extensions (``ducklake`` plus the catalog
           backend extension when applicable, e.g. ``postgres``).
        3. Create the credential secret when one is configured (PostgreSQL).
        4. Build and execute the ``ATTACH`` statement with appropriate options.
        5. Activate the target schema with ``USE``.

        Returns:
            duckdb.DuckDBPyConnection: Configured DuckDB connection ready for use.

        Raises:
            duckdb.Error: If an extension cannot be loaded or the catalog cannot
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
        # est géré séparément (fichier .ducklake ou serveur Postgres).
        conn = duckdb.connect(":memory:")

        # Chargement des extensions requises puis création éventuelle du secret
        self._load_extensions(conn)
        self._apply_secret(conn)

        # Construction de la chaîne d'options ATTACH
        attach_sql = self._build_attach_sql()
        conn.execute(attach_sql)
        self.logger.info(
            f"DuckLake catalog attached : '{self.catalog_path}' "
            f"(type={self.catalog_type.value}, alias={self.catalog_alias}, "
            f"read_only={self.read_only})"
        )

        # Activation du schéma cible pour que les requêtes non qualifiées fonctionnent
        conn.execute(f"USE {self.catalog_alias}.{self.schema}")
        self.logger.info(f"Activated scheme : {self.catalog_alias}.{self.schema}")

        return conn

    # ---------------------------------------------------------------------------
    # Méthodes de voyage dans le temps (time-travel) sur une connexion existante
    # ---------------------------------------------------------------------------

    # Génération d'une clause SQL AT pour requête time-travel sur connexion existante
    def at_clause(self) -> str:
        """
        Return the ``AT (...)`` SQL clause for time-travel queries on an existing
        connection.

        DuckLake prevents the same catalog file from being attached more than once
        per process (even under different aliases), so ``snapshot_version`` /
        ``snapshot_time`` cannot be used by opening a second connection.  Instead,
        append the returned clause to the ``FROM`` part of any query:

            ``SELECT * FROM table_name {connector.at_clause()}``

        Returns ``""`` when neither ``snapshot_version`` nor ``snapshot_time`` was
        supplied, so the method is safe to call unconditionally.

        Returns:
            str: SQL ``AT (VERSION => n)`` or ``AT (TIMESTAMP => 'ts')`` clause,
                or an empty string for the current snapshot.

        Raises:
            ValueError: If both ``snapshot_version`` and ``snapshot_time`` are set.

        Examples:
            >>> c = DuckLakeConnector('cat.ducklake', 'data/', snapshot_version=3)
            >>> c.at_clause()
            'AT (VERSION => 3)'
            >>> c2 = DuckLakeConnector('cat.ducklake', 'data/',
            snapshot_time='2025-01-01')
            >>> c2.at_clause()
            "AT (TIMESTAMP => '2025-01-01')"
            >>> DuckLakeConnector('cat.ducklake', 'data/').at_clause()
            ''
        """
        # Génération de la clause AT selon les paramètres de time-travel configurés
        if self.snapshot_version is not None:
            return f"AT (VERSION => {self.snapshot_version})"
        if self.snapshot_time is not None:
            return f"AT (TIMESTAMP => '{self.snapshot_time}')"
        return ""

    # Attachement d'un catalogue DuckLake à une connexion DuckDB existante
    def attach(self, conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
        """
        Attach the DuckLake catalog to an already-open DuckDB connection.

        Useful when sharing a connection across multiple catalogs. Required
        extensions are (idempotently) loaded and the credential secret is created
        when configured, so the call also works for the PostgreSQL backend. The
        ``USE`` statement is then executed to activate the configured schema.

        Args:
            conn (duckdb.DuckDBPyConnection): Existing DuckDB connection.

        Returns:
            duckdb.DuckDBPyConnection: The same connection, now with the catalog
            attached and the schema activated.

        Examples:
            >>> import duckdb
            >>> conn = duckdb.connect(':memory:')
            >>> connector = DuckLakeConnector('catalog.ducklake', 'data/')
            >>> connector.attach(conn)
        """
        # Préparation de la connexion existante : chargement des extensions
        # (idempotent) puis création éventuelle du secret d'identifiants
        self._load_extensions(conn)
        self._apply_secret(conn)

        # Attachement du catalogue sur la connexion existante
        attach_sql = self._build_attach_sql()
        conn.execute(attach_sql)
        conn.execute(f"USE {self.catalog_alias}.{self.schema}")
        self.logger.info(
            f"DuckLake catalog attached to the existing connection:"
            f"'{self.catalog_path}'"
        )
        return conn

    # ---------------------------------------------------------------------------
    # Constructeur de classe pour le backend PostgreSQL
    # ---------------------------------------------------------------------------

    # Création d'un connecteur attaché à un catalogue PostgreSQL
    @classmethod
    def from_postgres(
        cls,
        data_path: str,
        *,
        dbname: str,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        meta_secret: str | None = None,
        secret_name: str = "ducklake_pg_secret",
        read_only: bool = False,
        snapshot_version: int | None = None,
        snapshot_time: str | None = None,
        catalog_alias: str = "db",
        schema: str = "main",
        log_filename: str | os.PathLike[str] | None = None,
    ) -> "DuckLakeConnector":
        """
        Build a connector backed by a PostgreSQL catalog.

        This is the recommended entry point for concurrent read/write deployments,
        where a read-only consumer (e.g. a GraphQL API) queries the catalog while
        another process updates it. The catalog metadata lives in PostgreSQL while
        row data stays in Parquet files under ``data_path`` (local or object store).

        Credentials are passed to DuckDB through a secret (``CREATE SECRET ...
        TYPE postgres``) referenced by ``META_SECRET`` in the ``ATTACH`` statement,
        instead of being embedded in the connection string. Two modes are supported:

        - **Inline credentials** (``host``/``user``/``password``/...): a *session*
          (non-persistent) secret is created on the connection at ``connect()`` time
          from the provided fields. Any field left as ``None`` is omitted, letting
          DuckDB's PostgreSQL driver fall back to the standard libpq environment
          variables (``PGHOST``, ``PGUSER``, ``PGPASSWORD``, ...).
        - **Externally managed secret** (``meta_secret`` set): an existing secret is
          referenced by name and no credentials flow through Python. Recommended for
          production, where the secret is created out of band.

        Args:
            data_path (str): Directory where Parquet data files are stored.
            dbname (str): Name of the PostgreSQL catalog database. Included in the
                ``ATTACH`` connection string (it is not a secret).
            host (Optional[str]): PostgreSQL host. Falls back to ``PGHOST`` if None.
            port (Optional[int]): PostgreSQL port. Falls back to ``PGPORT`` if None.
            user (Optional[str]): PostgreSQL user. Falls back to ``PGUSER`` if None.
            password (Optional[str]): PostgreSQL password. Falls back to
                ``PGPASSWORD`` if None.
            meta_secret (Optional[str]): Name of an existing DuckDB secret to
                reference. When provided, no secret is created and the
                ``host``/``port``/``user``/``password`` arguments are ignored.
            secret_name (str): Name of the session secret to create from inline
                credentials. Defaults to ``'ducklake_pg_secret'``.
            read_only (bool): Open the catalog in read-only mode (e.g. for the API).
            snapshot_version (Optional[int]): Time-travel by snapshot version.
            snapshot_time (Optional[str]): Time-travel by ISO-8601 timestamp.
            catalog_alias (str): Alias for the attached catalog. Defaults to ``'db'``.
            schema (str): DuckLake schema to activate. Defaults to ``'main'``.
            log_filename (Optional[os.PathLike]): Path to the log file.

        Returns:
            DuckLakeConnector: A connector configured for the PostgreSQL backend.

        Examples:
            >>> # Read-write update job (inline credentials)
            >>> rw = DuckLakeConnector.from_postgres(
            ...     'data/', dbname='ducklake', host='localhost',
            ...     user='app', password='***',
            ... )
            >>> # Read-only API, reusing a secret created out of band
            >>> ro = DuckLakeConnector.from_postgres(
            ...     'data/', dbname='ducklake',
            ...     meta_secret='pg_catalog_secret', read_only=True,
            ... )
        """
        # Cible ATTACH du backend Postgres : le nom de la base figure dans la chaîne
        # de connexion (non sensible) ; les identifiants passent par le secret.
        catalog_path = f"postgres:dbname={dbname}"

        # Détermination du secret à référencer et du SQL de création éventuel
        secret_sql: str | None = None
        if meta_secret is not None:
            # Mode « secret externe » : référence d'un secret existant, sans création.
            effective_secret = meta_secret
        else:
            # Mode « identifiants en ligne » : création d'un secret de session.
            effective_secret = secret_name
            secret_sql = cls._build_postgres_secret_sql(
                secret_name=secret_name,
                dbname=dbname,
                host=host,
                port=port,
                user=user,
                password=password,
            )

        # Instanciation du connecteur en mode Postgres
        connector = cls(
            catalog_path=catalog_path,
            data_path=data_path,
            read_only=read_only,
            snapshot_version=snapshot_version,
            snapshot_time=snapshot_time,
            catalog_alias=catalog_alias,
            schema=schema,
            catalog_type=CatalogType.POSTGRES,
            meta_secret=effective_secret,
            log_filename=log_filename,
        )
        # Mémorisation du SQL de création du secret (exécuté avant ATTACH)
        connector._secret_sql = secret_sql
        return connector

    # ---------------------------------------------------------------------------
    # Méthodes privées de préparation de connexion et de construction SQL
    # ---------------------------------------------------------------------------

    # Chargement des extensions DuckDB requises selon le backend de catalogue
    def _load_extensions(self, conn: duckdb.DuckDBPyConnection) -> None:
        """
        Install and load the DuckDB extensions required by the catalog backend.

        ``ducklake`` is always loaded; the PostgreSQL or SQLite extension is added
        for the corresponding backend. ``INSTALL``/``LOAD`` are idempotent, so the
        method is safe to call on an already-prepared connection.

        Args:
            conn (duckdb.DuckDBPyConnection): Connection to configure.
        """
        # Extension DuckLake : toujours nécessaire
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        self.logger.info("DuckLake extension loaded")

        # Extension spécifique au backend de catalogue
        if self.catalog_type == CatalogType.POSTGRES:
            conn.execute("INSTALL postgres; LOAD postgres;")
            self.logger.info("PostgreSQL extension loaded")
        elif self.catalog_type == CatalogType.SQLITE:
            conn.execute("INSTALL sqlite; LOAD sqlite;")
            self.logger.info("SQLite extension loaded")

    # Création du secret d'identifiants du catalogue lorsqu'il est configuré
    def _apply_secret(self, conn: duckdb.DuckDBPyConnection) -> None:
        """
        Create the catalog credential secret on the connection, if configured.

        Only the secret *name* is logged: the SQL statement carries the password
        and must never be written to the logs.

        Args:
            conn (duckdb.DuckDBPyConnection): Connection on which to create the secret.
        """
        # Création du secret uniquement si un SQL a été préparé (identifiants en ligne)
        if self._secret_sql is not None:
            conn.execute(self._secret_sql)
            self.logger.info(
                f"Catalog credential secret created : '{self.meta_secret}'"
            )

    # Construction du SQL de création d'un secret PostgreSQL
    @staticmethod
    def _build_postgres_secret_sql(
        secret_name: str,
        dbname: str,
        host: str | None,
        port: int | None,
        user: str | None,
        password: str | None,
    ) -> str:
        """
        Build a ``CREATE OR REPLACE SECRET ... (TYPE postgres, ...)`` statement.

        Fields left as ``None`` are omitted so that DuckDB's PostgreSQL driver can
        fall back to the libpq environment variables. The created secret is a
        session (non-persistent) secret. String values are escaped for SQL.

        Args:
            secret_name (str): Identifier of the secret to create.
            dbname (str): PostgreSQL database name (always included).
            host (Optional[str]): PostgreSQL host.
            port (Optional[int]): PostgreSQL port.
            user (Optional[str]): PostgreSQL user.
            password (Optional[str]): PostgreSQL password.

        Returns:
            str: The ``CREATE OR REPLACE SECRET`` SQL statement.
        """
        # Paramètres du secret : TYPE et DATABASE toujours présents
        params: list[str] = [
            "TYPE postgres",
            f"DATABASE '{_quote_literal(dbname)}'",
        ]
        # Champs optionnels : omis si non fournis (repli sur l'environnement libpq)
        if host is not None:
            params.append(f"HOST '{_quote_literal(host)}'")
        if port is not None:
            params.append(f"PORT {int(port)}")
        if user is not None:
            params.append(f"USER '{_quote_literal(user)}'")
        if password is not None:
            params.append(f"PASSWORD '{_quote_literal(password)}'")

        params_str = ", ".join(params)
        return f"CREATE OR REPLACE SECRET {secret_name} ({params_str})"

    # Construction de la clause SQL ATTACH avec les options appropriées
    def _build_attach_sql(self) -> str:
        """
        Build the ``ATTACH`` SQL statement from the connector configuration.

        The option string is built incrementally:
        - ``DATA_PATH`` is always included.
        - ``META_SECRET`` is appended when a credential secret is configured
          (PostgreSQL backend).
        - ``READ_ONLY`` is appended for read-only or time-travel connections.
        - ``SNAPSHOT_VERSION`` or ``SNAPSHOT_TIME`` is appended for time travel.

        Returns:
            str: The complete ``ATTACH`` SQL statement.
        """
        # Liste des options ATTACH à construire
        options = [f"DATA_PATH '{self.data_path}'"]

        # Référence au secret d'identifiants du catalogue (backend PostgreSQL)
        if self.meta_secret is not None:
            options.append(f"META_SECRET '{self.meta_secret}'")

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
            f"ATTACH 'ducklake:{self.catalog_path}' AS {self.catalog_alias}"
            f" ({options_str})"
        )
