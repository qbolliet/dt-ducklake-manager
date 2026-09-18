# Importation des modules
# Modules de base
# Logging
import logging
import os
from pathlib import Path

# Nom du logger racine du package et format commun des messages
_PACKAGE_LOGGER_NAME = "dt_ducklake_manager"
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


# Fonction de résolution du chemin du fichier de log par défaut
def _default_log_path(name: str) -> Path:
    """
    Build the default log file path for a given component name.

    The path is anchored on the current working directory rather than on the
    package installation directory, so logs never land inside the installed
    package tree.

    Args:
        name (str): Component name (e.g. ``'ducklake_connector'``). Used as the
            log file stem.

    Returns:
        Path: ``<cwd>/logs/<name>.log``.

    Examples:
        >>> _default_log_path("ducklake_connector").name
        'ducklake_connector.log'
        >>> _default_log_path("ducklake_connector").parent.name
        'logs'
    """
    # Ancrage sur le répertoire de travail courant
    return Path.cwd() / "logs" / f"{name}.log"


# Fonction d'initialisation d'un logger nommé écrivant dans un fichier
def _init_logger(
    filename: str | os.PathLike[str] | None = None,
    name: str = _PACKAGE_LOGGER_NAME,
) -> logging.Logger:
    """
    Initialize a named logger writing to both a file and the console.

    A **named** logger is used (never the root logger) and the function is
    idempotent: before adding a file or stream handler it checks that no
    equivalent handler is already attached, so repeated calls (one per manager
    instance) never accumulate duplicate handlers — which previously caused each
    log line to be emitted once per instantiated manager.

    Args:
        filename (Optional[os.PathLike]): Explicit path to the log file. When
            ``None`` (default), ``<cwd>/logs/<name>.log`` is used and the parent
            directory is created if missing.
        name (str): Logger name. Also used as the default log file stem. Defaults
            to ``'dt_ducklake_manager'``.

    Returns:
        logging.Logger: The configured named logger (level ``INFO``), carrying one
        file handler and one stream handler.

    Examples:
        >>> logger = _init_logger(name="ducklake_connector")
        >>> logger.name
        'ducklake_connector'
        >>> logger.level == logging.INFO
        True
        >>> # Second call: no duplicate handler is added
        >>> before = len(logger.handlers)
        >>> _ = _init_logger(name="ducklake_connector")
        >>> len(logger.handlers) == before
        True
    """
    # Résolution du chemin du fichier de log
    log_path = Path(filename) if filename is not None else _default_log_path(name)

    # Création du dossier parent si absent (chemin relatif nu exclu)
    if str(log_path.parent) not in ("", "."):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Logger nommé (jamais le logger racine) pour éviter l'accumulation de handlers
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Formatteur commun aux deux handlers
    formatter = logging.Formatter(_LOG_FORMAT)

    # Cible absolue du fichier, pour comparer aux handlers déjà attachés
    target = str(log_path.resolve())

    # Ajout du FileHandler uniquement si aucun handler équivalent n'est présent
    has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == target
        for handler in logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Ajout d'un StreamHandler (console) une seule fois : un FileHandler étant une
    # sous-classe de StreamHandler, on exclut explicitement ce cas.
    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
