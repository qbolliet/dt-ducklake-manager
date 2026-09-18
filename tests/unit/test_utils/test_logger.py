# Importation des modules
# Modules de base
import logging
import os
from pathlib import Path

# Module à tester
from dt_ducklake_manager.utils.logger import _init_logger


# Test de la création automatique du dossier de logs
def test_init_logger_creates_directory() -> None:
    # Test avec un nouveau dossier
    test_log_dir = "test_logs"
    test_log_file = os.path.join(test_log_dir, "test.log")

    # Vérification que le dossier n'existe pas déjà
    if os.path.exists(test_log_dir):
        os.rmdir(test_log_dir)

    # Initialisation du logger
    logger = _init_logger(test_log_file)

    # Vérification que le dossier a été créé
    assert os.path.exists(test_log_dir)

    # Nettoyage
    logger.handlers = []  # Suppression des handlers relatifs à la création du fichier
    os.remove(test_log_file)
    os.rmdir(test_log_dir)


# Test des éléments retournés par le logger
def test_init_logger_returns_logger() -> None:
    # Test avec un fichier temporaire
    test_log_file = "temp_test.log"

    # Initialisation du logger
    logger = _init_logger(test_log_file)

    # vérification de son type
    assert isinstance(logger, logging.Logger)

    # Vérification du niveau par défaut
    assert logger.level == logging.INFO

    # Vérification des handlers
    assert (
        len(logger.handlers) >= 2
    )  # Il doit y avoir un handler pour les fichiers et pour le flux a minima

    # Nettoyage
    logger.handlers = []  # Suppression des handlers relatifs à la création du fichier
    os.remove(test_log_file)


# Test de la configuration du file handler
def test_init_logger_file_handler_configuration() -> None:
    # Initialisation du logger avec un fichier de tests
    test_log_file = "test_config.log"
    logger = _init_logger(test_log_file)

    # Recherche du fichier de handler
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break

    # Vérification que le handler existe
    assert file_handler is not None
    # Vérification du niveau du handler
    assert file_handler.level == logging.INFO
    # Vérification du formatter
    assert isinstance(file_handler.formatter, logging.Formatter)

    # Nettoyage
    file_handler.close()
    logger.handlers = []  # Suppression des handlers relatifs à la création du fichier
    os.remove(test_log_file)


# Test que des appels répétés n'accumulent pas de handlers en double
def test_init_logger_is_idempotent_for_handlers() -> None:
    """Test that repeated calls with the same name do not stack duplicate handlers.

    Previously a FileHandler was added to the root logger on every call, so each
    instantiated manager duplicated every log line.
    """
    logger_name = "test_idempotent_logger"
    test_log_file = "test_idempotent.log"

    # Nettoyage préalable d'un éventuel état résiduel
    logging.getLogger(logger_name).handlers = []

    logger = _init_logger(test_log_file, name=logger_name)
    handler_count = len(logger.handlers)

    # Appels supplémentaires : aucun handler équivalent ne doit être ajouté
    for _ in range(3):
        again = _init_logger(test_log_file, name=logger_name)
        assert again is logger
        assert len(again.handlers) == handler_count

    # Nettoyage
    for handler in logger.handlers:
        handler.close()
    logger.handlers = []
    if os.path.exists(test_log_file):
        os.remove(test_log_file)


# Test que le chemin de log par défaut est ancré sur le répertoire de travail
def test_init_logger_default_path_under_cwd(tmp_path: Path) -> None:
    """Test that the default log file lands under ``<cwd>/logs`` and never in the
    package installation directory.

    Args:
        tmp_path: pytest temporary directory used as the working directory.
    """
    logger_name = "test_default_path_logger"
    logging.getLogger(logger_name).handlers = []

    previous_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        logger = _init_logger(name=logger_name)
        # Émission d'un message pour matérialiser le fichier
        logger.info("materialise the log file")
        expected = tmp_path / "logs" / f"{logger_name}.log"
        assert expected.exists()
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []
        os.chdir(previous_cwd)
