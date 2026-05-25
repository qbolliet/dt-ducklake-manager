# Importation des modules
# Modules de base
import logging
import os

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
