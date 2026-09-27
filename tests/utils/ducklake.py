# Importation des modules
# Modules de base
from functools import cache

import duckdb

# Module de tests
import pytest


# Détection, une seule fois par session, de la disponibilité de l'extension
@cache
def ducklake_available() -> bool:
    """Check whether the DuckLake extension can be loaded in this environment.

    The result is cached for the whole test session: installing and loading the
    extension is attempted once, whatever the number of modules asking.

    Returns:
        bool: True if ``INSTALL ducklake; LOAD ducklake;`` succeeds.

    Examples:
        >>> isinstance(ducklake_available(), bool)
        True
    """
    try:
        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        conn.close()
        return True
    except Exception:
        return False


# Marqueur des tests exigeant un vrai catalogue DuckLake
requires_ducklake = pytest.mark.skipif(
    not ducklake_available(),
    reason="DuckLake extension unavailable in this environment",
)
