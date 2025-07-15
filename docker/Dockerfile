# Dockerfile
FROM python:3.12-slim

# Installation d'uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers de configuration uv
COPY pyproject.toml uv.lock* ./

# Installation des dépendances avec uv
RUN uv sync --frozen --no-install-project

# Copie du code source
COPY . .

# Installation du projet
RUN uv sync --frozen

# Variables d'environnement
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Exposition du port pour Jupyter (optionnel)
EXPOSE 8888

# Point d'entrée par défaut
CMD ["python", "-c", "print('Dashboard Template Database container is ready!')"]