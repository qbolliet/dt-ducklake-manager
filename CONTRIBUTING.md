# Contributing

## Prérequis

- [uv](https://docs.astral.sh/uv/) >= 0.5
- Python >= 3.13
- [pre-commit](https://pre-commit.com/)

## Installation de l'environnement

```bash
uv sync
pre-commit install
```

## Commandes utiles

```bash
# Tests
uv run pytest                          # tous les tests unitaires
uv run pytest -m integration           # tests d'intégration uniquement
uv run pytest -m "not integration"     # tests unitaires uniquement

# Qualité de code
uv run ruff check .                    # linting
uv run ruff format .                   # formatage
uv run mypy dt_ducklake_manager/       # vérification des types

# Documentation
uv run mkdocs serve                    # aperçu local sur http://127.0.0.1:8000

# Versioning & release
uv run git-cliff --output CHANGELOG.md # régénérer le CHANGELOG complet
uv run bump-my-version bump patch      # bumper la version patch (0.1.0 → 0.1.1)
uv run bump-my-version bump minor      # bumper la version mineure (0.1.0 → 0.2.0)
uv run bump-my-version bump major      # bumper la version majeure (0.1.0 → 1.0.0)
uv run bump-my-version bump patch --dry-run  # simulation sans modification
```

## Format des commits (Conventional Commits)

Ce projet utilise [Conventional Commits](https://www.conventionalcommits.org/). Chaque message de commit doit suivre ce format :

```
<type>[(<scope>)]: <description>

[corps optionnel]

[footer optionnel]
```

### Types autorisés

| Type       | Description                                      | Impact version |
|------------|--------------------------------------------------|----------------|
| `feat`     | Nouvelle fonctionnalité                          | minor          |
| `fix`      | Correction de bug                                | patch          |
| `perf`     | Amélioration de performance                      | patch          |
| `refactor` | Refactoring sans changement de comportement      | —              |
| `test`     | Ajout ou modification de tests                   | —              |
| `docs`     | Documentation uniquement                         | —              |
| `style`    | Formatage, espaces, virgules...                  | —              |
| `chore`    | Maintenance (dépendances, config CI...)          | —              |

Un `!` après le type ou `BREAKING CHANGE:` dans le footer indique un **changement cassant** → impact `major`.

### Exemples

```
feat(operations): add database merge operation
fix(schema): handle empty fact table during schema inference
test(updater): add edge case for null primary key
docs: update README with uv installation instructions
chore(deps): bump duckdb to 1.6.0
feat!: remove support for Python 3.12
```

## Workflow de release

```bash
# 1. S'assurer que tous les tests passent sur main
git checkout main && git pull

# 2. Bumper la version (met à jour pyproject.toml ET crée le tag git)
uv run bump-my-version bump minor   # ou patch / major

# 3. Régénérer le CHANGELOG
uv run git-cliff --output CHANGELOG.md

# 4. Committer le CHANGELOG et pusher avec le tag
git add CHANGELOG.md
git commit -m "chore(release): prepare for v$(uv run bump-my-version show current_version)"
git push origin main --tags
# → GitHub Actions crée automatiquement la release GitHub
```

## Utiliser le package depuis un autre projet

```bash
# Via uv (recommandé)
uv add dt-ducklake-manager --source git+https://github.com/qbolliet/dt-ducklake-manager

# Épingler une version spécifique
uv add dt-ducklake-manager --source git+https://github.com/qbolliet/dt-ducklake-manager@v0.1.0
```
