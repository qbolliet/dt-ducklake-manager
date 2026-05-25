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
```

> Le versioning et le CHANGELOG sont entièrement automatisés par
> [release-please](https://github.com/googleapis/release-please) (cf. *Workflow
> de release*). Aucune commande manuelle de bump ou de génération de changelog
> n'est nécessaire.

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

Les releases sont automatisées par [release-please](https://github.com/googleapis/release-please).
Aucune action manuelle de versioning n'est requise : il suffit de merger des
commits conventionnels sur `main`.

1. **Merge sur `main`** de commits conventionnels (`feat:`, `fix:`, etc.).
2. release-please ouvre (ou met à jour) automatiquement une **PR « release »**
   qui agrège les changements, calcule la prochaine version selon le SemVer
   (`feat` → minor, `fix` → patch, `!`/`BREAKING CHANGE` → major) et met à jour
   `CHANGELOG.md` ainsi que la version dans `pyproject.toml`.
3. **Merge de la PR « release »** → release-please crée le tag `vX.Y.Z` et la
   release GitHub correspondante.

> La version dans `pyproject.toml` et le `CHANGELOG.md` sont gérés par le bot ;
> ne les éditez pas à la main.

## Utiliser le package depuis un autre projet

```bash
# Via uv (recommandé)
uv add dt-ducklake-manager --source git+https://github.com/qbolliet/dt-ducklake-manager

# Épingler une version spécifique
uv add dt-ducklake-manager --source git+https://github.com/qbolliet/dt-ducklake-manager@v0.1.0
```
