# Prompts d'implémentation — refonte du schéma de la base de données

> Série de prompts à exécuter **dans l'ordre**, chacun dans une session Claude Code
> fraîche, depuis la racine du dépôt. La spécification de référence est
> `specification-bdd.md` (à la racine) : chaque prompt en demande la lecture, ne pas la
> supprimer avant la fin de la série. `revue-technique-bdd.md` est la revue historique qui
> a précédé la spécification ; les prompts ne s'y réfèrent plus.
>
> **Pas de versionnage v1 → v2.** Le schéma produit par cette série **est** la version 1
> du schéma et de l'API (projet en développement, rien de publié). Aucun script de
> migration n'est écrit : les catalogues de développement sont reconstruits depuis les
> données sources. Le mot « v2 » ne doit apparaître nulle part dans le code, les tests, les
> docstrings ni la documentation.
>
> **Conventions communes** (rappelées dans chaque prompt, en complément de CLAUDE.md) :
> annotations de types partout ; commentaires en FRANÇAIS (formulations nominales) ;
> docstrings en ANGLAIS, convention Google, avec exemples ; tests pytest ciblant les
> cas limites ; lancer les tests avec `uv run --no-sync pytest` (le `--no-sync` évite
> l'échec « Accès refusé » d'uv sous OneDrive). Les faits DuckLake marqués *mesuré* dans la
> spécification priment sur la mémoire du modèle ; en cas de doute, revérifier sur la
> version installée (annexe A de la spécification).
>
> **Choix du modèle** : Opus pour les prompts qui exigent des décisions d'architecture
> ou touchent beaucoup de fichiers interdépendants ; Sonnet pour les tâches mécaniques
> bien spécifiées. **Plan mode** : activé quand la tâche comporte des choix
> d'implémentation à valider avant d'écrire ; inutile quand la spécification ci-dessous
> est déjà un plan.

---

## Étape M (manuelle, réalisée) — intégration du `DuckLakeConnector`

**Faite le 31 août 2026**, variante minimale : `catalog_alias` disponible partout où
`schema` l'est (`BaseSchemaManager` et sous-classes, `DuckLakeTablesBuilder`,
`DatabaseAuditor`, `AtomicDatabaseOperations`), `schema` ajouté à `DuckLakeMaintenance`,
`DuckLakeConnector.attach()` doté de `activate_schema: bool = True`. Le `DuckLakeContext`
et la propagation de `read_only` sont différés ; les prompts ne les supposent pas.

---

## Prompt 1 — Corrections préalables (types, quoting, qualification, index, logs)

**Modèle : Sonnet · Plan mode : non · Dépendances : étape M**

```text
Lis d'abord specification-bdd.md (sections 3, 7 et 10) pour le contexte. Applique
cinq corrections indépendantes du schéma, sans changer aucun comportement public autre
que ceux décrits :

1) Mapping des types (dt_ducklake_manager/utils/types.py) : préserve les largeurs
   d'entiers — Int8→TINYINT, Int16→SMALLINT, Int32→INTEGER, Int64→BIGINT, et les
   non signés UInt8→UTINYINT … UInt64→UBIGINT — et mappe Float32→FLOAT,
   Float64→DOUBLE. Mets à jour la docstring et ses exemples (l'exemple actuel annonce
   'INTEGER' pour une colonne d'entiers polars, qui est Int64 : il doit devenir
   'BIGINT'), puis tous les tests qui supposaient INTEGER/DOUBLE pour ces types
   (tests/unit/test_utils/test_types.py et les tests d'intégration).

2) Quoting des identifiants SQL : ajoute dans dt_ducklake_manager/utils/sql.py une
   fonction quote_ident(name: str) -> str qui entoure l'identifiant de guillemets
   doubles en doublant les guillemets internes (norme SQL). Utilise-la partout où un nom
   de table ou de colonne issu des données est interpolé dans une requête
   (schema/persistence.py, _internal/managers/*.py, operations/*.py, maintenance/*.py).
   Attention aux clauses où l'identifiant apparaît plusieurs fois : conditions de
   jointure construites par " AND ".join (updater.py, data.py), clause SET de
   _direct_upsert_data (data.py), listes de colonnes des INSERT ... SELECT (data.py),
   et build_database_duplicate_removal_query (utils/sql.py).

3) Qualification par le catalogue : qualify_table ne qualifie aujourd'hui que par le
   schéma, si bien que les requêtes se résolvent dans le catalogue COURANT de la
   connexion et non dans catalog_alias (cf. section 7 de la spécification). Fais
   évoluer la signature en
   qualify_table(table: str, schema: str = "main", catalog: str | None = None) -> str
   retournant "catalog"."schema"."table" quand catalog est fourni et "schema"."table"
   sinon (indispensable pour les connexions in-memory des tests, qui n'ont pas d'alias),
   en réutilisant quote_ident. Propage l'alias jusqu'à BaseSchemaManager._qualified et
   aux builders. NE QUALIFIE PAS : les vues temporaires enregistrées par conn.register()
   (temp_fact, temp_insert, temp_upsert, temp_metadata, temp_dim, _upd_split), ni les
   table functions ducklake_* — elles vivent dans le catalogue mémoire et la
   qualification les casserait. Ajoute un test vérifiant qu'avec deux catalogues
   attachés, un manager configuré sur l'un n'écrit jamais dans l'autre, quel que soit le
   dernier USE exécuté.

4) Suppression de la logique d'index, inopérante sous DuckLake : retire
   _drop_column_indexes_safe, _restore_column_indexes, _cleanup_orphaned_indexes dans
   operations/deleter.py, _rollback_index_changes dans operations/updater.py, ainsi que
   leurs appels (notamment les TransactionOperation 'drop_indexes' et
   'cleanup_indexes' de deleter.py) et leurs tests. Si l'auditor (maintenance/auditor.py)
   ou atomic.py référencent des index, retire aussi ces vérifications. Ne touche à rien
   d'autre dans ces fichiers.

5) Emplacement des logs : les chemins par défaut du type
   os.path.join(FILE_PATH.parents[2], "logs/...") écrivent dans le répertoire
   d'installation du package (connector.py, schema/persistence.py,
   schema/inference.py, _internal/managers/base.py, maintenance/compaction.py).
   Remplace ce défaut par Path.cwd() / "logs" / <nom>.log (création du dossier si
   absent), de façon centralisée dans utils/logger.py plutôt que répétée dans chaque
   module. Conserve la possibilité de passer un chemin explicite. Profites-en pour
   corriger _init_logger, qui ajoute un FileHandler au logger RACINE à chaque appel :
   les handlers s'accumulent (une ligne de log dupliquée par manager instancié).
   Utilise un logger nommé et vérifie l'absence de handler équivalent avant d'en
   ajouter un.

Conventions : types partout ; commentaires en français (formulations nominales) ;
docstrings anglaises Google avec exemples. Termine par uv run --no-sync pytest et
corrige jusqu'au vert. Résume les fichiers touchés.
```

*Pourquoi Sonnet sans plan mode : cinq changements mécaniques, entièrement spécifiés,
sans décision d'architecture. Le point 3 est ici parce qu'il touche exactement les mêmes
lignes que le point 2.*

---

## Prompt 2 — Labels dans la fact table, suppression des dimensions, `dataset_metadata`

**Modèle : Opus · Plan mode : OUI · Dépendances : prompt 1**

```text
Lis d'abord specification-bdd.md en entier — c'est la spécification. Objectif :
la fact_table stocke directement les labels (strings d'origine) dans les colonnes
catégorielles ; plus aucun code synthétique, plus aucune table de dimension, d'aucune
sorte. Le dictionary-encoding Parquet absorbe le coût de stockage. Périmètre : ce dépôt
uniquement (l'API GraphQL sera adaptée séparément). Il n'y a PAS de versionnage : le
schéma obtenu est la version 1 ; le mot « v2 » ne doit apparaître nulle part.

Spécification du comportement cible :

- SchemaBuilder (schema/inference.py) : create_fact_table ne substitue plus rien (la
  fact table est le DataFrame d'entrée dédupliqué) ; create_dimension_tables est
  supprimé. is_categorical reste calculé (String et n_unique <= categorical_threshold)
  mais devient une métadonnée d'UI pure. python_type est supprimé de la table metadata ;
  sql_type est conservé. Ajoute categorical_overrides: dict[str, bool] | None
  permettant de forcer le statut d'une colonne indépendamment du seuil ; une colonne
  forcée n'est jamais rebasculée par un update.

- DuckLakeTablesBuilder (schema/persistence.py) : n'écrit plus de tables dim_* ;
  metadata est créée sans python_type. Ajoute une table dataset_metadata (une ligne :
  label, description, source, updated_at TIMESTAMP, schema_version INTEGER = 1,
  cluster_by VARCHAR nullable — laissé NULL ici, renseigné au prompt 5), alimentée par
  des arguments optionnels du builder (dataset_label, dataset_description,
  dataset_source) ; updated_at et schema_version sont toujours renseignés. Retire la
  boucle de logging des « clés étrangères » de create_duckdb_fact_table.

- DimensionManager (_internal/managers/dimension.py) : supprime la classe et le
  module, ainsi que tous leurs usages (aucune table dim_* ne survit, ni inférée ni
  opt-in — cf. section 2.5 de la spécification pour la justification).

- DatabaseUpdater (operations/updater.py) : _prepare_dataframe_for_fact_table ne
  mappe plus labels→codes (les données passent telles quelles) ; _update_dimensions_safe
  se réduit à la mise à jour du booléen is_categorical dans metadata quand une colonne
  VARCHAR non forcée franchit le seuil dans un sens ou dans l'autre (un simple UPDATE
  metadata, plus aucune réécriture de la fact_table) ; les étapes de transaction
  « dimension_update » et « dimension_cleanup » disparaissent. ATTENTION : ce filtrage
  s'appuie aujourd'hui sur nw.col("python_type") == "String" — il doit passer sur
  sql_type == 'VARCHAR'. Le comptage du seuil reste fondé sur un COUNT(DISTINCT) de la
  fact_table (état post-upsert). Mets à jour dataset_metadata.updated_at à chaque
  update réussi.

- BaseSchemaManager (_internal/managers/base.py) : adapte _load_current_metadata (y
  compris le DataFrame vide de repli, qui liste encore python_type),
  _add_column_to_metadata (dont le CREATE TABLE IF NOT EXISTS metadata doit refléter le
  nouveau schéma) et _resolve_type_conflicts : la hiérarchie de résolution travaille sur
  les types SQL — BOOLEAN < TINYINT/SMALLINT/INTEGER/BIGINT < FLOAT/DOUBLE < VARCHAR,
  avec conservation de la largeur la plus grande à l'intérieur d'un même niveau (un
  BIGINT enregistré ne doit pas être rétrogradé en INTEGER par un lot d'Int32).

- DatabaseDeleter, auditor, recovery : retire les références aux dims et aux
  conversions catégorielles ; l'auditor ne valide plus la cohérence fact/dim mais
  vérifie la présence de dataset_metadata et la cohérence metadata/fact_table
  (mêmes colonnes, mêmes types SQL).

- README.md et docs/index.md : le paragraphe décrivant les « dimension tables » est
  faux dès ce prompt ; remplace-le par une phrase décrivant les trois tables (la
  documentation complète est faite au prompt 10). Corrige au passage l'exemple du
  README qui passe connector= au builder alors que la signature attend connection=.

Contraintes de style : types partout ; commentaires en français (formulations
nominales) ; docstrings anglaises Google avec exemples. Réécris les tests unitaires et
d'intégration impactés (tests/) en gardant la logique « cas limites » ; supprime les
tests des mécanismes disparus. Termine par uv run --no-sync pytest et corrige jusqu'au
vert, puis fournis un résumé des changements structuré par module.
```

*Pourquoi Opus + plan mode : refactor le plus large de la série, interdépendances
entre six modules.*

---

## Prompt 3 — Métadonnées d'UI : unit, display_format, family, description, default_aggregation

**Modèle : Sonnet · Plan mode : non · Dépendances : prompt 2**

```text
Lis d'abord specification-bdd.md (section 2.2). Ajoute à la table metadata les
colonnes d'UI, toutes VARCHAR nullable (NULL par défaut) : unit, display_format
(chaîne d3-format), family, description, default_aggregation (valeurs attendues :
SUM, AVG, MAX, MIN, COUNT, MEDIAN, MODE — à valider à l'écriture, erreur explicite
sinon).

Implémentation :
- SchemaBuilder.create_metadata_table accepte un paramètre column_metadata:
  dict[str, dict[str, str]] | None mappant nom de colonne → {label, unit,
  display_format, family, description, default_aggregation} (clés toutes optionnelles).
  Le paramètre column_labels existant est conservé tel quel ; si un label est aussi
  fourni via column_metadata, column_metadata prime. Valide que les noms de colonnes
  référencés existent dans le DataFrame (ValueError sinon) et qu'aucune clé inconnue ne
  se glisse dans les sous-dictionnaires (ValueError listant les clés fautives).
- DuckLakeTablesBuilder écrit ces colonnes dans le DDL de metadata et les propage.
- BaseSchemaManager._add_column_to_metadata insère NULL pour ces colonnes ; ajoute une
  méthode publique update_column_metadata(column: str, **fields) permettant de
  renseigner ou corriger ces champs (et label) sur une base existante sans
  reconstruction (UPDATE + invalidation du cache ; erreur explicite si la colonne
  n'existe pas dans metadata ou si le champ n'est pas un champ autorisé).
- DatabaseUpdater : lors d'un update, ces champs ne sont jamais écrasés (ils
  n'appartiennent qu'au producteur de métadonnées). Vérifie en particulier que
  _add_column_to_metadata, qui fait un UPDATE quand la colonne existe déjà, ne remet
  pas ces champs à NULL au passage.

Tests pytest sur les cas limites : colonne inconnue dans column_metadata,
default_aggregation invalide, clé inconnue dans un sous-dictionnaire, champs
partiellement renseignés, update qui ne doit pas écraser les champs,
update_column_metadata sur colonne absente. Complète le notebook d'illustration
existant (ou crée notebooks/4 - Métadonnées d'interface.ipynb) montrant l'effet de
column_metadata sur la table metadata. Conventions habituelles : types, commentaires
français nominaux, docstrings anglaises Google avec exemples. Termine par
uv run --no-sync pytest.
```

---

## Prompt 4 — Hiérarchies de colonnes (`parent_name`)

**Modèle : Sonnet · Plan mode : non · Dépendances : prompt 3**

```text
Lis d'abord specification-bdd.md (section 2.5). Une hiérarchie de menu (group-options,
arbre de sélection de profondeur arbitraire) est une CHAÎNE DE COLONNES de la fact_table
déclarée dans metadata.parent_name. Il n'y a aucune table auxiliaire : la hiérarchie
des valeurs est déjà dans la fact_table (SELECT DISTINCT region, departement, commune).

Implémentation :
- Ajoute parent_name VARCHAR nullable à la table metadata. Renseignable via
  column_metadata (prompt 3, clé parent_name) ou un paramètre dédié
  hierarchies: dict[str, str] | None (colonne → colonne parente) sur
  SchemaBuilder/DuckLakeTablesBuilder ; les deux sources doivent être cohérentes
  (ValueError sinon).
- Validations à l'écriture (ValueError explicite) : la colonne parente existe dans le
  DataFrame ; le graphe des parent_name est une forêt (pas de cycle, détection par
  parcours de proche en proche) ; une colonne d'une hiérarchie est catégorielle (si
  elle ne l'est pas par le seuil, elle est forcée à True avec un warning, comme le
  ferait categorical_overrides).
- update_column_metadata (prompt 3) accepte parent_name avec les mêmes validations
  contre l'état courant de metadata (une modification ne doit pas créer de cycle).
- Fonction utilitaire publique get_column_hierarchies(conn, schema, catalog_alias)
  -> list[list[str]] retournant les chaînes racine→feuille reconstituées depuis
  metadata (référence pour l'API).
- Convention documentée dans la docstring de get_column_hierarchies et dans le
  notebook : pour un arbre irrégulier, les niveaux absents sont NULL et l'arbre
  s'arrête au premier NULL ; on ne répète jamais la valeur du niveau supérieur.
- DatabaseDeleter.delete_columns : refuse la suppression d'une colonne parente d'une
  autre colonne (sauf cascade=True, qui met les parent_name des enfants à NULL avec un
  warning).

Tests pytest cas limites : cycle dans parent_name (A→B→A et auto-référence), parente
inexistante, hierarchies et column_metadata contradictoires, hiérarchie à un seul
niveau, hiérarchie profonde (5+ niveaux), deux hiérarchies indépendantes, colonne
non catégorielle forcée, suppression d'une colonne parente avec et sans cascade.
Crée notebooks/5 - Hiérarchies.ipynb illustrant la déclaration, la reconstitution des
chaînes et la construction d'un arbre de menu par SELECT DISTINCT (avec un cas
irrégulier à NULL). Conventions habituelles. Termine par uv run --no-sync pytest.
```

*Pourquoi Sonnet sans plan mode : un seul mécanisme (une colonne de metadata et ses
invariants), sans table auxiliaire ni choix d'API ouvert.*

---

## Prompt 5 — Écriture triée, options DuckLake et cycle de maintenance corrigé

**Modèle : Sonnet · Plan mode : OUI · Dépendances : prompt 2**

```text
Lis d'abord specification-bdd.md (sections 5.1 à 5.5 et annexe A). Les noms d'options
et signatures y sont MESURÉS sur la version installée (DuckDB 1.5.2) : appuie-toi
dessus plutôt que sur ta mémoire, et revérifie avant de coder
(SELECT * FROM ducklake_options('<alias>') ; SELECT function_name, parameters FROM
duckdb_functions() WHERE function_name LIKE 'ducklake%').

Faits à ne pas perdre de vue :
- la compaction actuelle de DatabaseUpdater._run_ducklake_compaction est un NO-OP :
  ducklake_rewrite_data_files sans delete_threshold explicite ne réécrit rien, et
  ducklake_merge_adjacent_files ne fusionne que des fichiers plus petits que
  min_file_size (entier en OCTETS ; '1KB' est rejeté) ;
- target_file_size exige une unité ('100MB') ;
- le data inlining est ACTIF PAR DÉFAUT : un petit INSERT ne produit aucun fichier
  Parquet ; ducklake_flush_inlined_data(catalog [, table_name, schema_name]) les écrit
  et retourne (schema, table, rows).

Implémente :

1) Tri à l'écriture (cluster_by). DuckLakeTablesBuilder.build_schema accepte
   cluster_by: list[str] | None ; défaut : les clés primaires dans leur ordre de
   déclaration ; les colonnes doivent exister (ValueError sinon). La valeur est
   persistée dans dataset_metadata.cluster_by (liste JSON). L'INSERT ... SELECT (et le
   chemin CTAS) se font avec ORDER BY correspondant. DataManager (_direct_insert_data,
   _batch_insert_data, chemin d'upsert) lit cluster_by dans dataset_metadata et trie
   chaque lot avant insertion. Une méthode update_cluster_by(columns) sur
   BaseSchemaManager modifie la valeur persistée (validation des colonnes) sans
   réécrire les données (le réordonnancement est le prompt 9).

2) Options DuckLake exposées sur DuckLakeConnector : paramètre optionnel
   ducklake_options: dict[str, str | int] | Literal["recommended"] | None appliqué
   après ATTACH via CALL <alias>.set_option(nom, valeur). Dictionnaire
   RECOMMENDED_DUCKLAKE_OPTIONS exporté par le module :
     parquet_compression = 'zstd'
     parquet_version = 2
     target_file_size = '100MB'
     parquet_row_group_size = 122880
   Ne mets PAS data_inlining_row_limit dans les défauts : expose-le comme argument dédié
   du connecteur, transmis comme option d'ATTACH (DATA_INLINING_ROW_LIMIT). Aucune
   option n'est appliquée sur une connexion read_only ; journalise chaque option
   effectivement positionnée.

3) Cycle de maintenance corrigé (maintenance/compaction.py) :
   - merge_files et rewrite_data_files acceptent leurs paramètres réels
     (min_file_size / max_file_size / max_compacted_files ; delete_threshold) avec des
     défauts explicites et documentés (delete_threshold = 0.1) ;
   - ces deux procédures RETOURNENT (schema_name, table_name, files_processed,
     files_created) : récupère-le, retourne-le et journalise les compteurs réels ; un 0
     doit être dit explicitement (« 0 fichier réécrit : seuil de suppression non
     atteint ») ;
   - nouvelle méthode flush_inlined_data(table=None) ;
   - nouvelle méthode delete_orphaned_files(older_than=None, dry_run=True) ;
   - expire_snapshots et cleanup_files retournent leurs résultats et exposent dry_run ;
   - DatabaseUpdater._run_ducklake_compaction et le chemin de DatabaseDeleter
     transmettent un delete_threshold et journalisent le résultat. Ils ne doivent JAMAIS
     appeler expire_snapshots, cleanup_old_files ni delete_orphaned_files : ces
     procédures détruisent le time travel ou sont irréversibles, et restent réservées à
     full_maintenance avec une rétention explicite ;
   - full_maintenance intègre flush_inlined_data avant merge ;
   - la docstring de module de compaction.py reprend le tableau « opération / quand /
     risque » de la section 5.5 de la spécification.

Tests. ATTENTION au data inlining : tout test inspectant les fichiers attache avec
DATA_INLINING_ROW_LIMIT 0 ou écrit un lot largement au-dessus de la limite. Couvre :
l'ordre physique d'insertion suit cluster_by (lecture du fichier via read_parquet et
contrôle de monotonie) ; cluster_by persisté et relu par DataManager ; les options sont
positionnées (ducklake_options(...) après connexion) ; une connexion read_only ne
tente pas de les appliquer ; rewrite_data_files avec un delete_threshold bas réécrit
effectivement après un UPDATE partiel, et ne réécrit rien avec le défaut du moteur ;
flush_inlined_data écrit un fichier et retourne le nombre de lignes. Complète le
notebook d'illustration des paramètres du builder. Conventions habituelles. Termine
par uv run --no-sync pytest.
```

*Pourquoi Sonnet + plan mode : tâche cadrée ; le plan sert à verrouiller les noms réels
après lecture de `ducklake_options` et `duckdb_functions` avant de coder.*

---

## Prompt 6 — Gestion explicite des colonnes de valeurs

**Modèle : Sonnet · Plan mode : non · Dépendances : prompts 3 et 5**

```text
Lis d'abord specification-bdd.md (section 4.2 et 4.3). Objectif : rendre explicite
l'ajout et la suppression de colonnes de valeurs de la fact_table, avec la table
metadata toujours synchronisée. La diffusion (broadcast) d'une valeur sur un
sous-ensemble de clés n'est PAS implémentée : un DataFrame porté par une autre clé est
un autre jeu de résultats (autre schéma) ; si l'utilisateur veut réellement diffuser,
il le fait dans son DataFrame avant d'appeler add_columns (recette documentée).

État actuel à connaître : DataManager._ensure_columns_exist ajoute silencieusement
toute colonne inconnue lors d'un insert/upsert ; DataManager.add_column et
drop_columns existent au niveau interne ; DatabaseDeleter.delete_columns existe.

1) Upsert : DatabaseUpdater.update_database gagne allow_new_columns: bool = False et
   column_metadata: dict[str, dict[str, str]] | None. Sans allow_new_columns, une
   colonne inconnue du DataFrame → ValueError listant les colonnes (plus d'ajout
   implicite). Avec, la colonne est ajoutée (type via map_python_to_sql_type), sa ligne
   metadata créée (is_primary_key = FALSE, is_categorical inféré, champs d'UI depuis
   column_metadata) et le tout journalisé.

2) DatabaseUpdater.add_columns(df, column_metadata=None, overwrite=False,
   compact_after_update=True) -> bool (OperationReport après le prompt 8) :
   - df doit porter TOUTES les clés primaires (ValueError listant les manquantes) et
     être unique sur celles-ci ;
   - les autres colonnes de df sont les colonnes à ajouter ; colonne déjà existante →
     ValueError sauf overwrite=True ;
   - ALTER TABLE ... ADD COLUMN <col> <type> DEFAULT NULL puis UN SEUL
     UPDATE fact f SET c1 = t.c1, ... FROM <vue temporaire> t WHERE f.k1 = t.k1 AND ...
     (identifiants quotés et qualifiés) ; ligne metadata ; dataset_metadata.updated_at ;
   - comptages journalisés : lignes mises à jour ; lignes de la base restées NULL ;
     combinaisons de df sans correspondance en base (warning avec échantillon, pas
     d'insertion) ;
   - transaction unique : en cas d'échec, ni la colonne ni la ligne metadata ne
     subsistent ;
   - un UPDATE qui touche toutes les lignes est une réécriture complète de la table
     (copy-on-write) : journalise le volume avant d'agir et termine par
     rewrite_data_files avec un delete_threshold bas.

3) DatabaseDeleter.delete_columns(columns, cascade=False) : ALTER TABLE ... DROP COLUMN
   (opération de métadonnées chez DuckLake, aucun fichier réécrit — mesuré), suppression
   de la ligne metadata, retrait de cluster_by, gestion des parent_name (prompt 4).
   Refus si clé primaire. Transaction unique.

4) Helper DatabaseUpdater.get_key_combinations(columns: list[str] | None = None)
   -> narwhals DataFrame des combinaisons distinctes de clés existantes (défaut :
   toutes les clés primaires), pour permettre la recette de diffusion explicite côté
   utilisateur : keys.join(df_partial, on=[...]) puis add_columns.

Tests pytest cas limites : colonne inconnue sans/avec allow_new_columns ; clés
manquantes dans add_columns ; doublons sur les clés ; colonne existante avec et sans
overwrite ; combinaisons absentes en base ; lignes de la base sans correspondance
(restent NULL) ; échec en milieu d'opération (la colonne ne subsiste pas) ;
delete_columns sur clé primaire, sur colonne de cluster_by, sur colonne parente ;
vérification que delete_columns ne change pas file_count. Complète le notebook
« Mise à jour de la base de données » avec l'ajout/suppression de colonnes et la recette
de diffusion explicite. Conventions habituelles. Termine par uv run --no-sync pytest.
```

---

## Prompt 7 — Simplification de la machinerie transactionnelle

**Modèle : Opus · Plan mode : OUI · Dépendances : prompts 2, 5 et 6**

```text
Lis d'abord specification-bdd.md (sections 1.5 et 7). Objectif : réduire la dette
technique de la couche transactionnelle en s'appuyant sur ce que DuckDB et DuckLake
fournissent nativement, sans changer l'API publique de DatabaseUpdater ni
DatabaseDeleter (signatures et sémantique de retour conservées).

Constats à vérifier puis traiter :
- TransactionManager réimplémente une gestion de transactions applicative
  (TransactionOperation, savepoints, états) au-dessus de BEGIN/COMMIT DuckDB, et la
  plupart des rollback_func passées par updater.py sont des placeholders qui ne
  restaurent rien (_rollback_fact_changes, _restore_database_state se contentent
  d'écrire une ligne de log).
- Le couple atomic.py (backups applicatifs) / recovery.py (points de restauration)
  recouvre le time travel DuckLake (snapshots + AT VERSION) déjà exposé par
  DuckLakeConnector.

Cible : chaque opération publique (update_database, add_columns, delete_rows,
delete_columns) devient un unique bloc BEGIN/COMMIT DuckDB avec ROLLBACK sur exception,
en conservant l'ordre des étapes, la validation par l'auditor et les logs ; les
classes devenues inutiles sont supprimées avec leurs tests ; ce qui reste utile de
recovery.py (list_ducklake_snapshots, restauration par snapshot) est conservé et
documenté comme LE mécanisme de récupération. Propose dans ton plan la liste exacte
des classes/méthodes supprimées et conservées, avec justification, avant d'implémenter.

Deux points à valider dans le plan :
- ce qui reste doit offrir un point d'accroche unique par opération (entrée/sortie de
  transaction) sur lequel le prompt 8 branchera la collecte du rapport et le message de
  commit ;
- la maintenance post-écriture (rewrite_data_files) s'exécute APRÈS le commit et ne
  doit pas être incluse dans la transaction.

Mets à jour mkdocs.yml et docs/api/ (pages mkdocstrings) pour retirer les classes
supprimées. Réécris les tests des chemins transactionnels (échec en milieu d'update →
la base revient à l'état initial, vérifié par comptage et contenu). Conventions
habituelles. Termine par uv run --no-sync pytest.
```

---

## Prompt 8 — Rapport d'opération, journalisation et traçabilité des runs

**Modèle : Sonnet · Plan mode : OUI · Dépendances : prompts 5, 6 et 7**

```text
Lis d'abord specification-bdd.md (sections 4.4 et 6, et annexe A). Objectif : rendre
explicite ce que chaque opération a réellement fait, et relier chaque écriture au run
qui l'a produite.

1) Dataclass OperationReport dans un nouveau module dt_ducklake_manager/reporting.py
   (champs de la section 6 de la spécification, dont run_id). Méthodes summary() -> str
   (ligne de synthèse lisible) et to_dict().

2) Collecte. Sources mesurées :
   - ducklake_table_info('<alias>') : file_count, file_size_bytes, delete_file_count,
     delete_file_size_bytes → avant et après l'opération ; fonction publique à préférer
     aux tables internes __ducklake_metadata_* ;
   - ducklake_snapshots('<alias>') : snapshot_id courant, colonne changes ;
   - ducklake_table_changes('<alias>', schema, table, start, end) : comptage exact par
     change_type entre le snapshot d'avant et celui d'après (insert, delete,
     update_preimage/update_postimage) — à utiliser pour rows_inserted/updated/deleted
     plutôt que des estimations Python ;
   - résultats retournés par merge/rewrite/flush/expire/cleanup (prompt 5).

3) Traçabilité des runs : update_database, add_columns, delete_rows, delete_columns et
   build_schema acceptent run_id: str | None = None, commit_message: str | None = None
   et commit_info: dict[str, Any] | None = None. Dans la transaction (avant le commit),
   exécute CALL ducklake_set_commit_message('<alias>', <run_id>, <message>,
   extra_info := <JSON>) où le JSON contient operation, schema, les champs de
   commit_info et, quand disponible, un résumé compact du rapport. Mesuré :
   ducklake_snapshots() expose ensuite author, commit_message, commit_extra_info. Sur
   une connexion sans DuckLake (in-memory des tests), l'appel est ignoré avec un DEBUG.
   Ajoute à list_ducklake_snapshots(...) (recovery.py) ces trois colonnes, pour répondre à
   « quel run a produit cet état ? » sans colonne technique dans la fact_table.

4) Branchement. update_database, add_columns, delete_rows, delete_columns,
   build_schema et DuckLakeMaintenance.full_maintenance construisent un
   OperationReport. L'API publique NE CHANGE PAS : update_database continue de
   retourner bool et expose le rapport via updater.last_report ; add_columns retourne
   directement son rapport.

5) Contrat de journalisation :
   - une ligne INFO de synthèse par opération réussie, du type
     « update main [run-42]: +1 240 lignes, ~380 mises à jour, +2 colonnes (score,
       rang), 3 → 2 fichiers (48,2 → 31,7 Mo), snapshot 17 → 18, 4,1 s » ;
   - le détail par étape en DEBUG, pas en INFO ;
   - les compteurs à zéro sont EXPLICITÉS : « rewrite_data_files : 0 fichier réécrit
     (seuil de suppression non atteint) » plutôt que « terminé » ;
   - chaque warning collecté dans report.warnings est aussi journalisé au fil de l'eau ;
   - en cas d'échec, un rapport partiel est journalisé en ERROR avec l'étape atteinte.

Tests pytest : un update dont on connaît le résultat produit les bons compteurs
(lignes par table_changes, colonnes, snapshots) ; une compaction sans effet est
journalisée comme telle ; summary() est stable et lisible ; un échec produit un
rapport partiel ; run_id et commit_message se retrouvent dans ducklake_snapshots.
Conventions habituelles. Termine par uv run --no-sync pytest.
```

---

## Prompt 9 — Réordonnancement physique et politique de maintenance

**Modèle : Opus · Plan mode : OUI · Dépendances : prompts 5 et 8**

```text
Lis d'abord specification-bdd.md (sections 5.3 et 5.5, annexe A). DuckLake n'a pas
d'index : l'élagage repose sur les statistiques min/max par fichier
(ducklake_file_column_stats, utilisées par le planificateur — « Total Files Read » dans
EXPLAIN ANALYZE) et par row group Parquet. Elles ne servent que si les données sont
physiquement groupées ; les updates successifs dégradent ce groupement. Objectif :
permettre de réordonner une table et de décider quand le faire.

1) DuckLakeMaintenance.recluster(table='fact_table', order_by=None, schema=None)
   -> OperationReport :
   - order_by par défaut : dataset_metadata.cluster_by du schéma ; colonnes validées ;
   - dans une transaction : CREATE TEMP TABLE _recluster AS SELECT * FROM <table> ;
     DELETE FROM <table> ; INSERT INTO <table> SELECT * FROM _recluster ORDER BY … ;
     DROP TABLE _recluster ; COMMIT. La table conserve son identité (id, partitionnement,
     historique) ; un DELETE intégral ne crée pas de fichier de suppression (mesuré) ;
   - FAIT MESURÉ DÉCISIF : avec plusieurs threads, DuckDB répartit les lignes triées
     entre les fichiers de façon non monotone (fichiers 0–999 et 409–819) ; avec
     SET threads = 1 le temps de l'INSERT, les fichiers sont disjoints et monotones
     (0–409, 409–819, 819–999). Positionne threads = 1 avant l'INSERT et restaure la
     valeur précédente dans un finally (SET n'est pas transactionnel) ;
   - après commit : merge_adjacent_files (petits fichiers ou fichiers vides
     résiduels) ; jamais expire/cleanup (réservés à la maintenance planifiée) ;
   - le rapport donne fichiers/octets avant-après et le recouvrement avant-après ;
   - documente dans la docstring le coût (réécriture complète, espace doublé jusqu'au
     cleanup) et le cas d'usage (après N updates, quand le recouvrement dépasse un
     seuil).

2) DuckLakeMaintenance.storage_report(table='fact_table', schema=None)
   -> StorageReport (dataclass) : file_count, total_bytes, delete_file_count,
   delete_bytes, delete_ratio, small_file_count (< min_file_size), inlined_rows
   (snapshots avec inlined_insert non flushés ou, à défaut, indicateur booléen),
   snapshot_count, oldest_snapshot_age_days, overlap_ratio (part des fichiers actifs
   dont la plage [min, max] de la première colonne de cluster_by chevauche celle d'un
   autre fichier ; calculé depuis __ducklake_metadata_<alias>.ducklake_data_file joint
   à ducklake_file_column_stats, end_snapshot IS NULL, avec CAST selon le type de la
   colonne ; NULL si cluster_by absent). Méthode summary().

3) MaintenancePolicy (dataclass, valeurs par défaut documentées) : delete_threshold
   (0.1), min_file_size_bytes, max_small_files, flush_inlined (True),
   max_overlap_ratio (0.5), recluster (False par défaut : opt-in), retention_days
   (None = ne jamais expirer), delete_orphaned (False), dry_run (False).
   DuckLakeMaintenance.maintain(policy, table='fact_table', schema=None)
   -> OperationReport : lit storage_report(), n'exécute chaque étape que si son
   indicateur le justifie, dans l'ordre flush → rewrite → merge → recluster →
   expire → cleanup → delete_orphaned, et journalise explicitement chaque étape sautée
   et pourquoi (« recluster ignoré : recouvrement 0,12 < 0,5 »). full_maintenance
   devient un appel à maintain avec une politique par défaut équivalente à l'existant
   (conserve sa signature).

Tests pytest (DATA_INLINING_ROW_LIMIT 0, target_file_size réduit pour obtenir
plusieurs fichiers) : après plusieurs inserts non triés, overlap_ratio élevé ; après
recluster, fichiers disjoints (vérifier via ducklake_file_column_stats) et
overlap_ratio nul ; nombre de lignes et contenu inchangés ; threads restauré même en
cas d'exception ; maintain avec une politique qui ne justifie rien n'exécute rien et le
journalise ; maintain avec retention_days=None n'appelle jamais expire ; dry_run.
Notebook : complète le notebook d'illustration de la maintenance (ou crée
notebooks/6 - Maintenance et réordonnancement.ipynb) montrant l'effet sur « Total
Files Read » d'une requête filtrée avant/après recluster. Conventions habituelles.
Termine par uv run --no-sync pytest.
```

*Pourquoi Opus + plan mode : opération de réécriture complète avec un réglage de session
non transactionnel, un indicateur à définir proprement et une politique dont les
déclencheurs méritent validation.*

---

## Prompt 10 — Documentation, notebooks, schéma draw.io, README, CLAUDE.md, skill

**Modèle : Sonnet · Plan mode : non · Dépendances : prompts 2 à 9**

```text
Lis d'abord specification-bdd.md en entier (la section 9 liste la documentation
attendue). Mets la documentation du dépôt en cohérence avec le schéma implémenté.
La documentation est en anglais (docs/, README.md) ; les notebooks en français.

1) Schéma draw.io. docs/assets/schema_bdd.png est un export draw.io « avec diagramme
   embarqué » : le XML mxfile est dans un chunk texte du PNG (clé « mxfile »,
   URL-encodé ; le contenu du <diagram> est compressé deflate+base64). Extrais-le
   (script Python jetable : lecture des chunks tEXt/zTXt, urllib.parse.unquote,
   zlib.decompress(..., -15) puis base64) et écris docs/assets/schema_bdd.drawio en
   clair (XML non compressé) — c'est désormais la source. Mets le diagramme à jour :
   trois tables par schéma (fact_table avec libellés, metadata avec toutes ses
   colonnes, dataset_metadata), plus aucune table de dimension, une flèche
   metadata.name → colonnes de fact_table, et une seconde zone montrant un catalogue
   avec deux schémas (predictions, shapley). Réexporte le PNG avec draw.io desktop :
   "C:\Program Files\draw.io\draw.io.exe" -x -f png -e -b 10 -o docs/assets/schema_bdd.png docs/assets/schema_bdd.drawio
   (-e embarque le diagramme dans le PNG). Si l'export échoue, laisse le .drawio à jour
   et signale-le en fin de réponse.

2) Nouvelles pages docs (à ajouter à la nav de mkdocs.yml, avant « API
   Documentation ») :
   - docs/schema.md « Schema and data model » : les trois tables (tableaux de
     colonnes), metadata comme contrat avec l'interface, statut catégoriel (inféré
     une seule fois à la création de la colonne, forcé à la construction, corrigé par
     update_column_metadata, jamais recalculé), hiérarchies de colonnes et convention NULL, types, cluster_by,
     traçabilité des runs (commit message, lecture par les snapshots), et une section
     « Design choices not retained » reprenant les sections 2.5 et 4.3 de la
     spécification (hiérarchies de valeurs, tables de libellés, diffusion sur clé
     partielle) avec les raisons ;
   - docs/maintenance.md « Storage lifecycle and maintenance » : copy-on-write et
     fichiers de suppression (avec la lecture brute trompeuse du répertoire), data
     inlining, tableau des opérations « effet / quand / risque », le cycle rewrite →
     merge → expire → cleanup, recluster et l'indicateur de recouvrement,
     MaintenancePolicy et maintain, puis des scénarios concrets avec le code
     correspondant : après un gros update ; après une suppression massive ; après N
     updates (recouvrement) ; libérer de l'espace (rétention) ; après une transaction
     interrompue (delete_orphaned_files en dry_run) ; retrouver le run à l'origine
     d'un état (snapshots) ; revenir en arrière (time travel). Termine par la lecture
     d'un OperationReport et d'un StorageReport.

3) docs/architecture.md : la section 1 (« Fact table granularity and dimension
   sharing ») décrit des tables de dimension qui n'existent plus ; réécris-la en
   « Result-set granularity » : un catalogue par projet, un schéma par jeu de
   résultats défini par sa clé, jointures sur les libellés, et pourquoi la diffusion
   de colonnes entre jeux de résultats n'est pas faite. Ajoute la règle multi-catalogues
   (lecture seule, jamais d'écriture couvrant deux catalogues, mesuré). Section 2
   conservée, débarrassée des mentions de dim_*.

4) docs/index.md et README.md : description des trois tables, image du schéma,
   exemple d'usage à jour (signatures réelles : connection=, cluster_by,
   column_metadata, hierarchies, run_id, allow_new_columns, MaintenancePolicy) et
   section « Organization » corrigée (logs sous le répertoire de travail). Le README
   renvoie vers les pages schema et maintenance du site.

5) docs/api/ et mkdocs.yml : pages mkdocstrings alignées sur les classes existantes
   (supprime celles des classes retirées, ajoute OperationReport, StorageReport,
   MaintenancePolicy, quote_ident, qualify_table, SchemaScoped, METADATA_COLUMNS,
   get_column_hierarchies, etc. ; BaseSchemaManager et DataManager sont désormais dans
   operations/_base.py et operations/_data.py).
   Vérifie que uv run --no-sync mkdocs build --strict passe.

6) Notebooks : passe en revue notebooks/0 à 6 et adapte-les (plus de
   categorical_threshold pilotant le stockage, plus de dim_*) ; assure-toi que la série
   couvre : construction simple ; column_metadata ; hiérarchies ; cluster_by et options
   DuckLake ; update avec nouvelles modalités (statut catégoriel stable, correction
   via update_column_metadata) et allow_new_columns ; add_columns en fusion externe
   (nouvelles clés insérées) et recette de diffusion explicite ; run_id et lecture des snapshots ; OperationReport ;
   storage_report, recluster et maintain. Chaque notebook illustre l'effet de la
   paramétrisation en variant les arguments. Exécute-les de bout en bout
   (uv run --no-sync jupyter nbconvert --execute --to notebook --inplace).

7) CLAUDE.md : section « Database structure » conforme à la section 2 de la
   spécification (trois tables, plus de tables de dimension, cluster_by, hiérarchies
   par parent_name), et retrait de la phrase « The dimension tables logic should change
   during this refactoring ».

8) Skill : mets à jour C:\Users\bolli\.claude\skills\dashboard-api-client\SKILL.md —
   sans modifier ce qui décrit l'API GraphQL actuelle — en ajoutant une section
   « Database schema (target for the API) » reprenant les sections 2 et 8 de
   specification-bdd.md : structure des tables, getSelectOptions par DISTINCT,
   suppression de dimensionDetails et getDimensionTable, getSelectOptionsTree (JSON)
   remplaçant getGroupedSelectOptions, nouveaux champs de getCatalogSchema, métadonnées
   de jeu de résultats dans getCatalogs. Cette section servira de contexte pour faire
   évoluer l'API dans son dépôt.

Conventions habituelles. Termine par uv run --no-sync mkdocs build --strict et
uv run --no-sync pytest, et liste ce qui n'a pas pu être fait (export PNG, notebook
en échec).
```

---

## Prompt 11 (optionnel) — Lecture multi-catalogues

**Modèle : Sonnet · Plan mode : non · Dépendances : prompts 1 et 10**

```text
Lis d'abord specification-bdd.md (section 7), qui contient les résultats mesurés :
plusieurs catalogues DuckLake peuvent être attachés à une même connexion et joints
entre eux, MAIS une transaction ne peut écrire que dans un seul catalogue attaché
(limite DuckDB : « a single transaction can only write to a single attached
database »).

Objectif : rendre l'usage multi-catalogues explicite et sûr, EN LECTURE SEULE.

1) DuckLakeConnector.attach(conn, activate_schema: bool = True) existe déjà (étape M) :
   vérifie qu'avec activate_schema=False aucun USE n'est exécuté.
2) Fonction utilitaire attach_read_only_catalog(conn, connector) qui attache un
   catalogue supplémentaire en READ_ONLY sans activer son schéma, et retourne l'alias.
3) Garde-fou : lever une erreur explicite si l'on tente d'instancier un DatabaseUpdater
   ou un DatabaseDeleter sur un alias attaché en lecture seule.
4) docs/architecture.md : complète la règle d'architecture (un catalogue par projet, un
   schéma par jeu de résultats, jamais d'écriture couvrant deux catalogues dans une même
   transaction ; copies inter-catalogues possibles mais séquentielles et non atomiques).

Tests : deux catalogues attachés, requête de jointure inter-catalogues ; vérification
qu'un manager configuré sur le catalogue A n'écrit jamais dans B quel que soit le
dernier USE ; refus d'écriture sur un alias read-only. Conventions habituelles.
Termine par uv run --no-sync pytest.

NON couvert par ce prompt, et à ne pas tenter : le partage d'une même base PostgreSQL
par plusieurs catalogues via l'option ATTACH METADATA_SCHEMA (non validée sur un
serveur réel).
```

---

## Ordre d'exécution et jalons

| # | Prompt | Modèle | Plan mode | Après |
|---|---|---|---|---|
| M | Intégration `DuckLakeConnector` (manuelle, faite) | — | — | — |
| 1 | Corrections préalables (types, quoting, qualification catalogue, index, logs) | Sonnet | non | M |
| 2 | Labels dans la fact table, suppression des dimensions, `dataset_metadata` | Opus | oui | 1 |
| 3 | Métadonnées d'UI | Sonnet | non | 2 |
| 4 | Hiérarchies de colonnes (`parent_name`) | Sonnet | non | 3 |
| 5 | Écriture triée (`cluster_by`), options DuckLake, cycle de maintenance corrigé | Sonnet | oui | 2 |
| 6 | Gestion explicite des colonnes | Sonnet | non | 3, 5 |
| 7 | Simplification transactionnelle | Opus | oui | 2, 5, 6 |
| 8 | Rapport d'opération, journalisation, traçabilité des runs | Sonnet | oui | 5, 6, 7 |
| 9 | Réordonnancement et politique de maintenance | Opus | oui | 5, 8 |
| 10 | Documentation, notebooks, draw.io, README, CLAUDE.md, skill | Sonnet | non | 2–9 |
| 11 | Lecture multi-catalogues (optionnel) | Sonnet | non | 1, 10 |

Différences avec la série précédente (`prompts-migration-schema-v2.md`) : plus de
migration v1 → v2 (supprimée, projet non publié) ; hiérarchies réduites au cas des
colonnes (plus de tables `dim_*` opt-in ni de `label_maps`) ; la diffusion sur clé
partielle remplacée par une gestion explicite des colonnes ; ajout de `cluster_by`,
`flush_inlined_data`, `delete_orphaned_files`, du message de commit DuckLake, de
`table_changes` pour les comptages, du réordonnancement (`recluster`), du
`storage_report` et de la `MaintenancePolicy` ; prompt de documentation étendu (schéma
draw.io, pages « schema » et « maintenance », README, CLAUDE.md).

Jalons de vérification entre prompts : suite de tests verte (`uv run --no-sync
pytest`), puis un commit par prompt (message `feat:`/`refactor:` conventionnel) pour
pouvoir revenir en arrière prompt par prompt. L'évolution de l'API GraphQL se fera
ensuite dans son propre dépôt, avec la section 8 de `specification-bdd.md` en contexte.
