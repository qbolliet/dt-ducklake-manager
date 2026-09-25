# Spécification de la base de données — schéma, écriture, stockage et maintenance

> **Statut** : document de référence pour l'implémentation de `dt_ducklake_manager` et
> pour la rédaction de sa documentation. Il remplace `revue-technique-bdd.md` (revue
> historique, conservée pour mémoire). Les prompts d'implémentation sont dans
> `prompts-refonte-schema.md`.
>
> **Version du schéma** : le schéma décrit ici est **la version 1** du schéma de la base et
> de l'API (le projet est en phase de développement ; l'état antérieur du code n'a jamais été
> publié et ne fait l'objet d'aucune migration). Le champ `dataset_metadata.schema_version`
> vaut `1` et n'existe que pour permettre une évolution future. L'ajout de
> `metadata.label_for` (§2.6, septembre 2026) reste dans la version 1 : les catalogues
> de développement construits auparavant sont reconstruits, sans migration.
>
> **Base empirique** : tous les comportements DuckLake cités comme *mesurés* l'ont été sur
> l'environnement du dépôt (DuckDB 1.5.2, extension `ducklake` 415a9ebd, catalogue fichier),
> en juillet-septembre 2026. Les commandes de vérification figurent en annexe A.

---

## 1. Principes de conception

1. **La table `metadata` est le contrat entre la base et l'interface.** Tout ce dont
   l'interface a besoin pour se piloter (libellé, type, statut catégoriel, hiérarchie, unité,
   format, famille, agrégation par défaut) est dans `metadata`, jamais déduit des données
   à l'exécution.
2. **La fact table stocke les libellés, pas des codes synthétiques.** Le
   dictionary-encoding Parquet absorbe le coût de stockage ; aucune table de dimension
   n'est nécessaire. Il n'existe **aucune table `dim_*`** dans le schéma. Quand un
   **code métier** doit être restitué tel quel (nomenclature, code INSEE), le code et
   son libellé sont deux colonnes de la fact table, reliées par `metadata.label_for`
   (§2.6).
3. **Un catalogue par projet, un schéma par jeu de résultats.** Un jeu de résultats est
   défini par sa clé (ex. `(date, region, produit)`) ; un résultat porté par une autre clé
   est un autre schéma, pas des colonnes supplémentaires.
4. **DuckLake est écrit par lots.** Jamais d'`INSERT` ligne à ligne ; les DataFrames sont
   enregistrés comme vues temporaires et écrits en une instruction.
5. **Le time travel est le mécanisme de récupération.** Pas de sauvegarde applicative ;
   l'expiration des snapshots est une opération de maintenance planifiée, avec rétention
   explicite, jamais un effet de bord d'une mise à jour.
6. **Pas d'index : l'ordre physique des données tient lieu d'index.** Tri à l'écriture sur
   les clés de filtre (`cluster_by`), réordonnancement périodique.
7. **Toute opération rend compte de ce qu'elle a fait** (rapport structuré, compteurs
   réels, zéros explicités) et peut être reliée au run qui l'a produite.

---

## 2. Schéma d'un jeu de résultats

Un catalogue DuckLake contient un ou plusieurs schémas ; chaque schéma contient exactement
trois tables.

### 2.1 `fact_table`

Les observations, une colonne par variable, types au plus juste (§3). Les colonnes
catégorielles contiennent **directement les libellés** d'origine. Pas de code synthétique,
pas de colonne technique ajoutée par le package. La table est :

- dédupliquée sur les clés primaires à la construction ;
- écrite **triée** selon `dataset_metadata.cluster_by` (§5.3) ;
- partitionnée (Hive) de façon optionnelle sur des colonnes de très faible cardinalité
  systématiquement filtrées (année, pays), jamais au point de produire des fichiers de
  moins de ~100 Mo.

### 2.2 `metadata` — une ligne par colonne de `fact_table`

| Colonne | Type | Nullable | Rôle |
|---|---|---|---|
| `name` | VARCHAR | non | Nom technique de la colonne |
| `label` | VARCHAR | non | Libellé d'affichage (défaut : `name`) |
| `sql_type` | VARCHAR | non | Type SQL DuckDB (`BIGINT`, `DOUBLE`, `VARCHAR`, …) ; sert au DDL et au mapping type → graphique côté front |
| `is_primary_key` | BOOLEAN | non | Fait partie de la clé logique (déduplication, upsert) |
| `is_categorical` | BOOLEAN | non | **Métadonnée d'UI pure** : la colonne se filtre par un menu et peut servir de `groupBy`. Ne pilote aucun choix de stockage |
| `parent_name` | VARCHAR | oui | Colonne parente dans une hiérarchie de colonnes (§2.5) |
| `label_for` | VARCHAR | oui | Renseigné sur une **colonne de libellés** : nom de la colonne de code dont elle porte le libellé de chaque valeur (§2.6) |
| `unit` | VARCHAR | oui | Suffixe d'axe / tooltip (`"€"`, `"%"`, `"MW"`) |
| `display_format` | VARCHAR | oui | Chaîne d3-format (`",.2f"`, `".0%"`) |
| `family` | VARCHAR | oui | Famille thématique (regroupement des variables dans les menus) |
| `description` | VARCHAR | oui | Aide contextuelle |
| `default_aggregation` | VARCHAR | oui | `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `MEDIAN`, `MODE` ; validé à l'écriture |

Règles :

- `python_type` n'existe pas (redondant avec `sql_type`). La résolution des conflits de
  types lors d'un update travaille sur les types SQL : `BOOLEAN < TINYINT < SMALLINT <
  INTEGER < BIGINT < FLOAT < DOUBLE < VARCHAR`, la largeur la plus grande étant conservée
  (un `BIGINT` enregistré n'est jamais rétrogradé par un lot d'`Int32`).
- `is_categorical` est **inféré une seule fois, à la création de la colonne** (colonne
  textuelle et nombre de modalités non nulles `<= categorical_threshold`) : à la
  construction, où il peut être forcé colonne par colonne (`categorical_overrides`), puis
  pour chaque colonne ajoutée (`allow_new_columns`, `add_columns`). Il n'est **jamais
  recalculé** par une écriture : un update ou une suppression qui fait varier le nombre
  de modalités ne le modifie pas. Il se corrige explicitement par
  `update_column_metadata(column, is_categorical=...)`. Une colonne appartenant à une
  hiérarchie est toujours catégorielle (`is_categorical=False` refusé). Motif : le
  booléen ne sert qu'à choisir le composant de filtre de l'interface (menu select ou
  champ de recherche) ; un statut qui bascule au gré des lots rendrait l'interface
  instable pour un gain nul.
- Les champs d'UI (`label`, `parent_name`, `label_for`, `unit`, `display_format`,
  `family`, `description`, `default_aggregation`) et `is_categorical` appartiennent au
  producteur de métadonnées : un update de données ne les écrase jamais. Ils se
  corrigent sur une base existante par `update_column_metadata(column, **fields)`.
- `label` et `label_for` ne se confondent pas : `label` est le libellé **de la
  colonne** (en-tête « Code NC8 ») ; `label_for` désigne la colonne dont on porte le
  libellé **de chaque valeur** (« 01012100 » → « Chevaux reproducteurs de race pure »).

### 2.3 `dataset_metadata` — une ligne par schéma

| Colonne | Type | Rôle |
|---|---|---|
| `label` | VARCHAR | Titre du jeu de résultats |
| `description` | VARCHAR | Sous-titre / description |
| `source` | VARCHAR | Provenance (modèle, pipeline) |
| `updated_at` | TIMESTAMP | Dernière écriture réussie (build, update, ajout/suppression de colonne) |
| `schema_version` | INTEGER | `1` |
| `cluster_by` | VARCHAR | Liste JSON des colonnes de tri physique (§5.3) ; défaut : clés primaires dans l'ordre de déclaration |

`updated_at` et `schema_version` sont toujours renseignés ; les autres champs sont fournis
par des arguments optionnels du builder.

### 2.4 Statut catégoriel et menus

`is_categorical` choisit le composant de filtre : menu select pour une colonne
catégorielle, champ de recherche sinon. L'interface obtient les modalités d'une colonne
catégorielle par `SELECT DISTINCT` sur la fact table (colonne dictionary-encodée, requête
bon marché), avec recherche et limite. `label = value`, sauf pour une colonne de code
dotée d'une colonne de libellés (§2.6) : `value` = code, `label` = libellé.

### 2.5 Hiérarchies : uniquement des hiérarchies de colonnes

Une hiérarchie (menu à group-options, arbre de sélection) est **une chaîne de colonnes** de
la fact table, déclarée par `metadata.parent_name` : la parente de `commune` est
`departement`, la parente de `departement` est `region`. Sans limite de profondeur. La
hiérarchie des *valeurs* est déjà dans la fact table : `SELECT DISTINCT region,
departement, commune` donne l'arbre complet sans aucune table auxiliaire.

Invariants validés à l'écriture : la colonne parente existe ; le graphe des `parent_name`
est une forêt (pas de cycle, une seule parente par colonne) ; une colonne d'une hiérarchie
est catégorielle.

**Convention pour les arbres irréguliers** (feuilles à des profondeurs différentes) : les
niveaux absents sont `NULL`. Le constructeur d'arbre s'arrête au premier niveau `NULL`
et l'API ignore les `NULL` dans le `DISTINCT`. On ne répète pas la valeur du niveau
supérieur (cela ferait apparaître un faux nœud).

**Ce qui n'est pas retenu, et pourquoi.**

- *Hiérarchie de valeurs dans une seule colonne (liste d'adjacence + chemin
  matérialisé, `dim_<col>(value, label, parent_value, path, depth)`)*. Elle ne serait
  préférable à l'aplatissement en colonnes que si (a) la profondeur est irrégulière et
  inconnue du producteur, ou (b) la taxonomie évolue indépendamment des faits et est
  maintenue comme référentiel externe. Dans un pipeline de résultats de modèles, le
  producteur contrôle le DataFrame et connaît ses niveaux : le cas (a) se traite par la
  convention `NULL` ci-dessus et le cas (b) n'existe pas. Le coût — une seconde table à
  maintenir en cohérence avec la fact table (valeurs orphelines, remplacement atomique,
  validation de cycles) — est exactement la complexité que la suppression des `dim_*`
  élimine. Si un projet en a besoin un jour, l'extension est locale : une table déclarée
  explicitement au build (`value_hierarchies={col: df}`), remplacée en bloc à chaque mise
  à jour, sans impact sur la fact table ni sur `metadata`.
- *Table de libellés `dim_<col>(value, label)` pour code métier ≠ libellé* : voir
  §2.6, qui traite ce cas par deux colonnes de la fact table et un lien dans
  `metadata`.

### 2.6 Codes et libellés : colonnes de libellés (`label_for`)

Certaines colonnes portent un **code métier** qui doit être restitué tel quel (code de
nomenclature tarifaire NC8, code INSEE, code pays ISO) et auquel on veut associer un
**libellé** lisible. Le code et le libellé sont **deux colonnes de la fact table** ; le
lien est déclaré dans `metadata.label_for`, **sur la colonne de libellés**, qui pointe
vers la colonne de code :

| `name` | `parent_name` | `label_for` |
|---|---|---|
| `nc6` | NULL | NULL |
| `nc6_libelle` | NULL | `nc6` |
| `nc8` | `nc6` | NULL |
| `nc8_libelle_fr` | NULL | `nc8` |
| `nc8_libelle_en` | NULL | `nc8` |

Le pointeur est porté par le libellé et non par le code pour deux raisons : c'est la
même forme que `parent_name` (l'enfant pointe vers sa cible : mêmes validations, même
cascade à la suppression), et un code peut avoir **plusieurs** colonnes de libellés
(langues, libellé court/long) sans changement de schéma. Chaque niveau d'une hiérarchie
de codes porte ses propres libellés ; la hiérarchie, elle, relie les codes.

**Invariants validés à l'écriture** (`ValueError` explicite, `ROLLBACK`) :

1. la colonne cible existe dans la fact table et diffère de la colonne de libellés ;
2. pas de chaîne : la cible n'est pas elle-même une colonne de libellés ;
3. la colonne de libellés est `VARCHAR`, n'est pas clé primaire et n'appartient à
   aucune hiérarchie (ni `parent_name` renseigné, ni parente d'une autre colonne) ;
4. **dépendance fonctionnelle code → libellé** : pour chaque valeur non `NULL` du code,
   la colonne de libellés porte **une seule** valeur sur toute la table, `NULL` compris
   (un code dont certaines lignes ont un libellé et d'autres non est incohérent) ; une
   ligne dont le code est `NULL` a un libellé `NULL`. Contrôle :

   ```sql
   SELECT code FROM fact_table WHERE code IS NOT NULL GROUP BY code
   HAVING COUNT(DISTINCT libelle) > 1 OR (COUNT(libelle) > 0 AND COUNT(libelle) < COUNT(*));
   SELECT COUNT(*) FROM fact_table WHERE code IS NULL AND libelle IS NOT NULL;
   ```

Le contrôle s'exécute au build (sur le DataFrame dédupliqué), à chaque update (état
post-upsert, **dans la transaction**, restreint aux codes présents dans le lot), à
chaque `add_columns` qui écrit une colonne de code ou de libellés, à la déclaration
d'un lien sur une base existante (`update_column_metadata(col, label_for=...)`, toute
la table) et après `update_value_labels`. Le message d'erreur liste au plus dix codes
fautifs avec leurs libellés concurrents.

Aucune contrainte sur `is_categorical` : un code à 10 000 modalités peut rester non
catégoriel (champ de recherche, dont l'autocomplétion cherche dans le code **et** le
libellé). Le statut de la colonne de libellés est sans objet pour l'interface, qui la
masque des listes de variables.

**Changement de libellé.** Une révision de nomenclature qui renomme un code viole la
dépendance si elle arrive par upsert (les lignes anciennes gardent l'ancien libellé) :
l'upsert est refusé et le message renvoie vers
`DatabaseUpdater.update_value_labels(label_column, labels)`, où `labels` est un
DataFrame `(code, libellé)`. L'opération est un unique `UPDATE … FROM` sur toutes les
lignes des codes concernés (réécriture copy-on-write de ces lignes, suivie de
`compact`), dans une transaction, avec `OperationReport` (`operation =
'update_value_labels'`). Le libellé affiché est donc toujours le libellé **courant** ;
les libellés antérieurs restent lisibles par time travel. Un code qui change de *sens*
d'un millésime à l'autre n'est pas un changement de libellé : le producteur intègre le
millésime au code ou garde une colonne ordinaire, sans `label_for`.

**Lecture.** `SELECT DISTINCT code, libelle` donne la table de correspondance ; un
agrégat groupé par code récupère le libellé par `ANY_VALUE(libelle)` (licite grâce à la
dépendance fonctionnelle) ; filtres et jointures entre jeux de résultats portent sur le
code. `get_value_label_columns(conn, schema, catalog_alias) -> dict[str, list[str]]`
(code → colonnes de libellés, triées par nom) est l'implémentation de référence pour
l'API, comme `get_column_hierarchies` pour les hiérarchies.

**Stockage.** Coût faible : le libellé est dictionary-encodé et parfaitement corrélé au
code. Réserve **non mesurée** : pour un code à forte cardinalité avec des libellés
longs (NC8, ~10 000 codes), le dictionnaire d'un row group peut dépasser le seuil
au-delà duquel l'écriture Parquet repasse en encodage plain ; la compression `zstd`
absorbe alors l'essentiel des répétitions. À mesurer sur un volume réel avant d'en
faire une recommandation.

**Ce qui n'est pas retenu, et pourquoi.**

- *Table de libellés `dim_<col>(code, label)`, ou catalogue de référentiels partagé
  entre projets.* Préférable seulement si le référentiel est maintenu indépendamment des
  modèles **et** que ses libellés doivent changer sans toucher aux faits. Le coût est
  celui que la suppression des `dim_*` élimine : jointure à chaque requête (et, pour un
  catalogue séparé, lecture multi-catalogues, §7), versionnage par millésime (la
  nomenclature NC change chaque année), codes orphelins, cohérence entre deux tables.
  Un référentiel partagé, s'il existe, s'utilise **en amont** : le producteur le joint à
  son DataFrame avant l'écriture.
- *Convention de nommage (`<col>_label`) sans métadonnée.* Contrat implicite que
  l'interface devrait deviner ; `label_for` le rend explicite pour le prix d'une
  colonne nullable.
- *Pointeur sur la colonne de code (`label_column`).* Limiterait chaque code à un seul
  libellé.

---

## 3. Types

Mapping DataFrame → SQL, largeurs préservées :

| DataFrame (narwhals) | SQL |
|---|---|
| `Boolean` | `BOOLEAN` |
| `Int8` / `Int16` / `Int32` / `Int64` | `TINYINT` / `SMALLINT` / `INTEGER` / `BIGINT` |
| `UInt8` / `UInt16` / `UInt32` / `UInt64` | `UTINYINT` / `USMALLINT` / `UINTEGER` / `UBIGINT` |
| `Float32` / `Float64` | `FLOAT` / `DOUBLE` |
| `String` / `Categorical` | `VARCHAR` |
| `Date` / `Datetime` | `DATE` / `TIMESTAMP` |

Le package ne réduit jamais une largeur silencieusement : c'est au producteur de fournir
un `Float32` s'il juge que 32 bits suffisent pour des valeurs destinées à des graphiques.

---

## 4. Opérations d'écriture

Toutes les écritures : identifiants **quotés** (`quote_ident`) et **qualifiés par le
catalogue** (`"catalog"."schema"."table"`, §7) ; DataFrames enregistrés comme vues
temporaires (non qualifiées) ; une transaction DuckDB (`BEGIN`/`COMMIT`, `ROLLBACK` sur
exception) par opération ; un `OperationReport` (§6) par opération ; `dataset_metadata.
updated_at` mis à jour **dans la transaction** (même snapshot, même `run_id` que
l'écriture) ; audit structurel `BASIC` de `DatabaseAuditor` avant le `COMMIT` (lecture du
catalogue et des petites tables de métadonnées uniquement, jamais de la table des faits ;
un problème critique annule l'écriture), réglable par `audit_level` (`COMPREHENSIVE`
ajoute les contrôles qui balaient la table des faits, `None` le désactive).

### 4.1 Construction (`DuckLakeTablesBuilder.build_schema`)

Entrées : DataFrame, `primary_keys`, `categorical_threshold`, `categorical_overrides`,
`column_labels`, `column_metadata` (champs d'UI par colonne), `hierarchies` (colonne →
parente), `value_labels` (colonne de libellés → colonne de code, §2.6), `partition_by`,
`cluster_by`, métadonnées de jeu de résultats (`label`, `description`, `source`).

Étapes : validation (clés présentes, `column_metadata` sur colonnes existantes, clés de
sous-dictionnaire autorisées, `default_aggregation` valide, forêt de `parent_name`,
invariants des colonnes de libellés dont la dépendance fonctionnelle) ;
déduplication ; création de `metadata` ; création de `fact_table` par
`INSERT … SELECT … ORDER BY cluster_by` (DDL explicite quand clés ou partition, CTAS
sinon) ; création de `dataset_metadata`.

### 4.2 Mise à jour (`DatabaseUpdater.update_database`)

Upsert sur les clés primaires (toutes requises, non nulles) en deux instructions : un
`UPDATE … FROM` restreint aux lignes existantes dont au moins une valeur change (une
ligne inchangée n'est pas réécrite, `rows_updated` ne compte que les lignes modifiées),
puis un `INSERT … SELECT … WHERE NOT EXISTS … ORDER BY cluster_by` du lot entier (pas de
découpage en lots Python : DuckDB traite déjà l'instruction en flux). Le lot est
dédupliqué **sur les clés primaires** selon `keep` ; un lot resté non unique est refusé.
Après l'écriture, l'unicité de la clé de la table des faits est contrôlée
(`check_duplicates_db`) : un doublon, qui ne peut venir que d'une écriture hors du
package, fait échouer l'update au lieu d'être supprimé. Nouvelles colonnes :
**refusées par défaut** ; acceptées avec `allow_new_columns=True` (ajout de la colonne,
ligne `metadata`, champs d'UI depuis `column_metadata`). Nouvelles modalités d'une
colonne catégorielle : rien à faire (ce sont des libellés), et `is_categorical` n'est
pas recalculé. Dépendance fonctionnelle des colonnes de libellés contrôlée sur l'état
post-upsert, dans la transaction, pour les codes du lot (§2.6) ; un changement de
libellé passe par `update_value_labels`. Les colonnes nouvelles (`allow_new_columns`)
sont ajoutées dans la transaction de l'update : un update refusé ne les laisse pas
derrière lui. Suivie de la compaction post-écriture (`compact` : fusion des petits
fichiers puis `rewrite_data_files(delete_threshold)`, §5.5), jamais d'une expiration de
snapshots.

### 4.3 Gestion explicite des colonnes

- `DatabaseUpdater.add_columns(df, column_metadata=None, overwrite=False)` : ajoute des
  colonnes de valeurs à partir d'un DataFrame portant **toutes les clés primaires**
  (non nulles, uniques). **Fusion externe sur les clés primaires** : combinaisons
  présentes des deux côtés → valeurs du DataFrame ; combinaisons du DataFrame absentes
  de la base → **lignes insérées** (clés + colonnes du DataFrame, autres colonnes de
  valeur à `NULL`) ; lignes de la base absentes du DataFrame → `NULL` dans les nouvelles
  colonnes (anciennes valeurs conservées dans les colonnes écrasées), comptées et
  journalisées. Colonne existante → erreur sauf `overwrite`. Réalisée par `ALTER TABLE …
  ADD COLUMN`, puis **un seul** `UPDATE … FROM` et **un seul** `INSERT … SELECT … ORDER
  BY cluster_by`, dans une transaction ; le rapport distingue `rows_updated` et
  `rows_inserted`.
- `DatabaseDeleter.delete_columns(columns)` : `ALTER TABLE … DROP COLUMN` (*mesuré* :
  opération de métadonnées, aucun fichier réécrit) + suppression de la ligne `metadata` +
  retrait de la colonne de `cluster_by` et des `parent_name` qui la référencent (erreur si
  elle est clé primaire ou parente d'une autre colonne, sauf `cascade`). Même règle pour
  une colonne de code visée par des colonnes de libellés : erreur, sauf `cascade`, qui
  remet leur `label_for` à `NULL` (elles deviennent des colonnes ordinaires) avec un
  warning. Supprimer une colonne de libellés ne demande rien de particulier.
- `DatabaseUpdater.update_value_labels(label_column, labels)` : remplacement explicite
  des libellés de certains codes (§2.6).

**Non retenu : diffusion (broadcast) d'une colonne sur un sous-ensemble de clés.** Un
DataFrame portant `(region, produit, score)` face à une fact table de clé `(date, region,
produit)` est **un autre jeu de résultats** (autre clé) : il va dans un autre schéma, que
l'API joint ou affiche séparément. Dupliquer `score` sur chaque `date` dénormalise la
valeur (les lignes insérées plus tard n'en héritent pas, la valeur diverge de sa source),
cache une réécriture complète de la fact table et rend implicite un produit cartésien.
Si un utilisateur veut réellement la diffusion, il la fait explicitement dans son
DataFrame (jointure avec les combinaisons de clés existantes, obtenues par
`updater.get_key_combinations(columns)`) puis appelle `add_columns`. La documentation
donne cette recette.

### 4.4 Traçabilité des runs

Chaque opération d'écriture accepte `run_id: str | None` et `commit_message: str | None`
et les enregistre dans le snapshot DuckLake via
`CALL ducklake_set_commit_message(catalog, author := run_id, message, extra_info :=
<JSON>)` **dans la transaction** (*mesuré* : `ducklake_snapshots()` expose ensuite
`author`, `commit_message`, `commit_extra_info`). `extra_info` porte au minimum
`operation`, `schema`, et les champs libres passés par l'appelant (`model_version`, …).
Aucune colonne technique n'est ajoutée à la fact table : la question « quel run a produit
cet état ? » se répond par les snapshots.

### 4.5 Suppression

`DatabaseDeleter.delete_rows(conditions)` : `DELETE` par lot ; suivi d'un
`rewrite_data_files(delete_threshold)`. Suppression d'un schéma entier : `DROP SCHEMA …
CASCADE` puis maintenance planifiée pour libérer les fichiers.

---

## 5. Stockage physique DuckLake

### 5.1 Copy-on-write (*mesuré*)

Les fichiers Parquet sont immuables. Un `UPDATE` de 5 000 lignes sur un fichier de 20 000
produit : un nouveau fichier de 5 000 lignes (avec `_ducklake_internal_row_id`), un fichier
`…-delete.parquet` de 5 000 positions, et l'ancien fichier laissé intact (référencé par le
snapshot précédent). Une lecture brute du répertoire (`read_parquet('**/*.parquet')`)
montre donc 30 000 lignes de trois schémas différents : c'est un artefact de la lecture
brute, la vue logique est correcte à tout instant.

Un `DELETE` qui vide entièrement un fichier ne crée **pas** de fichier de suppression : le
fichier est simplement retiré du snapshot courant (*mesuré* : `delete_file_count = 0`
après `DELETE FROM t` sans clause).

### 5.2 Data inlining (*mesuré*)

Actif par défaut : un petit `INSERT` ne produit **aucun** fichier Parquet, les lignes
vivent dans le catalogue (`{inlined_insert=[1]}`). Réglage : option d'`ATTACH`
`DATA_INLINING_ROW_LIMIT n` (`0` = jamais) ou `set_option('data_inlining_row_limit')`.
Les lignes inlinées sont écrites en Parquet par `ducklake_flush_inlined_data(catalog
[, table_name, schema_name])` (*mesuré* : retourne `(schema, table, rows)` et crée un
snapshot `{flushed_inlined=[1]}`). Politique : `0` au build initial ; quelques milliers en
update incrémental ; flush dans la maintenance planifiée. Tout test qui inspecte les
fichiers attache avec `DATA_INLINING_ROW_LIMIT 0`.

### 5.3 Tri physique et élagage

DuckLake enregistre, **par fichier et par colonne**, `min_value`/`max_value`
(`ducklake_file_column_stats`) et les utilise pour ne pas ouvrir les fichiers hors plage
(*mesuré* : `EXPLAIN ANALYZE` affiche `Total Files Read`). À l'intérieur d'un fichier,
les statistiques de row group Parquet font le même travail. Les deux ne servent que si les
données sont physiquement groupées : d'où `cluster_by`.

- **`cluster_by`** (dataset_metadata) : colonnes de tri, défaut = clés primaires dans
  l'ordre de déclaration, à ordonner de la plus sélective à la moins sélective. Build et
  lots d'update sont écrits `ORDER BY cluster_by`.
- **Dégradation** : chaque update écrit ses lignes dans de nouveaux fichiers, triés dans
  le lot mais pas fusionnés dans l'ordre global. Les plages des fichiers se recouvrent
  progressivement.
- **Réordonnancement** (`DuckLakeMaintenance.recluster(table, order_by=None)`) : dans une
  transaction, matérialisation dans une table temporaire, `DELETE FROM table`, `INSERT
  INTO table SELECT * FROM tmp ORDER BY …`. **Point mesuré et décisif** : avec plusieurs
  threads, DuckDB répartit les lignes triées entre plusieurs fichiers de façon non
  monotone (fichiers 0–999 et 409–819) ; avec `SET threads = 1` le temps de
  l'insertion, les fichiers sont disjoints et monotones (0–409, 409–819, 819–999). Le
  réglage est restauré en `finally`. C'est une réécriture complète : à suivre d'un
  `merge_adjacent_files` (petits fichiers résiduels) et, en maintenance planifiée,
  d'`expire_snapshots` + `cleanup_old_files`. Un indicateur de **recouvrement** (part des
  fichiers dont la plage de la première colonne de `cluster_by` chevauche celle d'un
  autre) permet de décider quand réordonner (`DuckLakeMaintenance.storage_report()
  .overlap_ratio`). Définition retenue (*mesurée*) : fichiers actifs non vides à
  statistiques non nulles, bornes converties au type de la colonne, chevauchement en
  **inégalités strictes** (`a.min < b.max AND b.min < a.max`) ou plages identiques. Un
  réordonnancement avec clés dupliquées produit des fichiers partageant une borne
  (0–409, 409–819) : avec des inégalités larges, le recouvrement ne tomberait jamais à
  0. La table temporaire de `recluster` peut être créée dans la même transaction que
  le `DELETE`/`INSERT` DuckLake (*mesuré*, DuckDB 1.5.2) ; `threads` est restauré après
  `COMMIT`/`ROLLBACK`.
- **Partitionnement** : complémentaire, réservé aux colonnes de très faible cardinalité ;
  `set_partitioned_by` n'affecte que les écritures futures, `repartition` réécrit.

### 5.4 Options DuckLake (*mesuré*)

`CALL <alias>.set_option(name, value [, table_name := …, schema := …])` ; lecture par
`ducklake_options('<alias>')`.

| Option | Valeurs | Recommandation |
|---|---|---|
| `parquet_compression` | `snappy`, `zstd`, `gzip`, … | `zstd` |
| `parquet_version` | `1`, `2` | `2` |
| `target_file_size` | taille **avec unité** (`'100MB'`) ; sans unité : erreur | `'100MB'` |
| `parquet_row_group_size` | entier (lignes) | 122 880 (déjà le défaut effectif de DuckDB, *mesuré* ; explicité pour rendre visible la granularité de l'élagage par row group) |
| `data_inlining_row_limit` | entier | selon profil (§5.2), pas de défaut imposé |
| `encrypted` | `true`/`false` | selon déploiement |

Le connecteur accepte `ducklake_options` (dict ou `"recommended"`), n'applique rien sur
une connexion `read_only`, et journalise chaque option positionnée.

### 5.5 Opérations de maintenance : quoi, quand, risque

Signatures *mesurées* (`duckdb_functions()`) :

```
ducklake_flush_inlined_data(catalog [, table_name, schema_name])          -> (schema, table, rows)
ducklake_rewrite_data_files(catalog, table, delete_threshold, schema)    -> (schema, table, files_processed, files_created)
ducklake_merge_adjacent_files(catalog, table, min_file_size, max_file_size, max_compacted_files, schema) -> idem
ducklake_expire_snapshots(catalog, older_than, versions, dry_run)         -> snapshots périmés
ducklake_cleanup_old_files(catalog, older_than, cleanup_all, dry_run)     -> chemins supprimés
ducklake_delete_orphaned_files(catalog, older_than, cleanup_all, dry_run) -> chemins supprimés
ducklake_table_info(catalog)   -> table_name, schema_id, table_id, table_uuid, file_count, file_size_bytes, delete_file_count, delete_file_size_bytes
ducklake_snapshots(catalog)    -> snapshot_id, snapshot_time, schema_version, changes, author, commit_message, commit_extra_info
ducklake_table_changes(catalog, schema, table, start_snapshot, end_snapshot) -> snapshot_id, rowid, change_type, <colonnes>
ducklake_list_files(catalog, table [, schema, snapshot_version, snapshot_time])
```

Attention aux unités : `target_file_size` exige une unité, `min_file_size` /
`max_file_size` de `merge_adjacent_files` sont des **entiers en octets** (*mesuré* :
`'1KB'` est rejeté). `min_file_size` est une borne **basse** (*mesuré*) : les fichiers
*plus petits* que ce seuil sont exclus de la fusion ; il n'est donc pas transmis, et
`max_file_size` vaut le `target_file_size` du catalogue. Le résultat de
`merge_adjacent_files` et de `rewrite_data_files` doit être **lu en entier**
(`fetchall`) : lu partiellement (`fetchone`), la procédure renvoie ses compteurs mais
n'est pas validée et reste sans effet (*mesuré*).

| Opération | Effet | Quand | Risque |
|---|---|---|---|
| `rewrite_data_files(delete_threshold)` | Réécrit les fichiers dont la part de lignes supprimées dépasse le seuil ; **sans seuil explicite, ne fait rien** (*mesuré*, même à 25 % de suppressions) | Après chaque update / delete (`delete_threshold` 0,1–0,3) | Aucun (les anciens fichiers restent lisibles par time travel) |
| `merge_adjacent_files(max_file_size)` | Fusionne les fichiers adjacents en fichiers d'au plus `max_file_size` octets (`target_file_size` du catalogue) | Après chaque écriture (`compact`) ; après `recluster` | Aucun |
| `flush_inlined_data` | Écrit en Parquet les lignes inlinées dans le catalogue | Maintenance planifiée ; avant une lecture directe des fichiers | Aucun |
| `recluster(order_by)` | Réécrit la table entière dans l'ordre de `cluster_by` | Quand le recouvrement des fichiers dégrade l'élagage (indicateur §5.3), typiquement après N updates | Réécriture complète ; double l'espace jusqu'au cleanup |
| `repartition` | Change le partitionnement et réécrit | Changement de stratégie de filtre | Idem |
| `expire_snapshots(older_than)` | Rend inaccessibles les snapshots antérieurs | **Maintenance planifiée uniquement**, rétention explicite (jours) | **Détruit le time travel** au-delà de la rétention |
| `cleanup_old_files` | Supprime les fichiers que plus aucun snapshot ne référence | Après `expire_snapshots` ; seule étape qui libère l'espace (*mesuré*) | Irréversible |
| `delete_orphaned_files` | Supprime les fichiers du `data_path` inconnus du catalogue | Après un incident (transaction interrompue, copie manuelle) ; toujours en `dry_run` d'abord | Irréversible |

Cycle : *réécrire* (rewrite / merge / flush) après les écritures → *périmer* (expire) et
*supprimer* (cleanup) en maintenance planifiée. Une **politique de maintenance**
(`MaintenancePolicy`) regroupe les seuils (`delete_threshold`, `min_file_size_bytes` —
seuil de *comptage* des petits fichiers, qui décide si la fusion s'exécute —,
`retention_days`, `max_overlap_ratio`, `flush_inlined`) et `DuckLakeMaintenance.
maintain(policy)` ne déclenche chaque étape que si son indicateur (`storage_report()`)
le justifie, en journalisant les étapes sautées et pourquoi :

| Étape (ordre) | Exécutée si |
|---|---|
| `flush_inlined_data` | `flush_inlined` et lignes inlinées non vidangées |
| `rewrite_data_files` | au moins un fichier de suppression (le seuil par fichier est appliqué par DuckLake) |
| `merge_adjacent_files` | `small_file_count > max_small_files` |
| `recluster` | `recluster=True` (opt-in) et `overlap_ratio > max_overlap_ratio` |
| `expire_snapshots`, puis `cleanup_old_files` | `retention_days` non `None` |
| `delete_orphaned_files` | `delete_orphaned=True` |

`dry_run` : les étapes qui modifient (flush, rewrite, merge, recluster) sont seulement
journalisées ; expire/cleanup/orphaned sont appelés en `dry_run` (listage).
Il n'existe pas de raccourci `full_maintenance` : une maintenance planifiée s'écrit
`maintain(MaintenancePolicy(retention_days=30))` (ajouter `max_small_files=1` pour
fusionner dès deux petits fichiers). La compaction légère exécutée après chaque écriture
(merge + rewrite, jamais d'expiration) est `DuckLakeMaintenance.compact(table, schema,
delete_threshold)`. Lignes inlinées mesurées via
`ducklake_inlined_data_tables` (lignes à `end_snapshot IS NULL`, vidées par le flush) ;
âge des snapshots calculé en SQL (`epoch`), la lecture d'un `TIMESTAMPTZ` en Python
exigeant `pytz` (*mesuré*).

---

## 6. Rapport d'opération et journalisation

```python
@dataclass
class OperationReport:
    operation: str                  # 'build' | 'update' | 'add_columns' | 'update_value_labels' | 'delete_columns' | 'delete_rows' | 'recluster' | 'maintenance' | 'restore_snapshot'
    schema: str
    run_id: str | None
    started_at: datetime
    duration_seconds: float
    rows_inserted: int
    rows_updated: int
    rows_deleted: int
    rows_before: int
    rows_after: int
    columns_added: list[str]
    columns_dropped: list[str]
    metadata_changes: list[str]     # "is_categorical(region): False -> True"
    snapshot_before: int | None
    snapshot_after: int | None
    files_before: int
    files_after: int
    bytes_before: int
    bytes_after: int
    maintenance: dict[str, int | float]  # files_processed / files_created / rows_flushed par procédure ; <étape>_skipped ; overlap_ratio_before/after (recluster)
    warnings: list[str]
```

Sources (*mesurées*) : `ducklake_table_info` avant/après ; `ducklake_snapshots` (id,
`changes`) ; résultats retournés par les procédures de maintenance ;
`ducklake_table_changes(start, end)` pour les comptages exacts par type de changement.

Contrat : une ligne INFO de synthèse par opération (`update main [run-42]: +1 240 lignes,
~380 mises à jour, +2 colonnes (score, rang), 3 → 2 fichiers (48,2 → 31,7 Mo), snapshot
17 → 18, 4,1 s`) ; détail en DEBUG ; zéros explicités (`rewrite_data_files : 0 fichier
réécrit (seuil non atteint)`) ; warnings journalisés au fil de l'eau ; rapport partiel en
ERROR en cas d'échec avec l'étape atteinte. `update_database` continue de retourner
`bool` et expose `updater.last_report` ; les nouvelles méthodes retournent le rapport.

Erreurs (arbitrage entre §2.6 et le contrat `bool`) : une erreur de **saisie** (clé
primaire manquante ou nulle, colonne inconnue, clé dupliquée dans un lot, dépendance
code → libellé violée, filtre de suppression absent ou mal formé) lève une `ValueError`
explicite, dans toutes les opérations, `update_database` compris, après `ROLLBACK` si
une transaction était ouverte. Une erreur d'**exécution** (erreur de la base) est
annulée elle aussi ; `update_database` retourne alors `False` (le message est dans
`last_report.warnings`), les autres opérations relancent l'exception d'origine.

---

## 7. Identifiants, catalogue et connexion

- `quote_ident(name)` : guillemets doubles, guillemets internes doublés. Utilisé pour
  tout identifiant issu des données.
- `qualify_table(table, schema, catalog)` : `"catalog"."schema"."table"` ; `"schema".
  "table"` quand `catalog` est `None` (connexions in-memory des tests). Sans qualification
  par le catalogue, une requête se résout dans le catalogue **courant** de la connexion
  (dernier `USE`), ce qui écrirait silencieusement dans la mauvaise base dès qu'un second
  catalogue est attaché (*mesuré*).
- Ne sont **jamais** qualifiés : les vues temporaires enregistrées par `conn.register()`
  et les table functions `ducklake_*`.
- Multi-catalogues : plusieurs `ATTACH` cohabitent et se joignent (*mesuré*), mais une
  transaction ne peut écrire que dans un seul catalogue (*mesuré* : `a single transaction
  can only write to a single attached database`). Règle : lecture multi-catalogues
  explicite et `READ_ONLY`, `attach(activate_schema=False)` pour ne pas voler le
  catalogue courant, jamais d'écriture couvrant deux catalogues.
- Transactions : un `BEGIN`/`COMMIT` DuckDB par opération, `ROLLBACK` sur exception ;
  pas de machinerie applicative (savepoints, opérations enregistrées, sauvegardes). La
  maintenance post-écriture s'exécute **après** le commit.
- Logs : fichiers sous `Path.cwd() / "logs"` par défaut (jamais dans le répertoire du
  package), logger nommé, un seul handler.

---

## 8. Impacts sur l'API GraphQL (dépôt `dashboard-template-api`)

- `getCatalogs` expose `dataset_metadata` (`label`, `description`, `source`,
  `updatedAt`, `schemaVersion`).
- `getCatalogSchema` / `getFields` exposent `parentName`, `labelFor`, `unit`,
  `displayFormat`, `family`, `description`, `defaultAggregation`, plus le champ calculé
  `labelFields` (inverse de `labelFor` : colonnes de libellés d'un code, triées par nom,
  lues dans les mêmes lignes `metadata`, sans requête supplémentaire).
- `getFields` exclut par défaut les colonnes de libellés (argument
  `includeLabelFields: Boolean = false`) : ce ne sont pas des variables à proposer dans
  un menu. `getCatalogSchema` rend toutes les colonnes (contrat complet).
- `getSelectOptions(field, searchTerm, limit, labelField)` : `SELECT DISTINCT` sur la
  fact table, `label = value`, sauf colonne de code dotée de libellés (§2.6) :
  `value` = code, `label` = libellé. Colonne de libellés utilisée : `labelField` si
  fourni (doit viser `field`), sinon la seule colonne de libellés, sinon la première
  par ordre alphabétique. `searchTerm` cherche dans le code **ou** le libellé. Reste
  l'endpoint typé pour la recherche et la pagination sur un niveau (ex. 35 000
  communes).
- **`getSelectOptionsTree(field, maxDepth, searchTerm): JSON`** remplace
  `getGroupedSelectOptions` : arbre imbriqué `[{value, label, children: [...]}]` construit
  par `SELECT DISTINCT` sur la chaîne de colonnes remontée via `parentName`, `NULL`
  terminant une branche. Le scalaire JSON est retenu comme **seule** forme hiérarchique :
  GraphQL n'exprime pas une profondeur arbitraire, les menus sont bornés par nature, et
  le typage fort qu'apporterait une liste plate n'a pas de consommateur (la liste plate
  n'aurait d'intérêt que pour un chargement paresseux de sous-arbres très grands, cas
  couvert par `searchTerm` et `getSelectOptions` sur le niveau feuille). Chaque niveau
  prend son libellé dans sa colonne de libellés quand elle existe (même règle de choix
  par défaut que `getSelectOptions`).
- `AggregatedFact.keyLabel` et `ComparedFact.keyLabel` : libellé de la clé de groupe
  quand la colonne groupée a une colonne de libellés (`ANY_VALUE`, même requête),
  `null` sinon.
- `dimensionDetails`, `getDimensionTable` : **supprimés** (pas de dépréciation, le
  projet n'est pas publié).
- `compareFacts` : jointure directe sur les colonnes partagées (libellés, ou codes
  quand ils existent). `getSharedFields` exclut les colonnes de libellés.
- Profondeur maximale de requête (7) et pagination par offset (10 000) : limites de
  conception à documenter.

---

## 9. Documentation attendue

La documentation (`docs/`, MkDocs Material) doit contenir, en plus de la référence API
générée :

1. **Description du schéma** (`docs/index.md`, `README.md`) : les trois tables, le
   contrat `metadata`, l'absence de tables de dimension ; schéma draw.io
   (`docs/assets/schema_bdd.drawio` + export PNG) à jour.
2. **Fonctionnement du schéma** (nouvelle page) : statut catégoriel, hiérarchies de
   colonnes et convention `NULL`, codes et colonnes de libellés (`label_for`, dépendance
   fonctionnelle, changement de libellé), types, `cluster_by`, traçabilité des runs, ce
   qui n'est pas retenu et pourquoi (§2.5, §2.6, §4.3).
3. **Cycle de vie physique et maintenance** (nouvelle page) : copy-on-write, inlining,
   tableau des opérations avec « quand » et « risque » (§5.5), exemples concrets : après un
   gros update, après une suppression massive, après N updates (recouvrement), avant de
   libérer de l'espace, après un incident ; politique de rétention ; lecture d'un
   `OperationReport`.
4. **Architecture** (`docs/architecture.md`, existante) : mise à jour de la partie
   « axe logique » (plus de dimensions, redondantes ou conformes) ; règle un catalogue par
   projet / un schéma par jeu de résultats ; limites multi-catalogues.
5. `CLAUDE.md` : section « Database structure » conforme au §2.

---

## 10. Corrections embarquées dans la refonte

1. Mapping de types (§3) ; 2. suppression de la logique d'index (inopérante sous
DuckLake) ; 3. `quote_ident` + qualification par le catalogue (§7) ; 4. options DuckLake et
cycle de maintenance corrigés (§5) ; 5. logs hors du package ; 6. simplification
transactionnelle (§7) ; 7. suppression des `INSERT` ligne à ligne.

---

## Annexe A — vérifications reproductibles

```sql
-- Fonctions et signatures réelles
SELECT function_name, parameters FROM duckdb_functions() WHERE function_name LIKE 'ducklake%';
-- Options et valeurs courantes
SELECT * FROM ducklake_options('db');
-- Statistiques par fichier (élagage) : plages min/max de la colonne 1
SELECT f.data_file_id, f.record_count, s.min_value, s.max_value
FROM __ducklake_metadata_db.ducklake_data_file f
JOIN __ducklake_metadata_db.ducklake_file_column_stats s USING (data_file_id)
WHERE s.column_id = 1 AND f.end_snapshot IS NULL;
-- Fichiers effectivement lus par une requête filtrée
EXPLAIN ANALYZE SELECT count(*) FROM db.main.fact_table WHERE k = 5;   -- « Total Files Read »
-- Message de commit dans la transaction
BEGIN; CALL ducklake_set_commit_message('db', 'run-42', 'update', extra_info := '{"model_version":"1.3"}'); ...; COMMIT;
SELECT snapshot_id, changes, author, commit_message, commit_extra_info FROM ducklake_snapshots('db');
-- Flux de changements entre deux snapshots
SELECT change_type, count(*) FROM ducklake_table_changes('db', 'main', 'fact_table', 17, 18) GROUP BY 1;
```

Résultats observés (juillet–septembre 2026, DuckDB 1.5.2) :

| Expérience | Observation |
|---|---|
| `INSERT` 20 000 lignes puis `UPDATE` de 5 000 | 1 fichier + 1 fichier de 5 000 + 1 `-delete.parquet` de 5 000 ; lecture brute : 30 000 lignes |
| `rewrite_data_files` sans `delete_threshold` | aucun fichier réécrit |
| `rewrite_data_files(delete_threshold := 0.1)` | 1 fichier de 15 000 lignes créé, ancien conservé |
| `expire_snapshots` + `cleanup_old_files(cleanup_all)` | répertoire ramené à 20 000 lignes sans doublon |
| `INSERT` de 4 lignes, inlining par défaut | aucun Parquet, snapshot `{inlined_insert=[1]}` |
| `flush_inlined_data` | `(main, t, 2)`, snapshot `{flushed_inlined=[1]}` |
| Recluster multi-threads (300 000 lignes, `target_file_size` 200 KB) | fichiers 0–999 et 409–819 : plages recouvrantes |
| Recluster `threads = 1` | fichiers 0–409, 409–819, 819–999 ; `Total Files Read: 2` (dont un fichier vide résiduel) |
| `DELETE FROM t` sans clause | `delete_file_count = 0` : fichiers retirés du snapshot, pas de fichier de suppression |
| `UPDATE` partiel puis recluster (`DELETE` intégral + `INSERT`) | l'ancien `-delete.parquet` garde `end_snapshot IS NULL` et reste compté par `ducklake_table_info` alors que son fichier de données est inactif ; `rewrite_data_files` n'y change rien. `storage_report` ne compte que les fichiers de suppression visant un fichier de données actif |
| Recluster, `CREATE TEMP TABLE` dans la transaction DuckLake | accepté (le catalogue `temp` ne compte pas comme seconde base écrite) |
| Recouvrement après recluster `threads = 1` | fichiers 0–409, 409–819, 819–999 : borne partagée → comparaison stricte ; `Total Files Read` 4 → 1 pour `k = 5` (notebook 6) |
| `ALTER TABLE … ADD/DROP COLUMN` | `file_count` et `file_size_bytes` inchangés : métadonnées seules |
| `ALTER TABLE … RENAME TO` | supporté |
| `merge_adjacent_files(min_file_size := '1KB')` | erreur `Could not convert string '1KB' to UINT64` : entier en octets |
| Deux catalogues attachés, `UPDATE` sur chacun dans une transaction | `a single transaction can only write to a single attached database` |
