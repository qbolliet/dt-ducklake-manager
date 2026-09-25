# Revue d'implémentation — fin de série (prompts 1 à 13)

> **Statut** : diagnostic du TEMPS 1 du prompt 14. Aucun fichier du package n'a été
> modifié. Rien ne sera corrigé avant validation : le panier A est proposé à
> l'application, le panier B attend une décision par ligne, le panier C est clos.
> Référence normative : `specification-bdd.md`.

## 0. Méthode

**Environnement.** DuckDB 1.5.2 et extension `ducklake` (celle de `.venv` ; `pyproject`
exige `duckdb>=1.5.3`, voir C6), Windows 11, catalogue fichier sous
`outputs/bench_revue/`, `DATA_INLINING_ROW_LIMIT 0`.

**Jeu de mesure.** Il compte 500 000 lignes : 100 dates × 100 départements × 50 produits.
- Clé `(date, departement, produit)`.
- `region`, parente de `departement`.
- Une paire code/libellé `code_nc8` / `libelle`.
- Trois mesures `DOUBLE` pseudo-aléatoires, pour une entropie réaliste.

Build de base : 1 fichier de 9,0 Mo (5 row groups de 122 880 lignes).

**Lot d'update type.** 50 000 lignes :
- 25 000 nouvelles (5 dates nouvelles) ;
- 25 000 existantes, dont 12 500 réellement modifiées et 12 500 identiques.

**Instruments.**
- Un proxy de connexion compte chaque `execute()` émis par le package.
- `time.perf_counter` mesure chaque étape (méthodes enveloppées).
- `ducklake_table_info`, `ducklake_table_changes` et `EXPLAIN ANALYZE` (« Total Files Read »).
- `storage_report().overlap_ratio`.

**Reproductibilité.** Les scripts et les résultats bruts JSON sont dans
`outputs/bench_revue/scripts/` (hors git) :
- `bench_revue.py` : les mesures ;
- `repro_revue.py` : les reproductions de bugs, notées R1…R12 ;
- `pytest_cov.txt` : la couverture.

Relance : `uv run --no-sync python outputs/bench_revue/scripts/bench_revue.py [scénario]`.

**Tests.** `uv run --no-sync pytest --cov=dt_ducklake_manager` : 527 réussis, 1 ignoré
(Postgres), **0 échec**, couverture totale 76 %, en 125 s (`pytest-cov` installé dans
`.venv` par `uv pip install`, absent de `pyproject`).

### Les huit constats qui pèsent le plus

1. **La fusion post-écriture ne fusionne jamais rien** (A1). Le sens de `min_file_size`
   est inversé dans le code **et dans la spec §5.5** : 9 fichiers restent 9, contre 9 → 4
   avec le bon paramètre.
2. **Une erreur au milieu d'un lot est avalée et commitée** (A2). `update_database`
   renvoie `True` et la base contient 14 lignes au lieu de 10 ou 16.
3. **Des doublons de clé primaire peuvent entrer en base et y rester** (A3, A4), par
   deux chemins distincts.
4. **`VALIDATE_AND_FIX` supprime 60 % des lignes et renvoie un succès** (A12).
5. **Le chemin par lots, actif par défaut, coûte 45 % de temps en plus** et triple le
   nombre de fichiers écrits (B2).
6. **Un update fait 96 requêtes SQL, dont 11 `COUNT(*)` complets** à 17,5 ms chacun à
   500 k lignes (A6, A7).
7. **Chaque update produit 2 snapshots**, dont un sans auteur (A8), **et 15 lignes INFO**
   au lieu d'une (A9).
8. **Le test « en échec connu » passe à HEAD.** Il reste fragile (A16).

---

## Panier A — à corriger (gain net, risque faible)

Classés par gain décroissant. Pour chacun : emplacement, observé, conséquence, correction, risque.

### A1. `merge_adjacent_files` appelé avec un `min_file_size` qui exclut tous les petits fichiers

- **Emplacements**
  - `maintenance/compaction.py:333-334` : défauts `min_file_size=100_000_000`,
    `max_file_size=500_000_000`.
  - `compaction.py:866` : `compact` les utilise.
  - `compaction.py:1370` : l'étape merge de `maintain` passe `policy.min_file_size_bytes`.
  - `compaction.py:1191-1194` : la fusion de `recluster` passe `target // 4`.
  - Message de log : `compaction.py:386-387`.
  - Mêmes lignes à corriger dans la documentation : `specification-bdd.md:467` et
    `docs/maintenance.md:50`.
- **Observé (mesuré).** `min_file_size` est une borne **basse** : les fichiers **plus
  petits** que ce seuil sont *exclus* de la fusion. Même catalogue (9 fichiers actifs
  après un update par lots), un appel par ligne :

  | Paramètres de `ducklake_merge_adjacent_files` | Fichiers traités → créés | Fichiers actifs |
  |---|---|---|
  | aucun | 6 → 1 | 9 → 4 |
  | `min 100 Mo, max 500 Mo` (valeurs du package) | **0** | **9 → 9** |
  | `min 0, max 500 Mo` | 6 → 1 | 9 → 4 |
  | `max 1 Mo` seul | 6 → 1 | 9 → 4 |
  | `min 150 Ko` seul | 4 → 1 (les deux fichiers de 105 et 125 Ko sont exclus) | 9 → 6 |

- **Conséquence.**
  - La fusion appelée après chaque `update_database`, `add_columns`, `delete_rows` et
    `update_value_labels` est un no-op garanti. Le banc le confirme :
    `merge_files_processed = 0` sur toutes les mesures.
  - L'étape merge de `maintain` se déclenche dès qu'il y a « trop de petits fichiers »,
    puis ne fusionne précisément aucun petit fichier.
  - Après 10 updates avec un `target_file_size` de 2 Mo, la table passe de 5 à
    **25 fichiers** sans qu'aucune compaction ne les réduise ; seul `recluster` les
    ramène à 6.
  - Le log affirme « no file under the threshold », ce qui est faux.
  - Aucun test ne vérifie qu'une fusion réduit le nombre de fichiers
    (`test_compaction.py:142`, `:161` ne regardent que la forme du retour).
- **Correction.**
  - Ne plus passer `min_file_size` par défaut.
  - Utiliser `max_file_size` comme plafond, égal au `target_file_size` du catalogue
    (déjà lu par `_target_file_size`, `compaction.py:1754`).
  - Garder `min_file_size_bytes` uniquement comme seuil de *comptage* des petits
    fichiers dans `storage_report`.
  - Corriger le message de log, la spec §5.5 et `docs/maintenance.md`.
  - Ajouter un test qui prouve la réduction du nombre de fichiers (9 → 4).
- **Risque.** Faible. La compaction post-écriture se met à réellement réécrire les petits
  fichiers : c'est le comportement documenté, mais un coût d'écriture apparaît après
  chaque opération. Il sera mesuré au TEMPS 2 ; ordre de grandeur attendu : celui d'un
  `INSERT` des lignes fusionnées.

### A2. Échec d'un lot avalé, puis commit partiel

- **Emplacements** : `operations/_data.py:635-637` (`_batch_insert_data`) et
  `_data.py:702-704` (`_batch_upsert_data`) : `except Exception: logger.error(...)`, puis
  la boucle continue.
- **Observé (R1).** Avec `batch_size=2` et une erreur d'E/S simulée sur le 2ᵉ `INSERT`,
  `update_database` renvoie **`True`** et la table passe de 10 à **14 lignes**, au lieu de
  10 (rollback) ou 16 (succès complet).
- **Conséquence.** Violation directe de la spec §4 et §7 : une transaction par
  opération, `ROLLBACK` sur exception. Le chemin par lots est pourtant **le chemin par
  défaut** dès que le lot dépasse 10 000 lignes. Couverture : `_data.py:612-639` et
  `685-706` ne sont jamais exécutées par les tests.
- **Correction.** Relancer l'exception (ou supprimer le découpage, voir B2).
- **Risque.** Nul : on passe d'un succès mensonger à un échec avec rollback.

### A3. Repli « tout insérer » quand la séparation insert/update échoue

- **Emplacement** : `operations/updater.py:809-814`.
- **Observé (R2).** Une erreur simulée dans la requête de séparation fait que toutes les
  lignes sont traitées comme nouvelles. `update_database` renvoie `True` et la base
  contient **2 clés primaires en double**.
- **Conséquence.** Corruption silencieuse : l'invariant d'unicité de la clé est rompu.
  Seul l'auditeur le signale, en HIGH et non bloquant (`auditor.py:1219`).
- **Correction.** Relancer l'exception.
- **Risque.** Nul.

### A4. Déduplication du lot sur toutes les colonnes au lieu des clés primaires

- **Emplacement** : `updater.py:832`. `remove_dataframe_duplicates` est appelé sans
  `primary_keys`, alors que le paramètre existe (`utils/sql.py:227`).
- **Observé (R3).** Un lot contenant deux lignes de même clé et de valeurs différentes
  (`keep="none"` par défaut) : les deux sont insérées, `update_database` renvoie `True`,
  et il reste **1 doublon de clé**. L'update suivant, avec `check_duplicates_db=True`, ne
  le retire pas : sa déduplication porte aussi sur toutes les colonnes.
- **Conséquence.** Même corruption qu'en A3. Incohérence avec le build, qui déduplique
  sur les clés primaires (`persistence.py:568`), et avec `add_columns`, qui refuse
  (`updater.py:999`).
- **Correction.** Passer `primary_keys=self._get_primary_key_columns()`.
- **Risque.** Faible : c'est une correction de bug. Un lot à doublons de clé perd
  désormais ses doublons selon `keep`, comme au build.

### A5. Upsert en deux passes redondantes

- **Emplacements.**
  - Séparation : `updater.py:723-814` (2 semi-jointures, 2 conversions Arrow).
  - Chemin direct : `_data.py:709-806`. Il refait un `EXISTS` (`:731`), un
    `INSERT … NOT EXISTS` qui insère toujours 0 ligne puisque tout le lot existe déjà
    (`:774`), et 2 `COUNT(*)` complets (`:767`, `:786`).
  - Les deux chemins `_update_fact_table_direct` / `_update_fact_table_batch`
    (`updater.py:577-720`) sont identiques à un booléen près.
- **Observé (mesuré, lot type).**

  | Chemin | Durée de l'écriture | Fichiers après | `overlap_ratio` |
  |---|---|---|---|
  | package, direct (séparation + insert + upsert) | 0,47 s | 3 | 0,67 |
  | package, par lots (défaut) | 0,89 s | 7 | 1,00 |
  | **une passe SQL** (`UPDATE … FROM` + `INSERT … NOT EXISTS … ORDER BY cluster_by`) | **0,34 s** | 3 | 0,67 |

- **Conséquence.** −28 % sur l'écriture par rapport au chemin direct, −62 % par rapport
  au chemin par défaut, et environ 20 instructions SQL en moins.
- **Correction.**
  - Remplacer la séparation et les deux appels à `DataManager` par une seule méthode
    privée à deux instructions.
  - Tirer les comptages de `ducklake_table_changes` (déjà interrogé) ou, sans catalogue
    réel, de `len(lot)`.
  - Fusionner `_update_fact_table_direct` et `_update_fact_table_batch`.
- **Risque.** Faible. Les comptages exacts du rapport restent ceux de
  `ducklake_table_changes`. Les tests existants d'update couvrent le résultat logique.

### A6. `COUNT(*)` complets redondants

- **Observé (mesuré).**
  - Un `COUNT(*)` sur la `fact_table` coûte **17,5 ms à 500 k lignes** : `EXPLAIN` ne
    montre pas de réponse par les métadonnées, le coût croît avec la table.
  - `update_database`, chemin par défaut : **11** `COUNT(*)` complets et 9 autres
    comptages.
  - `delete_rows` (5 000 lignes) : **9** `COUNT(*)` complets et 13 autres comptages, soit
    22 requêtes de comptage sur 42 instructions, pour un seul `DELETE`.
  - `delete_columns`, une opération DDL pure : 2 `COUNT(*)`.
- **Emplacements.**
  - Update : `updater.py:764` (test de vacuité), `:851`/`:870` (autour de la
    déduplication), `_data.py:767`/`:786`.
  - Delete :
    - `DataManager.delete_rows` : `_data.py:389`/`:404` ;
    - `_run_delete_rows` : `deleter.py:335`/`:344` ;
    - `_transaction` : `_base.py:257`/`:326`, déjà complété par `ducklake_table_changes`.
  - `add_columns` : `updater.py:1073`, utilisé seulement pour un log.
- **Conséquence.** Environ 0,19 s par update et 0,16 s par delete à 500 k lignes, soit
  9 à 25 % de la durée. Linéaire en taille de table.
- **Correction.** Garder les deux comptages de `_transaction` (`rows_before` /
  `rows_after` du rapport) et supprimer les autres : `ducklake_table_changes` et la
  taille du lot donnent la réponse.
- **Risque.** Faible. Sur une connexion sans DuckLake (tests en mémoire), les comptages
  de repli viennent de `_transaction`.

### A7. Lectures répétées de ce qui ne change pas pendant l'opération

- **Observé (mesuré, `update_database` par défaut, 96 instructions).**
  - 13 `DESCRIBE fact_table`.
  - 16 requêtes `information_schema` (`_table_exists`).
  - 10 lectures de `metadata`, 7 de `dataset_metadata`, 4 lectures des clés primaires.
  - 2 lectures de `cluster_by`, une par appel à `_cluster_by_order_clause`.
- **Emplacements** (inventaire complet des appels dans la revue d'exploration) :
  `updater.py:257`, `:270`, `:526`, `:601`, `:756`, `:856` ; `_data.py:291`, `:345`,
  `:596`, `:743`, `:813` ; `auditor.py:453`, `:605`, `:611`, `:612`, `:662`, `:663`,
  `:739`.
- **Conséquence.** Chaque lecture ne coûte que quelques millisecondes : le gain en temps
  est modeste (< 0,1 s), mais le nombre d'allers-retours est divisé par deux environ,
  et c'est ce qui compte sur un catalogue Postgres distant.
- **Correction.** Lire clés primaires, colonnes et `cluster_by` une fois en tête
  d'opération et les passer en argument. Faire passer les lectures de metadata du
  manager par le cache existant (`_load_current_metadata`, `_base.py:435`).
- **Risque.** Faible, à condition d'invalider après chaque DDL (déjà en place).

### A8. `updated_at` écrit hors transaction : deux snapshots par update

- **Emplacements.** `_touch_dataset_metadata()` est appelé après le commit par
  `updater.py:323` (`update_database`), `deleter.py:253` (`delete_rows`) et
  `deleter.py:484` (`delete_columns`). Il est appelé dans la transaction par
  `updater.py:1136` et `:1324` et par `_base.py:1339`.
- **Observé (mesuré).** Un `update_database` crée **2 snapshots**. Le second ne touche
  que `dataset_metadata` et n'a **ni auteur ni message**, même quand `run_id` est fourni.
- **Conséquence.**
  - L'horodatage n'est pas atomique avec l'écriture.
  - La traçabilité (§4.4) est coupée : le dernier snapshot n'est pas celui du run.
  - Le test `test_reporting.py` a dû chercher le snapshot par auteur pour cette raison.
- **Correction.** Déplacer l'appel dans le bloc `_transaction` des trois opérations,
  comme c'est déjà le cas pour les trois autres.
- **Risque.** Faible. Un échec d'horodatage annulerait l'écriture : c'est le
  comportement déjà retenu par `add_columns`.

### A9. Journalisation : 15 lignes INFO par update, un zéro faussement expliqué, mélange FR/EN

- **Observé (mesuré).** Un `update_database` produit 15 lignes INFO :
  - découpage des lots ×2 et « Upsert completed » ×3 ;
  - début et fin d'audit ;
  - 3 lignes de compaction ;
  - la synthèse.

  La spec §6 prévoit **une** ligne INFO de synthèse et le détail en DEBUG.
- **Emplacements.**
  - `_data.py:614`, `:688`, `:794` ; `updater.py:638`, `:711` ;
    `auditor.py:265`, `:305`, `:353` ; `compaction.py:385`, `:447`, `:877`.
  - Message faux : `compaction.py:386`.
  - Coquilles : `:448` « rewriten », `:392` « files(s) created(s) », `:513` « ilined »,
    `:726`/`:751` « Partitionning », `:817` « reparttition ».
  - Messages en français (`:571`, `:665`, `:716`) au milieu de messages en anglais.
- **Correction.** Passer en DEBUG tout ce qui n'est pas la synthèse, un warning ou un
  zéro explicité ; corriger les messages.
- **Risque.** Nul, hors assertions de tests sur le texte des logs (voir A16).

### A10. `update_value_labels` : codes absents non signalés et réécriture de lignes inchangées

- **Emplacements** : `updater.py:1286-1292` (`NOT IN (SELECT code …)`) et l'`UPDATE`
  de `updater.py:1301`.
- **Observé.**
  - **R4** : la colonne de code contient un `NULL` et on passe un code `ZZZ` absent.
    Aucun avertissement n'est émis : `x NOT IN (… NULL …)` n'est jamais vrai.
  - **Mesuré** : relabelliser un code avec le libellé qu'il porte déjà réécrit quand même
    ses **10 000 lignes** (copy-on-write : +1 fichier, +1 fichier de suppression,
    +225 Ko).
- **Correction.** Utiliser `NOT EXISTS` pour la détection des codes absents. Ajouter
  `AND f.label IS DISTINCT FROM t.label` à l'`UPDATE`.
- **Risque.** Faible. `rows_updated` ne compte plus que les lignes réellement changées,
  ce qui est la sémantique attendue du rapport.

### A11. `delete_rows(dict)` supprime 0 ligne sans rien dire

- **Emplacements.**
  - Signature : `deleter.py:165` et `_data.py:362` (`dict` accepté).
  - `utils/sql.py:476` lève une `TypeError` pour un `dict`.
  - Cette exception est avalée à `_data.py:414`.
- **Observé (R5).** `delete_rows({"id": 1})` renvoie `rows_deleted = 0`, sans aucun
  warning dans le rapport.
- **Correction.** Retirer `dict` des annotations et lever une `ValueError` explicite à
  l'entrée de `delete_rows`.
- **Risque.** Nul : aucun appelant n'utilise `dict` (grep sur le package, les tests,
  les notebooks et la doc).

### A12. `VALIDATE_AND_FIX` supprime des lignes sans confirmation

- **Emplacements.**
  - `maintenance/recovery.py:981-983` (`DELETE … WHERE col IS NULL`), déclenché par
    `:779`.
  - La liste des stratégies destructives (`:866`) ne contient que `REPAIR_SCHEMA`.
  - `_fix_constraint_violation_issue` (`:1119-1127`) fait un `CREATE OR REPLACE TABLE`
    qui perd l'identité DuckLake de la table.
- **Observé (R8).** Une colonne à 60 % de `NULL` suffit : `recover_database(VALIDATE_AND_FIX)`
  renvoie `success=True` et la table passe de **10 à 4 lignes**. Le même rapport
  recommande de « recréer la `fact_table` avec `PARTITION BY` sur la colonne la plus
  filtrée » (`auditor.py:878`), ce qui contredit la spec §2.1.
- **Correction immédiate.** Ranger `VALIDATE_AND_FIX` et `CLEAN_ORPHANED_DATA` parmi les
  stratégies destructives, qui exigent `confirm_destructive=True`. Le sort du module est
  en B8.
- **Risque.** Nul : aucun appelant hors tests. Couverture de ces chemins : 0 %.

### A13. Contrôles morts ou toujours vrais

- **Auditeur, R6.**
  - `ducklake_snapshots()` sans catalogue lève une `BinderException`.
  - `ducklake_data_files()` n'existe pas (`CatalogException`).
  - Les deux erreurs sont avalées (`auditor.py:919-975`) : ce contrôle de maintenance
    n'a jamais tourné. Il fait doublon avec `storage_report` et ses seuils sont fixés
    en dur à 100 et 500 (`:913-914`).
  - Correction : supprimer ce contrôle.
- **Auditeur.** La branche `elif null_percentage == 100` (`auditor.py:1036`) est
  inatteignable, car `> 50` la capture avant. Correction : inverser l'ordre.
- **Deleter, R7.** `validate_operation("drop_column")` (`deleter.py:427`) produit une
  issue MEDIUM « Unknown operation type » et passe toujours. Correction : retirer
  l'appel. L'analyse de dépendances `_analyze_column_dependencies` fait déjà le vrai
  travail.
- **Risque.** Nul : aucun comportement observable ne change.

### A14. Colonnes ajoutées par `allow_new_columns` hors transaction, ordre des liens `parent_name`

- **Emplacements** : `updater.py:278` (appel avant le `with self._transaction`, `:288`)
  et `updater.py:365-377` (ajout de la colonne, puis de sa metadata, puis de ses champs
  d'UI, une colonne à la fois).
- **Observé.**
  - **R11** : un update refusé pour violation code/libellé renvoie `False`, mais la
    nouvelle colonne **reste dans la `fact_table`**.
  - **R12** : `column_metadata={"commune": {"parent_name": "dep"}}`, où `dep` est une
    autre colonne nouvelle du même lot, échoue avec `ValueError: Parent column 'dep' has
    no row in the metadata table`.
- **Correction.**
  - Ouvrir la transaction avant l'ajout des colonnes. Un `ALTER TABLE ADD COLUMN` est
    bien annulé par `ROLLBACK` : c'est mesuré au prompt 6 et utilisé par `add_columns`.
  - Ajouter toutes les colonnes et leurs lignes `metadata`, puis appliquer les champs
    d'UI.
- **Risque.** Faible. Le warning « those columns remain » (`updater.py:307-313`)
  disparaît.

### A15. Code mort sans aucun appelant

Vérifié par grep sur le package, `tests/`, `notebooks/*.ipynb`, `docs/` et le README.

- `DataManager.get_table_stats` (`_data.py:823`) : appelle `pg_size_pretty`, qui
  n'existe pas dans DuckDB, et renvoie donc toujours « Unknown ». Il ne sert qu'aux deux
  `get_*_status`, traités en B7.
- `DataManager.drop_columns` (`_data.py:484`) contourne le nettoyage de `cluster_by`,
  `parent_name` et `label_for`. `DataManager.update_column_type` (`_data.py:530`) écrit
  un type non validé, qui peut rétrécir une colonne.
- `DuckLakeMaintenance._file_ranges` (`compaction.py:1622`).
- Dans l'auditeur : `_get_table_columns` (`auditor.py:1294`) et
  `_column_exists_in_fact_table` (`:1335`).
- `max_workers` (`updater.py:58`, `:120`) : stocké, jamais utilisé.
- `RecoveryStrategy.FILE_PATH` (`recovery.py:23`), et une docstring qui cite une
  stratégie inexistante, `RESTORE_BACKUP` (`:185`).

- **Correction.** Supprimer, et pour `max_workers`, garder le paramètre avec un
  `DeprecationWarning` ou le retirer (voir B7). Je préfère le retirer.
- **Risque.** Nul.

### A16. Le test « en échec connu » : diagnostic

- **Observé.** `test_rewrite_data_files_zero_when_no_deletions` **passe** à HEAD.
  - Historique : le commit `331b134` a traduit le log en anglais, `3d83977` a ajouté le
    test avec l'assertion française « seuil de suppression », `01fbe36` a corrigé
    l'assertion.
  - La note de mémoire de session qui le déclarait en échec est donc périmée.
- **Fragilité restante.**
  - Deux libellés pour le même zéro : `compaction.py:447` quand DuckLake renvoie un
    résultat vide, `:455` s'il renvoie une ligne de zéros. Le second libellé ne contient
    pas « deletion threshold ».
  - `test_reporting.py:272` teste encore `"seuil de suppression" … or "threshold"`.
- **Correction.** Un seul libellé pour le zéro. Faire porter l'assertion sur le retour
  `(…, 0, 0)` et sur `report.maintenance`, pas sur le texte du log.
- **Risque.** Nul.

### A17. Tests : outillage et duplication

- `_ducklake_available()` est copié dans 9 modules de test ; il devrait vivre une fois
  dans `tests/conftest.py`, en portée session.
- `pytest-cov` est absent du groupe `dev` alors que `CLAUDE.md` prescrit
  `pytest --cov`.
- Le marqueur `slow` est déclaré mais jamais utilisé.
- Les tests de régression de A1, A2, A3, A4, A10, A11, A12 et A14 sont à ajouter
  (voir §Tests).
- **Risque.** Nul.

### A18. Petites inefficacités mesurables ou certaines

- `_get_null_only_columns` (`_base.py:1496-1506`) et l'audit COMPREHENSIVE
  (`auditor.py:1008-1017`) lancent un `COUNT(*) WHERE col IS NULL` par colonne. À 17 ms
  le scan, cela fait environ 150 ms pour 9 colonnes, contre un seul
  `SELECT COUNT(c1), COUNT(c2), …`.
- `_remove_database_duplicates` exclut une colonne nommée en dur `"value"`
  (`updater.py:858`), un reste du schéma d'avant la refonte.
- `updater.py:778` : `df.to_arrow().select(df.columns)`, où le `select` ne fait rien.
- Littéraux répétés : `100_000_000` ×5 et `500_000_000` ×2 dans `compaction.py`, et
  `0.1` ×3. À dériver de `RECOMMENDED_DUCKLAKE_OPTIONS` ou à regrouper en constantes de
  module.
- **Risque.** Nul.

---

## Panier B — à discuter (gain réel, mais comportement, API ou structure)

Chaque ligne attend un « oui », un « non » ou « oui, avec variante ».

### B1. `check_duplicates_db=True` par défaut : scan et `DELETE` de toute la table à chaque update

- **Emplacements** : `updater.py:180`, `:505`, `:835-887` et
  `utils/sql.py:284-351`.
- **Mesuré.**
  - 0,15 à 0,23 s par update à 500 k lignes, soit 8 à 10 % de la durée. C'est linéaire
    en taille de table, pas en taille de lot.
  - Un contrôle restreint à la clé primaire (`GROUP BY pk HAVING COUNT(*) > 1`) coûte
    0,056 s.
- **Mécanisme.**
  - La clé primaire est obligatoire (`updater.py:257`) et l'upsert préserve l'unicité.
    Une fois A3 et A4 corrigés, aucun chemin ne peut plus créer de doublon.
  - La déduplication porte sur **toutes les colonnes** : elle ne détecte même pas les
    doublons de clé qui ont des valeurs différentes (R3).
  - Avec `keep="none"`, elle **supprime toutes les occurrences**.
- **Proposition.** Passer le défaut à `False`, ou le remplacer par un contrôle
  d'unicité de la clé qui **lève** au lieu de supprimer. -> Oui, je souhaite le rempalcer par un controle d'unicité de la clé qui lève au lieu de supprimer
- **Enjeu.** Changement de défaut d'un paramètre public.

### B2. Supprimer le découpage Python en lots (`use_batch_processing`, `batch_size`)

- **Emplacements** : `updater.py:184`, `:513-518` ; `_data.py:268-359`, `:609-706`.
- **Mesuré (lot type).**

  | Mode | Durée | SQL | Fichiers | Overlap |
  |---|---|---|---|---|
  | lots de 10 000 (défaut) | 2,20 s | 96 | 7 | 1,00 |
  | direct | 1,51 s | 66 | 3 | 0,67 |

- **Mécanisme.**
  - Chaque lot est trié à part et écrit dans ses propres fichiers. Le lot entier n'est
    donc **pas** trié selon `cluster_by`, contrairement à la spec §4.2.
  - Chaque lot répète ses lectures et ses comptages.
  - DuckDB traite déjà un `INSERT … SELECT` en flux.
- **Proposition.** Une seule instruction par étape (A5). `use_batch_processing` et
  `batch_size` deviennent sans effet, avec un `DeprecationWarning`, puis sont retirés. -> Oui
- **Enjeu.** API publique : paramètre de `update_database` et du constructeur.

### B3. Audit `STANDARD` après chaque écriture, précondition avant

- **Emplacements** : `updater.py:249`, `:530-538` ; `deleter.py:130`, `:360`, `:542`.
- **Mesuré.** Audit : 0,17 à 0,21 s par update. Précondition : 0,02 à 0,06 s. Au total,
  environ 10 % de la durée.
- **Mécanisme.**
  - L'audit STANDARD re-scanne **toute** la table pour chaque paire code/libellé
    (`auditor.py:785-789`), juste après le contrôle restreint au lot de l'étape 4b.
  - Il relit `metadata` 4 fois et fait 2 `DESCRIBE`.
  - Ses seuls CRITICAL possibles portent sur des invariants que l'écriture garantit
    déjà.
- **Proposition.** `enable_validation=False` par défaut, ou un audit `BASIC` après
  écriture ; l'audit complet reste disponible à la demande. -> Oui, audit BASIC après écrituren par défaut avec la possibilité de réaliser un audit complet à la demande. Je ne souhaite en effet pas revérfier systématiquement des conditions qui sont toujours vérifiées à l'écriture. Si "enable_validation" n'est pas pertinent, je souhaite le supprimer.
- **Enjeu.** Changement de défaut.

### B4. L'upsert réécrit les lignes inchangées

- **Emplacement** : `_data.py:754-764`.
- **Mesuré.** Avec `AND (f.c IS DISTINCT FROM t.c OR …)` :
  - 12 500 lignes réécrites au lieu de 25 000 ;
  - 9,71 Mo au lieu de 9,98 Mo ;
  - même durée (0,34 s).
- **Enjeu.** `rows_updated` compterait les lignes *modifiées*, plus les lignes *fournies
  et existantes*. C'est plus juste, mais c'est un changement de sémantique du rapport.
  Le gain de stockage est proportionnel à la part de lignes inchangées ; pour des
  modèles qui repassent sur tout l'historique avec peu de changements, il peut être
  majoritaire.

-> Oui, je ne souhaite pas réécrire des lignes inchangées

### B5. `DataManager._ensure_columns_exist` : une porte dérobée

- **Emplacements.**
  - `_data.py:810-819`, appelée par `_data.py:646` et `:715`.
  - `DatabaseUpdater.data_mgr` et `DatabaseDeleter.data_mgr` sont des attributs publics.
- **Observé (R10).** `updater.data_mgr.insert_data(df_avec_colonne_intruse)` ajoute
  silencieusement la colonne à la `fact_table` et à `metadata`, contournant
  `allow_new_columns=False` (prompt 6).
- **Proposition.** Remplacer l'ajout par une `ValueError` et renommer l'attribut en
  `_data_mgr`. Une fois A5 fait, `DataManager` ne garde que des fonctions internes : il
  pourrait être fusionné dans `_base.py` ou `updater.py`.
- **Enjeu.** Attribut public : `tests/unit/test_operations/test_data.py` l'utilise.

-> Oui

### B6. Erreurs et types de retour hétérogènes

- **État.**
  - `update_database` renvoie un `bool` et **avale** la `ValueError` de dépendance
    code/libellé (R9 : `False`, le message est dans `last_report.warnings`). La spec
    §2.6 annonce pourtant une « `ValueError` explicite », alors que §6 garde le `bool`.
  - `delete_rows` et `delete_columns` ne lèvent jamais.
  - `add_columns` et `update_value_labels` lèvent.
  - `cleanup_database` et `get_*_status` renvoient `{"error": …}`.
  - Les méthodes de `DataManager` renvoient `0`, `False` ou `[]` sur erreur.
  - `_run_update_steps` transforme ces `False` en `RuntimeError` génériques, et perd le
    message d'origine (`updater.py:506-518`).
- **Proposition.** Les erreurs de *saisie* (clé manquante, colonne inconnue, dépendance
  violée) lèvent `ValueError` partout ; les erreurs d'*exécution* se propagent avec leur
  exception d'origine après rollback. `update_database` garde son `bool` pour les seules
  erreurs d'exécution. -> Oui
- **Enjeu.** Contrat public. Arbitrage à faire entre §2.6 et §6 de la spec.

### B7. Méthodes publiques sans usage réel

Comptage des usages hors de la définition :

| Méthode | Usages | Emplacement |
|---|---|---|
| `DatabaseUpdater.optimize_database` | tests seuls (3) | `updater.py:1446` |
| `DatabaseUpdater.get_update_status` | 0 | `updater.py:1403` |
| `DatabaseDeleter.get_deletion_status` | 0 | `deleter.py:768` |
| `DatabaseDeleter.get_deletion_impact` | 0 | `deleter.py:682` |
| `DatabaseDeleter.cleanup_database` | tests (8), `recovery.py:726` | `deleter.py:811` |
| `BaseSchemaManager.validate_database_state` | 0 | `_base.py:1543` |
| `set_partitioned_by`, `reset_partitioned_by`, `repartition` | tests seuls | `compaction.py:680-821` |
| `DatabaseRecoveryManager.auto_recover_from_validation` | 0 (ni test) | `recovery.py:272` |
| `DuckLakeConnector.at_clause` | notebooks 2 et 5, doc, 0 test | `connector.py:423` |

- **Proposition.**
  - Supprimer les cinq premières lignes et `auto_recover_from_validation`.
  - Garder le partitionnement, prévu par la spec §5.3, mais le documenter.
  - Tester `at_clause`, dont la docstring promet une `ValueError` jamais levée
    (`:442` contre `:457-461`). -> Oui
- **Enjeu.** Retrait d'API publique (le projet n'est pas publié, spec en-tête).

### B8. `recovery.py` : réduire au time travel natif

- **État.** 971 lignes, couverture 31 %.
  - `USE_SNAPSHOT_HISTORY` ne fait que lister les snapshots et renvoyer un texte
    d'instructions, avec des noms de tables non qualifiés.
  - `REPAIR_SCHEMA`, `CLEAN_ORPHANED_DATA` et `VALIDATE_AND_FIX` réparent en place de
    façon destructive (A12) ou sans effet : `_apply_schema_fix` (`:633`) exécute un
    `CREATE IF NOT EXISTS` sans effet et rapporte « Fixed ».
  - `CLEAN_ORPHANED_DATA` transforme `{"error": …}` en « Cleaned error » avec
    `success=True` (`:726-730`).
- **Proposition.** Ne garder que `list_ducklake_snapshots` et une restauration
  documentée `INSERT … SELECT … AT (VERSION => n)`. C'est le principe 5 de la spec :
  « le time travel est le mécanisme de récupération ». Suppression d'environ 800 lignes. -> Oui
- **Enjeu.** Retrait d'API et de structure de fichiers ; notebook 7 à ajuster.

### B9. Auditeur : recentrer sur ce que l'écriture ne garantit pas

- **État.** 1 287 lignes, couverture 70 %.
  - Il duplique les validations d'écriture : dépendance code/libellé, unicité de la clé,
    règles de nom de colonne, avec une règle différente de `_data.py:124`
    (l'auditeur accepte les espaces).
  - Il recommande un partitionnement contraire à la spec (`:878`).
  - Il n'audite pas les cycles de hiérarchie : `validate_hierarchy_forest` n'est jamais
    appelé.
  - Trois helpers font le même `DESCRIBE` (`:1283`, `:1294`, `:1305`) ; les helpers
    renvoient `[]` en silence sur erreur (`:1279-1358`).
- **Proposition.** Un auditeur structurel à la demande :
  - trois tables présentes ;
  - `metadata` et colonnes de la `fact_table` cohérentes ;
  - forêt `parent_name` et invariants `label_for` ;
  - unicité de la clé.

  La recommandation de partitionnement et les contrôles de maintenance (déjà dans
  `storage_report`) sont supprimés. -> Oui
- **Enjeu.** Structure et API (`ValidationLevel`).

### B10. Découpage de `compaction.py` (1 594 lignes)

- **Proposition.** `recluster`, `storage_report` et `maintain` forment un sous-ensemble
  cohérent (diagnostic et politique), distinct des enveloppes minces des procédures
  DuckLake. Découpage possible en `maintenance/procedures.py` et `maintenance/policy.py`.
- **Détails constatés.**
  - Signatures incohérentes : `(schema, table)` pour merge/rewrite, `(table, schema="main")`
    pour le partitionnement, `(table, schema=None)` pour le reste.
  - `maintain` peut appeler `storage_report` jusqu'à 4 fois. Mesuré : 12 requêtes et
    0,09 s par appel, `_table_id` résolu 3 fois.
- **Enjeu.** Structure de fichiers et signatures publiques. Gain lisibilité, pas
  performance.

-> Oui, je souhaite effectuer ce refactoring

### B11. `from_postgres` et l'`ATTACH`

- **Emplacements.** `from_postgres` (`connector.py:532-740`) n'accepte ni
  `ducklake_options` ni `data_inlining_row_limit`, alors que Postgres est le backend de
  production. Plusieurs littéraux de l'`ATTACH` ne sont pas échappés (`:1145`, `:1153`,
  `:1161`, `:1169`), alors que `_quote_literal` existe.
- **Mesuré.** Passer le même `DATA_PATH` en relatif au lieu d'absolu fait échouer
  l'`ATTACH` (« does not match existing data path »). Normaliser en chemin absolu
  l'éviterait.
- **Enjeu.** Ajout de paramètres : à la frontière de la fonctionnalité.

-> Oui, je souhaite ajouter ces paramètres

---

## Panier C — constaté, à ne pas corriger

- **C1. `target_file_size = 100MB`, `overlap_ratio` et `recluster` à ces volumes.**
  - Mesuré : 500 k lignes pèsent 9 Mo, soit **1 fichier**. L'élagage par fichier ne
    commence qu'au-delà d'environ 5,5 M lignes.
  - Pour simuler un grand volume, j'ai forcé un `target_file_size` de 2 Mo : 5 fichiers,
    puis 25 après 10 updates. Même dans ce cas, les requêtes de dashboard restent entre
    13 et 36 ms, qu'elles lisent 1 fichier ou 25. `date = …` lit 2 fichiers sur 25 ;
    `departement = …` en lit 25.
  - Après `recluster` : 6 fichiers, overlap 0,54 → 0.
  - Le défaut tient : aux volumes du projet, l'élagage se fait au niveau des row groups.
    Ne pas le changer. À dire dans `docs/maintenance.md` : « recluster ne sert qu'au-delà
    de quelques millions de lignes ». -> Oui, à spécifier dans maintenance
- **C2. `parquet_row_group_size = 122 880`.** Mesuré : c'est déjà le défaut effectif
  (5 row groups de 122 880 sans option). Le commentaire « aligné sur les lots »
  (`connector.py:20-24`) est trompeur, mais l'option ne coûte rien : corriger le
  commentaire au passage, rien d'autre. -> Oui
- **C3. Choix de `cluster_by`.** Mesuré à 2 Mo par fichier :
  - clé primaire en tête (`date`) : `date = …` lit 2 fichiers sur 5 et
    `departement = …` en lit 5 ;
  - `departement` en tête : `departement = …` lit 2 fichiers et `date = …` en lit 5.

  C'est le compromis attendu et déjà décrit par la spec §5.3 : rien à changer dans le
  code.
- **C4. `delete_threshold` (0,05, 0,1 ou 0,3).** Mesuré : après un upsert de 10 % des
  lignes, les trois seuils réécrivent le même fichier. Comme les updates sont groupés par
  `cluster_by`, les suppressions se concentrent (50 % du fichier touché) et le seuil
  exact n'a pas d'effet observable. L'écart entre 0,05 (`add_columns`) et 0,1 (update)
  est sans conséquence : rien à changer.
- **C5. Coûts intrinsèques.**
  - `add_columns` sur toutes les clés réécrit les 500 k lignes : 3,8 s, 9 → 14 Mo jusqu'au
    cleanup. C'est le copy-on-write d'un `UPDATE` complet, imposé par la spec §4.3.
  - Le contrôle de dépendance code/libellé restreint au lot coûte 0,30 à 0,35 s quand le
    lot touche tous les codes. L'invariant porte sur toutes les lignes d'un code, donc le
    scan est incompressible.
  - Inférence de `is_categorical` au build : 0,09 s sur 1,2 s.
- **C6. `recluster` via une table temporaire.**
  - Mesuré : l'alternative `INSERT … SELECT … AT (VERSION => n)` **dans** la même
    transaction que le `DELETE` renvoie **0 ligne**. La table temporaire est donc
    nécessaire.
  - Durée : 1,8 à 2,0 s pour 500 k lignes.
- **C7. `categorical_threshold`** est toujours vivant, et c'est voulu : il sert à
  l'inférence à la création d'une colonne. Ce n'est pas un reste du prompt 7.
- **C8. Arborescence de tests.** L'exemple de `CLAUDE.md` (`delays/`, `frequency/`)
  vient d'un autre projet ; l'arborescence réelle reflète le package. Il suffit de
  corriger l'exemple de `CLAUDE.md` lors d'une prochaine révision de ce fichier.
- **C9. `duckdb>=1.5.3` dans `pyproject` contre 1.5.2 installé.** Sans effet tant que
  `uv sync` est bloqué par OneDrive. Les mesures de la spec et de cette revue sont sur
  1.5.2. À revérifier (annexe A de la spec) lors de la montée de version.

---

## Tests : où la couverture est réellement insuffisante

Le pourcentage global (76 %) masque où se trouvent les risques :

| Zone | Couverture | Pourquoi c'est grave |
|---|---|---|
| `_data.py` chemins par lots `612-639`, `685-706` ; `updater.py:670-720` | 0 % | C'est le chemin **par défaut** au-delà de 10 000 lignes, et celui du bug A2. Tous les tests d'update utilisent des lots de moins de 10 lignes. |
| `updater.py:809-814` (repli de la séparation) | 0 % | Bug A3. |
| Fusion effective (`merge_files`) | forme seule | Bug A1 : aucun test ne compte les fichiers avant et après. |
| `recovery.py` hors `USE_SNAPSHOT_HISTORY` | 31 % du module | Chemins destructifs (A12, B8). |
| `update_database` : `keep` par défaut `"none"`, `"last"`, `"any"`, doublons de clé dans le lot, lot vide, clé nulle | non testés | Tous les tests passent `keep="first"`. Bug A4. |
| `update_value_labels` : code NULL, code absent, colonne inconnue | 4 tests, chemin heureux | Bug A10. |
| `delete_rows` : `dict`, filtre sans correspondance, `None` | non testés | Bug A11. |
| `build_schema` : `ValueError` d'unicité de la clé (`persistence.py:583-588`), entrée pandas | non testés | Pandas est annoncé (narwhals) mais absent de toute la suite. |
| `connector.at_clause` | 0 test | Contrat de docstring faux. |

Deux autres constats :
- **Tests lents pour ce qu'ils vérifient.** `clustered_conn` (`test_recluster.py:123`)
  reconstruit 300 000 lignes pour chacun des 41 tests qui l'utilisent. Sur les 125 s de
  la suite, les fixtures DuckLake réelles en portée fonction dominent. Une fixture de
  catalogue construite une fois, puis copiée, n'est pas possible : le `DATA_PATH`
  absolu empêche la copie (mesuré). Une construction en portée `module` pour les tests
  en lecture seule serait sûre.
- **Tests « DuckLake » sans DuckLake.** `built_ducklake_schema` (`tests/conftest.py:136`)
  est une connexion DuckDB en mémoire. La plupart des tests d'opérations n'exercent donc
  ni `ducklake_table_changes`, ni les snapshots, ni la compaction : c'est pour cela
  qu'A1 et A8 sont passés inaperçus.

---

## Proposition d'ordre d'application (TEMPS 2)

Un commit par sujet :
1. A2, A3, A4 (intégrité), avec leurs tests.
2. A1 (fusion), avec le test du nombre de fichiers, la spec §5.5 et `docs/maintenance.md`.
3. A5, A6, A7 (upsert en une passe, comptages, lectures), avec mesure avant/après sur le
   lot type.
4. A8 et A14 (transactions).
5. A10, A11, A12, A13.
6. A9 et A16 (logs et test).
7. A15, A17, A18 (nettoyage et outillage).

Puis les lignes du panier B retenues.
