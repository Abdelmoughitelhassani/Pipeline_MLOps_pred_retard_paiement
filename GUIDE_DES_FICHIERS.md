# Guide des fichiers du projet

À quoi sert chaque fichier, quand l'exécuter, et ce qu'il produit.

---

## Vue d'ensemble

Le projet s'organise en quatre couches :

```
data/raw/          les données brutes, jamais modifiées
      |
src/               le code réutilisable (chargement, features, config MLflow)
      |
scripts/           les scripts exécutables (expériences, entraînement, rapport)
      |
reports/ models/   tout ce qui est produit : métriques, figures, modèles, rapport
```

Une règle traverse le projet : **le code et les résultats légers vont dans git, les
artefacts lourds vont dans DVC**. Git ne conserve alors que des pointeurs `.dvc` porteurs
du hash de contenu.

---

## `src/` — code réutilisable

Ces deux modules ne s'exécutent pas seuls : ils sont importés par les scripts.

### `src/data_prep.py`
Le cœur du projet. Il fait trois choses :

- **Charge et nettoie** — lit le fichier Excel, renomme `PAY_0` en `PAY_1` et la cible en
  `DEFAULT`, puis fusionne les codes non documentés (`EDUCATION` 0/5/6 → 4, `MARRIAGE` 0 → 3)
  dans la catégorie « autre » prévue par le schéma UCI.
- **Construit les features** — 45 variables dérivées : taux d'utilisation du crédit, ratios de
  remboursement, agrégats et volatilités des montants, dynamique des retards.
- **Expose deux vues des mêmes données** :
  - `get_tabular()` → matrice (30000, 68) pour XGBoost, LightGBM, etc.
  - `get_sequences()` → tenseur (30000, 6, 5) + statiques (30000, 11) pour les LSTM/GRU,
    avec les 6 mois remis **en ordre chronologique** (le fichier les numérote à l'envers).

`train_test_indices()` fournit le découpage stratifié partagé par toutes les expériences,
ce qui garantit que tous les chiffres du projet sont comparables entre eux.

### `src/tracking.py`
Configuration centralisée de MLflow. Bascule automatiquement entre la base SQLite locale
(`mlflow.db`) et un serveur distant si `MLFLOW_TRACKING_URI` est défini — c'est ce qui permet
d'envoyer les runs vers DagsHub sans modifier une ligne de code.

---

## `scripts/` — scripts exécutables

### Pipeline de production (référencés dans `dvc.yaml`)

| Script | Rôle | Durée |
|---|---|---|
| `exp_ablation.py` | Compare 4 jeux de features à modèle constant. Répond à « le feature engineering est-il le goulot ? » | ~2 min |
| `run_experiments.py` | Compare 11 modèles + tuning. **Reprenable** : un modèle déjà calculé est ignoré au relancement | ~40 min |
| `train_final.py` | Entraîne le modèle retenu, le sérialise, l'enregistre au registre MLflow | ~3 min |
| `final_eval.py` | Évalue sur le test set et calcule les seuils par coût métier | ~2 min |
| `build_report.py` | Génère le rapport PDF de 31 pages | ~1 min |

`run_experiments.py` mérite un mot : il sauvegarde les prédictions **out-of-fold** de chaque
modèle dans `reports/oof/`. C'est ce qui rend le blending gratuit ensuite (il suffit de
moyenner des fichiers) et ce qui a permis de survivre aux interruptions du système.

### Expériences ponctuelles (hors pipeline)

| Script | Rôle | Résultat obtenu |
|---|---|---|
| `exp_boruta.py` | Sélection de variables par Boruta | Aucun gain, mais rejette les 4 variables démographiques sans perte |
| `exp_optuna.py` | Tuning bayésien, comparé à `RandomizedSearchCV` | Gagne en validation croisée, pas sur le test |
| `exp_outliers.py` | Isolation Forest, LOF, DBSCAN — protocoles correct et incorrect | Aucun gain ; démontre le piège de la prévalence |
| `mlflow_backfill.py` | Réinjecte l'historique dans MLflow **sans réentraîner**. Idempotent | 40 runs restaurés depuis les fichiers |

### Scripts historiques (remplacés)

`exp_tabular.py` et `exp_deep.py` étaient les premières versions de la comparaison de modèles.
Elles ont été interrompues par des manques de mémoire, ce qui a motivé la réécriture en
`run_experiments.py` avec sauvegarde incrémentale. Ils sont conservés pour la traçabilité mais
**ne doivent plus être utilisés** — leurs résultats sont intégrés dans `model_ranking.csv`.

---

## Documentation

| Fichier | Rôle |
|---|---|
| `README.md` | Point d'entrée du dépôt : résultats obtenus, conclusions, installation, utilisation. À lire en premier pour comprendre **ce que le projet a trouvé** |
| `GUIDE_DES_FICHIERS.md` | Ce document : à quoi sert chaque fichier. À lire pour comprendre **où se trouve quoi** |

---

## Configuration

| Fichier | Contenu |
|---|---|
| `params.yaml` | Tous les paramètres : découpage, validation croisée, hyperparamètres du modèle, seuils de coût. **Modifier une valeur ici suffit**, `dvc repro` relance les seules étapes concernées |
| `dvc.yaml` | Le pipeline : étapes, dépendances, sorties. C'est lui qui sait quoi relancer quand quelque chose change |
| `dvc.lock` | État figé du pipeline (hashes des entrées/sorties). Généré, ne pas éditer à la main |
| `.dvc/config` | Adresse du stockage distant DagsHub. **Sans identifiant** — ceux-ci vivent dans `.dvc/config.local`, ignoré par git |
| `requirements.txt` | Versions figées des dépendances |
| `.gitignore` | Exclut l'environnement virtuel, les caches, et les artefacts confiés à DVC |

---

## Données et artefacts (suivis par DVC)

Ces fichiers n'apparaissent pas dans git : seuls leurs pointeurs `.dvc` y figurent.
Récupérables par `dvc pull`.

| Artefact | Taille | Contenu |
|---|---|---|
| `data/raw/default_of_credit_card_clients.xls` | 5.4 Mo | Données source UCI, 30 000 clients, jamais modifiées |
| `reports/oof/*.npy` | 2.1 Mo | Prédictions out-of-fold des 11 modèles |
| `models/final_model.joblib` | 829 Ko | Modèle sérialisé + liste des features + seuil de décision |
| `models/model_card.json` | 3 Ko | Carte du modèle : config, métriques, données d'entraînement, seuil |
| `mlflow.db` | 1.1 Mo | Base des 57 runs d'expériences |
| `mlartifacts/` | 2.0 Mo | Artefacts MLflow, dont le modèle du registre |

---

## `reports/` — résultats (dans git)

Ces fichiers sont légers, lisibles et versionnés en texte, ce qui rend leurs évolutions
visibles dans les diffs git.

### Métriques

| Fichier | Ce qu'il contient |
|---|---|
| `phase1_strategy_results.csv` | 15 combinaisons modèle × stratégie de rééquilibrage |
| `ablation_results.csv` | 4 jeux de features évalués à modèle constant |
| `model_ranking.csv` | Classement des 11 modèles (PR-AUC, ROC-AUC) |
| `blend_ranking.csv` | Les 50 meilleurs ensembles parmi 682 testés |
| `best_params.json` | Hyperparamètres trouvés par `RandomizedSearchCV` |
| `optuna_results.csv` | Optuna vs `RandomizedSearchCV`, à budget égal et à 100 essais |
| `optuna_best_params.json` | Hyperparamètres trouvés par Optuna (ceux en production) |
| `optuna_history_*.csv` | Les 100 essais de chaque modèle, avec leurs paramètres |
| `boruta_results.csv` | Impact de la sélection de variables sur la performance |
| `boruta_ranking.csv` | Rang et statut de chacune des 68 variables |
| `boruta_features.json` | Listes des variables confirmées, indécises et rejetées |
| `outliers_results.csv` | 3 détecteurs × 2 protocoles |
| `outliers_enrichment.csv` | Taux de défaut parmi les lignes flaguées — explique pourquoi le nettoyage nuit |
| `final_evaluation.json` | Métriques finales sur le test set (déclaré comme métrique DVC) |
| `cost_thresholds.csv` | Seuil optimal selon le rapport de coût métier |

### Figures et rapport

`reports/figures/` contient les graphiques générés par `build_report.py`, et
`reports/figures/notebook/` ceux extraits du notebook d'analyse exploratoire.

`rapport_pipeline_credit_default.pdf` — le rapport complet de 31 pages, en trois parties :
pipeline de base, optimisation et analyse du plafond, puis sélection de variables et MLOps.
**Toutes ses valeurs sont lues depuis les fichiers ci-dessus** : régénérer le PDF après une
nouvelle expérience suffit à le mettre à jour, rien n'est saisi à la main.

---

## `notebooks/01_eda_analyse_donnees.ipynb`

L'analyse exploratoire initiale et le premier pipeline complet : chargement, vérification des
doublons et valeurs manquantes, cohérence des catégories, déséquilibre de la cible,
distributions, corrélations, puis nettoyage, feature engineering et comparaison de 15
combinaisons modèle × stratégie de rééquilibrage.

C'est le document pédagogique du projet : chaque décision y est expliquée à côté du code qui
l'applique. Le code réutilisable en a ensuite été extrait vers `src/` et `scripts/`.

---

## Ordre d'exécution

Pour reproduire le projet depuis un clone :

```bash
pip install -r requirements.txt
dvc pull                      # récupère données et artefacts
dvc repro                     # relance le pipeline (uniquement ce qui a changé)
```

Pour rejouer une expérience ponctuelle :

```bash
python scripts/exp_boruta.py      # sélection de variables
python scripts/exp_optuna.py      # tuning bayésien
python scripts/exp_outliers.py    # traitement des outliers
```

Pour consulter l'historique sans rien réentraîner :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Où trouver quoi

| Question | Réponse |
|---|---|
| Qu'a-t-on déjà essayé, et avec quel résultat ? | MLflow (57 runs) ou `reports/*.csv` |
| Quels hyperparamètres a le modèle en production ? | `params.yaml` et `models/model_card.json` |
| Comment les features sont-elles construites ? | `src/data_prep.py`, fonction `add_features()` |
| Pourquoi tel choix méthodologique ? | Le rapport PDF, ou les docstrings en tête de chaque script |
| Quelle version des données a servi ? | Le hash dans `data/raw/*.dvc` |
| Comment reprendre après une interruption ? | `run_experiments.py` ignore ce qui est déjà calculé |
