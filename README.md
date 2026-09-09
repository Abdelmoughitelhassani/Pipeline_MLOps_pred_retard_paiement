# Prédiction de retard de paiement — Pipeline MLOps

Prédiction du défaut de paiement de clients de cartes de crédit à partir de leur historique
de facturation et de remboursement, sur le jeu de données
[Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
— 30 000 clients à Taïwan, 22.12 % de défauts.

Le projet couvre la chaîne complète : analyse exploratoire, nettoyage justifié, feature
engineering, comparaison systématique de modèles, tuning, choix du seuil par coût métier,
avec versionnement des données (DVC) et traçabilité des expériences (MLflow).

## Résultats

| | Version 1 | Version 2 | Gain |
|---|---|---|---|
| PR-AUC (test) | 0.5432 | **0.5645** | +0.0213 (+3.9 %) |
| ROC-AUC (test) | 0.7661 | **0.7822** | +0.0161 (+2.1 %) |

Modèle retenu : **XGBoost tuné** (517 arbres, `depth=4`, `lr=0.0102`, `gamma=4.3`,
`scale_pos_weight=3.521`), sélectionné parmi 11 modèles comparés sur un protocole identique.

Le gain vient **entièrement du tuning d'hyperparamètres**. Trois autres pistes ont été testées
et n'ont rien apporté — ce sont des résultats mesurés, pas des impressions :

- **Feature engineering** : 4 jeux de features (23 variables brutes → 68 enrichies) donnent des
  scores indiscernables (écarts inférieurs à un écart-type entre folds).
- **Deep learning séquentiel** : LSTM (0.5482), GRU (0.5477) et MLP (0.5478) se tiennent en
  0.0005 et restent sous le boosting tuné, alors que l'architecture exploite explicitement les
  6 mois d'historique.
- **Ensembles** : le meilleur blend parmi 682 combinaisons gagne +0.0014, c'est-à-dire rien.

### Le plafond est dans les données, pas dans les modèles

Trois tests indépendants convergent :

1. La corrélation de rang entre les prédictions des 11 modèles atteint **0.907** en moyenne
   (minimum 0.800 pour la paire la plus diverse) : boosting, forêt aléatoire, réseau de neurones
   et régression logistique classent les clients dans le même ordre.
2. Le blending, qui exploite précisément les désaccords entre modèles, ne gagne rien — donc il
   n'y a pas de désaccord exploitable.
3. XGBoost et LightGBM tunés atterrissent tous deux sur 0.5650, au millième près, depuis des
   implémentations et des espaces de recherche indépendants.

Le modèle reste utile : il sépare un groupe à **4.5 %** de risque d'un groupe à **70.2 %**
(rapport de 15×) et il est bien calibré. Mais 30 % des clients du groupe le plus risqué ne font
pas défaut, et rien dans les variables disponibles ne permet de les distinguer.
**Pour progresser, il faut d'autres données** (revenus, ancienneté bancaire, encours dans
d'autres établissements), pas d'autres algorithmes.

## Structure

```
data/raw/            Données source (suivies par DVC)
notebooks/           01_eda_analyse_donnees.ipynb — EDA et pipeline de base
src/
  data_prep.py       Chargement, nettoyage, feature engineering (tabulaire + séquentiel)
  tracking.py        Configuration MLflow centralisée
scripts/
  exp_ablation.py    Diagnostic : effet du feature engineering à modèle constant
  run_experiments.py Comparaison des 11 modèles + tuning (reprenable après interruption)
  train_final.py     Entraîne, sérialise et enregistre le modèle retenu
  final_eval.py      Évaluation sur le test + seuils par coût métier
  mlflow_backfill.py Réinjecte l'historique dans MLflow sans réentraîner
  build_report.py    Génère le rapport PDF
models/              Modèle sérialisé + carte du modèle (DVC)
reports/             Métriques, figures, rapport PDF, prédictions out-of-fold
params.yaml          Paramètres du pipeline (suivis par DVC)
dvc.yaml             Définition du pipeline reproductible
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Les artefacts (données, modèles, prédictions out-of-fold) sont stockés sur
[DagsHub](https://dagshub.com/abdelmoughitelhassani2/Pipeline_MLOps_pred_retard_paiement).
Pour les récupérer, il faut un jeton personnel
([dagshub.com/user/settings/tokens](https://dagshub.com/user/settings/tokens)) :

```bash
dvc remote modify dagshub --local user <ton_utilisateur>
dvc remote modify dagshub --local password <ton_jeton>
dvc pull
```

Le drapeau `--local` écrit dans `.dvc/config.local`, volontairement ignoré par git :
**aucun identifiant ne doit se retrouver dans un dépôt public**.

## Utilisation

### Pipeline reproductible

```bash
dvc repro                 # relance uniquement les étapes dont une dépendance a changé
dvc repro train           # une étape précise
dvc metrics show          # métriques du modèle courant
dvc dag                   # visualise le graphe de dépendances
```

Modifier un hyperparamètre dans `params.yaml` suffit à invalider les étapes concernées :
`dvc repro` les relancera seules.

> L'étape `experiments` prend environ 40 minutes (11 modèles × 5 folds, dont du deep learning).
> Elle est reprenable : les prédictions out-of-fold déjà calculées sont conservées et ignorées
> au relancement.

### Historique des expériences

L'historique est consultable en ligne dans l'onglet **Experiments** du
[dépôt DagsHub](https://dagshub.com/abdelmoughitelhassani2/Pipeline_MLOps_pred_retard_paiement),
ou en local :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://localhost:5000
```

Pour enregistrer de nouveaux runs directement sur le serveur distant, il suffit de définir
trois variables d'environnement avant de lancer un script — le code bascule alors tout seul
du mode local au mode distant (voir `src/tracking.py`) :

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<jeton>
```

L'historique complet est déjà enregistré — **40 runs**, réinjectés depuis les fichiers de
résultats sans aucun réentraînement :

| Expérience | Runs | Contenu |
|---|---|---|
| `phase1-strategies` | 15 | Combinaisons modèle × stratégie de rééquilibrage |
| `phase2-ablation` | 4 | Jeux de features comparés à modèle constant |
| `phase2-models` | 11 | Modèles comparés (boosting, forêts, deep learning) |
| `phase2-blends` | 10 | Meilleurs ensembles |
| `production` | 1 | Modèle déployé, enregistré au registre |

Chaque run porte ses hyperparamètres, ses métriques, sa famille, son **architecture**, sa
**méthode d'entraînement** et une note expliquant ce qu'il a appris. Interroger l'historique
en Python :

```python
import mlflow, sys
sys.path.insert(0, "src"); import tracking
tracking.setup()

exp = mlflow.get_experiment_by_name("phase2-models")
df = mlflow.search_runs([exp.experiment_id], output_format="pandas")
print(df[["tags.mlflow.runName", "metrics.pr_auc", "tags.architecture"]])
```

Le script de backfill est idempotent : le relancer n'ajoute aucun doublon.

### Réentraîner

```bash
python scripts/train_final.py
```

Produit `models/final_model.joblib`, `models/model_card.json` (config, métriques, seuil,
description des données) et une nouvelle version au registre MLflow `credit-default-xgb`.

## Versionnement des données

Les fichiers volumineux (données source, prédictions out-of-fold, modèles, base MLflow) sont
suivis par DVC, pas par git. Git ne contient que les fichiers `.dvc` porteurs du hash de
contenu, ce qui permet de retrouver la version exacte d'un artefact à n'importe quel commit.

```bash
dvc push        # envoie les artefacts vers le stockage distant
dvc pull        # les récupère
dvc status      # compare l'état local au pipeline
```

Ce découplage a une utilité concrète : le fichier source a été modifié en cours de projet
(5 539 328 → 5 540 352 octets) sans qu'aucune donnée ne change — seules les métadonnées Excel
avaient bougé. Git signalait un fichier modifié là où le contenu était identique.

## Choix du seuil de décision

Le seuil n'est pas figé par une formule mais par un arbitrage métier. Seul le **rapport**
R = coût(défaut manqué) / coût(fausse alerte) importe :

| R | Seuil | Recall | Precision | % du portefeuille flagué |
|---|---|---|---|---|
| 2 | 0.630 | 52 % | 57 % | 20 % |
| 3 | 0.530 | 60 % | 49 % | 27 % |
| 5 | 0.370 | 80 % | 35 % | 51 % |
| 10 | 0.245 | 94 % | 28 % | 76 % |

Une approche antérieure maximisait le F2 et aboutissait à un seuil de 0.258 flaguant 60 % du
portefeuille — statistiquement défendable, opérationnellement inutilisable.

## Rigueur méthodologique

- Le **test set (6 000 clients) n'intervient dans aucune décision** : ni sélection de features,
  ni choix de modèle, ni tuning, ni calibration de seuil. Il n'est évalué qu'une fois.
- Le rééquilibrage est appliqué **à l'intérieur de chaque fold** de validation croisée, jamais
  avant le découpage, pour éviter toute fuite.
- Les seuils sont calibrés sur des probabilités **out-of-fold** du train.
- `ID` n'est jamais utilisé comme variable ni comme base de rééquilibrage.

## Limites connues

- Un BiLSTM et un Transformer étaient prévus mais n'ont pas pu être entraînés : la machine
  (4 cœurs, RAM limitée) a interrompu les processus. Au vu de la convergence des trois autres
  architectures, leur résultat serait probablement identique — mais cela reste une attente
  raisonnée, non un résultat mesuré.
- Le tuning s'est limité à 20 tirages de `RandomizedSearchCV`. Une recherche plus poussée
  (Optuna, davantage d'itérations) pourrait apporter quelques millièmes supplémentaires, sans
  changer les conclusions.
