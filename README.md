# Prédiction de retard de paiement — Pipeline MLOps

[![CI](https://github.com/Abdelmoughitelhassani/Pipeline_MLOps_pred_retard_paiement/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Abdelmoughitelhassani/Pipeline_MLOps_pred_retard_paiement/actions/workflows/ci.yml)

Prédiction du défaut de paiement de clients de cartes de crédit à partir de leur historique
de facturation et de remboursement, sur le jeu de données
[Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
— 30 000 clients à Taïwan, 22.12 % de défauts.

Le projet couvre la chaîne complète : analyse exploratoire, nettoyage justifié, feature
engineering, comparaison systématique de modèles, tuning, choix du seuil par coût métier,
avec versionnement des données (DVC) et traçabilité des expériences (MLflow).

## Résultats

L'analyse détaillée est consignée dans le
**[rapport complet (PDF, 31 pages)](reports/rapport_pipeline_credit_default.pdf)** :
méthodologie, comparaison des modèles, analyse du plafond de performance et choix du seuil.

| | Version 1 | Version 2 | Gain |
|---|---|---|---|
| PR-AUC (test) | 0.5432 | **0.5645** | +0.0213 (+3.9 %) |
| ROC-AUC (test) | 0.7661 | **0.7822** | +0.0161 (+2.1 %) |

Modèle retenu : **XGBoost tuné** (517 arbres, `depth=4`, `lr=0.0102`, `gamma=4.3`,
`scale_pos_weight=3.521`), sélectionné parmi 11 modèles comparés sur un protocole identique.

Le gain vient **entièrement du tuning d'hyperparamètres**. Cinq autres pistes ont été testées
et n'ont rien apporté — ce sont des résultats mesurés, pas des impressions :

- **Feature engineering** : 4 jeux de features (23 variables brutes → 68 enrichies) donnent des
  scores indiscernables (écarts inférieurs à un écart-type entre folds).
- **Deep learning séquentiel** : LSTM (0.5482), GRU (0.5477) et MLP (0.5478) se tiennent en
  0.0005 et restent sous le boosting tuné, alors que l'architecture exploite explicitement les
  6 mois d'historique.
- **Ensembles** : le meilleur blend parmi 682 combinaisons gagne +0.0014, c'est-à-dire rien.
- **Sélection de features (Boruta)** : retient 58 variables sur 68, pour une PR-AUC de 0.5620
  contre 0.5650 — aucune amélioration.
- **Optimisation bayésienne (Optuna TPE, 100 essais)** : gagne +0.0030 en validation interne
  mais **le gain ne se transfère pas** au test set (0.5620 contre 0.5645). Cent essais sur
  les mêmes plis suffisent à sur-ajuster la validation croisée.

### Ce que Boruta a révélé d'utile

Boruta rejette **les quatre variables démographiques** — `SEX`, `EDUCATION`, `MARRIAGE`,
`AGE` — dont l'importance n'excède pas celle de leurs versions permutées aléatoirement.
Les retirer coûte 0.0030 de PR-AUC, soit rien de mesurable.

C'est exploitable : dans beaucoup de juridictions, l'usage du sexe ou de la situation
matrimoniale dans un score de crédit est illégal ou strictement encadré. On dispose ici
d'une **démonstration chiffrée** qu'un modèle conforme ne perdrait pas en performance.
La liste des variables retenues est dans `reports/boruta_features.json`, activable via
`feature_selection.method` dans `params.yaml`.

### Le plafond est dans les données, pas dans les modèles

Trois tests indépendants convergent :

1. La corrélation de rang entre les prédictions des 11 modèles atteint **0.907** en moyenne
   (minimum 0.800 pour la paire la plus diverse) : boosting, forêt aléatoire, réseau de neurones
   et régression logistique classent les clients dans le même ordre.
2. Le blending, qui exploite précisément les désaccords entre modèles, ne gagne rien — donc il
   n'y a pas de désaccord exploitable.
3. XGBoost et LightGBM tunés atterrissent tous deux sur 0.5650, au millième près, depuis des
   implémentations et des espaces de recherche indépendants.
4. Un optimiseur bayésien (Optuna) et un sélecteur de variables (Boruta) échouent tous deux
   à franchir ce plafond, alors qu'ils attaquent le problème par des angles différents.

Le modèle reste utile : il sépare un groupe à **4.5 %** de risque d'un groupe à **70.2 %**
(rapport de 15×) et il est bien calibré. Mais 30 % des clients du groupe le plus risqué ne font
pas défaut, et rien dans les variables disponibles ne permet de les distinguer.
**Pour progresser, il faut d'autres données** (revenus, ancienneté bancaire, encours dans
d'autres établissements), pas d'autres algorithmes.

## Structure

Le rôle détaillé de chaque fichier est documenté dans
**[GUIDE_DES_FICHIERS.md](GUIDE_DES_FICHIERS.md)**.

```
data/raw/            Données source, jamais modifiées (DVC)
data/processed/      train.parquet et test.parquet, produits par l'étape prepare (DVC)
notebooks/           01_eda_analyse_donnees.ipynb — EDA et pipeline de base
src/
  data_prep.py       Chargement, nettoyage, feature engineering, matérialisation
  tracking.py        Configuration MLflow centralisée
scripts/
  prepare_data.py    data/raw/ -> data/processed/ (seule étape lisant le brut)
  exp_ablation.py    Diagnostic : effet du feature engineering à modèle constant
  exp_boruta.py      Sélection de features par Boruta (variables fantômes)
  exp_optuna.py      Tuning bayésien Optuna, comparé à RandomizedSearchCV
  exp_outliers.py    Détection d'outliers : Isolation Forest, LOF, DBSCAN
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
dvc repro prepare         # data/raw/ -> data/processed/ (train.parquet, test.parquet)
dvc repro train           # une étape précise
dvc metrics show          # métriques du modèle courant
dvc dag                   # visualise le graphe de dépendances
```

Modifier un hyperparamètre dans `params.yaml` suffit à invalider les étapes concernées :
`dvc repro` les relancera seules.

**Une seule étape lit les données brutes** — `prepare` — et écrit `data/processed/`
au format Parquet (réglable par `data.processed_format`). Tout l'aval consomme ces
fichiers préparés. Le pipeline reste donc identique quel que soit le volume : on
prépare une fois, puis on lit un format colonnaire compressé plutôt que de
reconstruire les features à chaque exécution.

> L'étape `experiments` prend environ 40 minutes (11 modèles × 5 folds, dont du deep learning).
> Elle est reprenable : les prédictions out-of-fold déjà calculées sont conservées et ignorées
> au relancement.

### Servir le modèle

```bash
uvicorn src.serve:app --reload --port 8000
```

| Route | Rôle |
|---|---|
| `GET /health` | État du service, nom du modèle, seuil appliqué |
| `POST /predict` | Probabilité de défaut et niveau de risque (HIGH/LOW) |
| `GET /predict/example` | Deux profils de test, prêts à copier-coller |

Documentation interactive sur `http://localhost:8000/docs`. L'API prend les **23 variables
brutes** du dossier client et calcule les features côté serveur, via le même code qu'à
l'entraînement — c'est ce qui garantit l'absence de décalage entraînement/service.

Exemple complet — un client au comportement de paiement dégradé :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_1": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
    "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
  }'
```

Réponse :

```json
{
  "default_probability": 0.8875,
  "risk_level": "HIGH",
  "threshold": 0.55,
  "model_name": "xgboost",
  "model_trained_at": "2026-09-09T16:45:54"
}
```

Ce client cumule deux mois de retard consécutifs (`PAY_1` et `PAY_2` à 2) pour un plafond
faible : le modèle lui attribue 88.75 % de probabilité de défaut, au-dessus du seuil de
0.55, d'où le niveau `HIGH`. Un profil régulier est disponible via `GET /predict/example`.

### Conteneurisation

```bash
docker compose up --build -d          # http://localhost:8000
docker compose ps                     # état et santé
docker compose logs -f scoring
docker compose down
```

L'image ne contient que les dépendances d'inférence (`requirements-serve.txt`) : ni PyTorch,
ni MLflow, ni DVC, ni CatBoost/LightGBM. Elle utilise aussi `xgboost-cpu` plutôt que
`xgboost`, dont la roue Linux embarque 476 Mo de bibliothèques CUDA inutiles à un service
CPU — **l'image passe ainsi de 1.92 Go à 713 Mo**, à prédictions rigoureusement identiques.

Le conteneur tourne sans privilèges, expose une sonde de vivacité sur `/health` et se voit
limité à 2 CPU et 2 Go de mémoire.

> **Prérequis** : `models/final_model.joblib` est suivi par DVC, pas par git. Sur un dépôt
> fraîchement cloné, lancer `dvc pull` avant `docker compose build`.

> **Conflit de port** : si un `uvicorn` local tourne déjà sur le port 8000, Windows lui
> route les appels à `127.0.0.1:8000` de préférence au conteneur (liaison plus spécifique).
> Arrêter le serveur local, ou publier le conteneur sur un autre port.

### Intégration continue

À chaque push sur `main`, `.github/workflows/ci.yml` enchaîne :

1. **Tests** — `pytest tests/` (19 tests)
2. **Porte de qualité** — la PR-AUC de `metrics/scores.json` doit dépasser 0.54
3. **Build et publication** — image Docker poussée sur GitHub Container Registry, puis
   vérifiée en la démarrant et en interrogeant `/health`
4. **Déploiement** — mise en ligne sur Render, uniquement sur `main` et si
   `RENDER_ENABLED` vaut `true` (voir *Déploiement sur Render* ci-dessous)

La porte de qualité s'exécute **avant** le build : si le modèle se dégrade, aucune image
n'est publiée. Le job Docker dépend d'elle (`needs: quality-gate`), il n'est donc même pas
lancé en cas d'échec.

> **Secrets requis** dans *Settings → Secrets and variables → Actions* :
> `DAGSHUB_USER` et `DAGSHUB_TOKEN`. Le modèle est suivi par DVC et absent de git ; sans
> ces secrets, le job Docker échoue explicitement plutôt que de publier une image sans modèle.

### Déploiement sur Render

Le job `deploy` s'exécute après la publication de l'image, uniquement sur `main`, et
uniquement si la variable `RENDER_ENABLED` vaut `true`. Sans ce drapeau il apparaît
**skipped** — plutôt qu'un vert trompeur signalant un déploiement qui n'a pas eu lieu.

**1. Rendre l'image accessible à Render.** Les paquets GHCR sont privés par défaut. Sur la
page du paquet (*Profil → Packages → Pipeline_MLOps_pred_retard_paiement*), ouvrir
*Package settings → Change visibility → Public*. Sans cela Render ne pourra pas la tirer,
et l'erreur apparaîtra côté Render, pas dans GitHub Actions.

**2. Créer le service sur Render.** *New → Web Service → Existing image*, avec :

| Champ | Valeur |
|---|---|
| Image URL | `ghcr.io/abdelmoughitelhassani/pipeline_mlops_pred_retard_paiement:latest` |
| Port | `8000` |
| Health check path | `/health` |

**3. Récupérer les identifiants.** L'identifiant du service se lit dans l'URL du tableau de
bord (`https://dashboard.render.com/web/srv-XXXXXXXX`), la clé d'API se crée dans
*Account Settings → API Keys*.

**4. Déclarer les secrets et variables** dans *Settings → Secrets and variables → Actions* :

| Nom | Type | Valeur |
|---|---|---|
| `RENDER_API_KEY` | Secret | Clé d'API Render |
| `RENDER_SERVICE_ID` | Secret | `srv-XXXXXXXX` |
| `RENDER_ENABLED` | Variable | `true` |
| `RENDER_SERVICE_URL` | Variable | `https://mon-service.onrender.com` |

`RENDER_SERVICE_URL` est facultative : si elle est absente, la vérification finale est
ignorée mais le déploiement reste piloté et vérifié côté Render.

Le job déclenche le déploiement via l'API REST, **interroge son statut toutes les 15
secondes** jusqu'à un état terminal (délai maximal de 15 minutes), puis appelle `/health`
sur l'URL publique. Un simple *deploy hook* aurait rendu la main immédiatement, sans
permettre de savoir si le déploiement avait abouti.

> **Note sur le plan gratuit** : Render met le service en veille après 15 minutes
> d'inactivité. La première requête suivante peut prendre 30 à 60 secondes, le temps du
> réveil — ce qui peut faire échouer la vérification `/health` si le service dormait.

### Supervision de la dérive

```bash
python monitoring/drift_report.py                                  # simulation
python monitoring/drift_report.py --batch lot.parquet --fail-on-drift
python monitoring/drift_report.py --batch lot.parquet --notify-slack
```

`--notify-slack` publie un résumé (nombre de colonnes dérivées, top 5 des variables, lien
vers le rapport) via le webhook lu dans `SLACK_WEBHOOK_URL`. Sans cette variable, le script
continue normalement — et une notification qui échoue ne fait jamais échouer le contrôle.

Compare chaque variable d'un lot entrant à sa distribution dans le jeu d'entraînement.
Produit un rapport HTML lisible et une synthèse JSON exploitable par une supervision.

**Attention au seuil d'alerte.** Evidently considère par convention qu'un jeu a dérivé
au-delà de 50 % de colonnes touchées. Sur ce projet, un lot simulant une récession
(plafonds réduits de 55 %, retards aggravés, remboursements divisés par trois) ne fait
dériver que 20.6 % des 68 colonnes — beaucoup sont des agrégats dérivés qui diluent la
part. Avec le seuil par défaut, **aucune alerte ne serait levée**. Le paramètre
`--drift-share-threshold` permet d'abaisser ce seuil ; 0.15 déclenche correctement sur ce
scénario.

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
