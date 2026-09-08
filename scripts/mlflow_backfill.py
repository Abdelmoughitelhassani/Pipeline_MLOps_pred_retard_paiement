"""Réinjecte tout l'historique d'expérimentation dans MLflow — sans réentraîner.

Tous les résultats produits jusqu'ici sont conservés dans des fichiers (CSV/JSON/NPY).
Ce script les relit et crée un run MLflow par expérience, avec ses paramètres, ses
métriques, son architecture et sa méthode. Objectif : pouvoir répondre plus tard à
« qu'est-ce qu'on a déjà essayé, avec quelle config, et qu'est-ce que ça donnait ? »
sans relancer un seul entraînement.

Quatre expériences MLflow sont créées :

  1. phase1-strategies   15 combinaisons modèle x stratégie de rééquilibrage
  2. phase2-ablation     4 jeux de features, même modèle (diagnostic du goulot)
  3. phase2-models       11 modèles comparés (boosting, forêts, deep learning)
  4. phase2-blends       les meilleurs ensembles évalués sur les prédictions OOF

Le script est idempotent : relancé, il ne duplique pas les runs déjà enregistrés.

Usage :
    python scripts/mlflow_backfill.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # puis http://localhost:5000
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPORTS = ROOT / "reports"

import tracking  # noqa: E402

tracking.setup()

# Description de chaque modèle : famille, architecture, méthode d'entraînement.
# C'est ce qui permet de retrouver « comment » un modèle a été obtenu, pas seulement son score.
MODEL_CARDS = {
    "logreg": dict(family="Linéaire", architecture="LogisticRegression + StandardScaler",
                   library="scikit-learn", imbalance="class_weight=balanced",
                   notes="Référence linéaire. Plafonne quelle que soit la stratégie."),
    "rf": dict(family="Bagging", architecture="RandomForest, 300 arbres, profondeur illimitée",
               library="scikit-learn", imbalance="class_weight=balanced",
               notes="Diversité utile en ensemble mais performance individuelle moyenne."),
    "histgb": dict(family="Boosting", architecture="HistGradientBoosting, 300 itérations (défaut)",
                   library="scikit-learn", imbalance="class_weight=balanced",
                   notes="Très bon rapport performance/temps sans aucun tuning (16 s)."),
    "histgb_tuned": dict(family="Boosting tuné",
                         architecture="HistGradientBoosting optimisé (RandomizedSearchCV)",
                         library="scikit-learn", imbalance="class_weight=balanced",
                         notes="Tuning 20 tirages, 3-fold, scoring=average_precision."),
    "xgb_base": dict(family="Boosting",
                     architecture="XGBoost 300 arbres, depth=5, lr=0.1",
                     library="xgboost", imbalance="scale_pos_weight=3.521",
                     notes="Paramètres initiaux du notebook — sur-apprentissage identifié."),
    "xgb_tuned": dict(family="Boosting tuné",
                      architecture="XGBoost 517 arbres, depth=4, lr=0.0102, gamma=4.3",
                      library="xgboost", imbalance="scale_pos_weight=3.521",
                      notes="MODÈLE RETENU. Régularisation forte + lr bas corrigent le "
                            "sur-apprentissage. Meilleur PR-AUC out-of-fold."),
    "lgbm_tuned": dict(family="Boosting tuné",
                       architecture="LightGBM 508 arbres, num_leaves=16, lr=0.0163, L2=10.9",
                       library="lightgbm", imbalance="scale_pos_weight=3.521",
                       notes="Ex aequo avec XGBoost tuné au millième — indice de plafond."),
    "catboost": dict(family="Boosting", architecture="CatBoost, 500 itérations (défaut)",
                     library="catboost", imbalance="scale_pos_weight=3.521",
                     notes="Bon sans tuning, mais plus lent que HistGB pour un score voisin."),
    "mlp": dict(family="Deep learning",
                architecture="MLP 68->128->64->1, BatchNorm, Dropout 0.3",
                library="pytorch", imbalance="pos_weight dans la BCE",
                notes="Adam lr=1e-3, early stopping sur PR-AUC de validation interne."),
    "lstm": dict(family="Deep learning",
                 architecture="LSTM(5->64) sur 6 mois chronologiques + branche statique(11->32)",
                 library="pytorch", imbalance="pos_weight dans la BCE",
                 notes="Exploite la séquence temporelle réelle. N'améliore pas le MLP : "
                       "l'information temporelle est déjà captée par les agrégats."),
    "gru": dict(family="Deep learning",
                architecture="GRU(5->64) sur 6 mois chronologiques + branche statique(11->32)",
                library="pytorch", imbalance="pos_weight dans la BCE",
                notes="Identique au LSTM à 0.0005 près."),
}

PHASE1_CARDS = {
    "none": "Aucun traitement du déséquilibre (référence)",
    "class_weight": "Pondération de la classe minoritaire — aucune donnée modifiée",
    "random_oversampling": "Duplication de lignes de la classe minoritaire (train uniquement)",
    "random_undersampling": "Sous-échantillonnage de la classe majoritaire",
    "smotenc": "Génération d'exemples synthétiques, catégorielles gérées (SMOTENC)",
}


def existing_run_names(experiment_name: str) -> set[str]:
    """Noms des runs déjà présents — rend le script rejouable sans doublon."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return set()
    df = mlflow.search_runs([exp.experiment_id], output_format="pandas")
    if df.empty or "tags.mlflow.runName" not in df:
        return set()
    return set(df["tags.mlflow.runName"].dropna())


def log_run(name: str, params: dict, metrics: dict, tags: dict) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


# --------------------------------------------------------------------- 1. phase 1
def backfill_phase1() -> int:
    exp = "phase1-strategies"
    tracking.set_experiment(exp)
    done = existing_run_names(exp)
    df = pd.read_csv(REPORTS / "phase1_strategy_results.csv")
    n = 0
    for _, r in df.iterrows():
        name = f"{r['model']}__{r['imbalance_strategy']}"
        if name in done:
            continue
        log_run(
            name,
            params={"model": r["model"], "imbalance_strategy": r["imbalance_strategy"],
                    "feature_set": "v1 (23 features)", "cv": "StratifiedKFold(5)",
                    "n_features": 23, "leak_control": "resampling dans le pipeline CV"},
            metrics={"pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"], "f1": r["f1"],
                     "recall": r["recall"], "precision": r["precision"]},
            tags={"phase": "1 - stratégies de rééquilibrage",
                  "method": PHASE1_CARDS.get(r["imbalance_strategy"], ""),
                  "metric_type": "moyenne des folds"},
        )
        n += 1
    return n


# --------------------------------------------------------------------- 2. ablation
def backfill_ablation() -> int:
    exp = "phase2-ablation"
    tracking.set_experiment(exp)
    done = existing_run_names(exp)
    df = pd.read_csv(REPORTS / "ablation_results.csv")
    n = 0
    for _, r in df.iterrows():
        name = r["features"].split(".")[0].strip() + " - " + str(r["n_features"]) + " features"
        if name in done:
            continue
        log_run(
            name,
            params={"feature_set": r["features"], "n_features": int(r["n_features"]),
                    "model": "XGBoost (identique pour tous)", "cv": "StratifiedKFold(5)"},
            metrics={"pr_auc": r["pr_auc"], "pr_auc_std": r["pr_auc_std"],
                     "roc_auc": r["roc_auc"], "roc_auc_std": r["roc_auc_std"]},
            tags={"phase": "2 - ablation features",
                  "objective": "isoler l'effet du feature engineering à modèle constant",
                  "conclusion": "les 4 jeux sont indiscernables : écarts < 1 écart-type"},
        )
        n += 1
    return n


# --------------------------------------------------------------------- 3. modèles
def backfill_models() -> int:
    exp = "phase2-models"
    tracking.set_experiment(exp)
    done = existing_run_names(exp)
    ranking = pd.read_csv(REPORTS / "model_ranking.csv")
    best_params = json.loads((REPORTS / "best_params.json").read_text(encoding="utf-8"))
    tuned_key = {"xgb_tuned": "xgboost", "lgbm_tuned": "lightgbm", "histgb_tuned": "histgb"}
    n = 0
    for _, r in ranking.iterrows():
        key = r["model"]
        if key in done:
            continue
        card = MODEL_CARDS.get(key, {})
        params = {"model": key, "feature_set": "v2 (68 features)", "n_features": 68,
                  "cv": "StratifiedKFold(5), prédictions out-of-fold",
                  "library": card.get("library", ""),
                  "imbalance_strategy": card.get("imbalance", "")}
        if key in tuned_key:
            params.update({f"hp_{k}": v for k, v in best_params[tuned_key[key]].items()})
            params["tuning"] = "RandomizedSearchCV, 20 tirages, 3-fold, average_precision"
        log_run(
            key, params,
            metrics={"pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"]},
            tags={"phase": "2 - comparaison de modèles", "family": card.get("family", ""),
                  "architecture": card.get("architecture", ""), "notes": card.get("notes", ""),
                  "metric_type": "out-of-fold agrégé"},
        )
        n += 1
    return n


# --------------------------------------------------------------------- 4. ensembles
def backfill_blends(top: int = 10) -> int:
    exp = "phase2-blends"
    tracking.set_experiment(exp)
    done = existing_run_names(exp)
    df = pd.read_csv(REPORTS / "blend_ranking.csv").head(top)
    best_single = pd.read_csv(REPORTS / "model_ranking.csv")["pr_auc"].max()
    n = 0
    for i, r in df.iterrows():
        name = f"blend_{i+1:02d}"
        if name in done:
            continue
        log_run(
            name,
            params={"members": r["blend"], "n_members": int(r["k"]),
                    "method": "moyenne des rangs des prédictions out-of-fold"},
            metrics={"pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
                     "gain_vs_best_single": r["pr_auc"] - best_single},
            tags={"phase": "2 - ensembles",
                  "conclusion": "gain négligeable : les modèles se trompent sur les mêmes "
                                "clients (corrélation inter-modèles 0.907)"},
        )
        n += 1
    return n


def main() -> None:
    print(f"Tracking MLflow : {mlflow.get_tracking_uri()}\n")
    counts = {
        "phase1-strategies": backfill_phase1(),
        "phase2-ablation": backfill_ablation(),
        "phase2-models": backfill_models(),
        "phase2-blends": backfill_blends(),
    }
    for exp, n in counts.items():
        print(f"  {exp:<22} {n} run(s) ajouté(s)" if n else f"  {exp:<22} déjà à jour")
    print(f"\nTotal : {sum(counts.values())} nouveaux runs")
    print("\nConsulter l'historique :  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
