"""Entraîne le modèle final, le sérialise et l'enregistre dans MLflow.

C'est le point d'entrée pour (ré)entraîner le modèle retenu. Les hyperparamètres viennent
de `params.yaml` : les modifier là-bas suffit, aucun changement de code n'est nécessaire,
et `dvc repro` relancera automatiquement cette étape.

Sorties :
    models/final_model.joblib     pipeline sérialisé (suivi par DVC)
    models/model_card.json        carte du modèle : config, métriques, seuils, données
    mlflow.db / mlartifacts/      run MLflow + modèle enregistré au registre

Usage :
    python scripts/train_final.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402
import tracking  # noqa: E402

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
tracking.setup()


def pick_threshold(y_true: np.ndarray, proba: np.ndarray, ratio: float) -> dict:
    """Seuil minimisant `ratio x faux_négatifs + faux_positifs`.

    Seul le rapport de coût importe, pas les valeurs absolues : la banque n'a donc pas
    besoin de chiffrer précisément ses coûts pour utiliser ce réglage.
    """
    best = None
    for t in np.linspace(0.05, 0.95, 181):
        pred = (proba >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = ratio * fn + fp
        if best is None or cost < best["cost"]:
            best = {"threshold": round(float(t), 3), "cost": cost,
                    "recall": float(recall_score(y_true, pred)),
                    "precision": float(precision_score(y_true, pred, zero_division=0)),
                    "pct_flagged": float(pred.mean())}
    return best


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    cfg, cv_cfg = params["final_model"], params["cv"]
    ratio = params["threshold"]["default_ratio"]

    # Lit data/processed/ produit par l'étape `prepare` ; retombe sur un calcul en
    # mémoire si le pipeline n'a pas encore été exécuté (dépôt fraîchement cloné).
    fmt = params["data"].get("processed_format", "parquet")
    X_train, y_train, X_test, y_test, feature_cols = dp.get_splits(fmt=fmt)

    n_neg, n_pos = np.bincount(y_train)
    spw = n_neg / n_pos
    model = XGBClassifier(**cfg["params"], scale_pos_weight=spw,
                          eval_metric="logloss", random_state=42, n_jobs=2)

    # Seuil calibré sur des probabilités out-of-fold du train : le test reste vierge.
    cv = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg["shuffle"],
                         random_state=cv_cfg["random_state"])
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    thr_info = pick_threshold(y_train.to_numpy(), oof, ratio)
    oof_metrics = {"oof_pr_auc": float(average_precision_score(y_train, oof)),
                   "oof_roc_auc": float(roc_auc_score(y_train, oof))}
    print(f"OOF train : PR-AUC={oof_metrics['oof_pr_auc']:.4f} "
          f"ROC-AUC={oof_metrics['oof_roc_auc']:.4f}")
    print(f"Seuil retenu (coût R={ratio}) : {thr_info['threshold']} "
          f"-> recall={thr_info['recall']:.1%}, precision={thr_info['precision']:.1%}, "
          f"{thr_info['pct_flagged']:.1%} du portefeuille")

    # Entraînement final et évaluation unique sur le test
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= thr_info["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    test_metrics = {
        "test_pr_auc": float(average_precision_score(y_test, proba)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_recall": float(recall_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred)),
    }
    print(f"TEST : PR-AUC={test_metrics['test_pr_auc']:.4f} "
          f"ROC-AUC={test_metrics['test_roc_auc']:.4f}")

    joblib.dump({"model": model, "feature_cols": feature_cols,
                 "threshold": thr_info["threshold"]}, MODELS / "final_model.joblib")

    card = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model": cfg["name"],
        "hyperparameters": cfg["params"],
        "imbalance_strategy": f"scale_pos_weight={spw:.3f}",
        "n_features": len(feature_cols),
        "features": feature_cols,
        "data": {"source": str(dp.DEFAULT_DATA_PATH.relative_to(ROOT)),
                 "n_total": int(len(y_train) + len(y_test)), "n_train": int(len(y_train)),
                 "n_test": int(len(y_test)),
                 "default_rate": float(y_train.mean())},
        "threshold": {"cost_ratio_FN_FP": ratio, **thr_info},
        "metrics": {**oof_metrics, **test_metrics},
        "confusion_matrix_test": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    # Enregistrement MLflow + registre de modèles
    tracking.set_experiment("production")
    with mlflow.start_run(run_name=f"final_{cfg['name']}"):
        mlflow.log_params({**{f"hp_{k}": v for k, v in cfg["params"].items()},
                           "model": cfg["name"], "n_features": len(feature_cols),
                           "imbalance_strategy": "scale_pos_weight",
                           "threshold": thr_info["threshold"], "cost_ratio": ratio})
        mlflow.log_metrics({**oof_metrics, **test_metrics})
        mlflow.set_tags({"stage": "production", "family": "Boosting tuné",
                         "architecture": f"XGBoost {cfg['params']['n_estimators']} arbres, "
                                         f"depth={cfg['params']['max_depth']}",
                         "selected_because": "meilleur PR-AUC out-of-fold parmi 11 modèles"})
        mlflow.log_artifact(str(MODELS / "model_card.json"))
        mlflow.xgboost.log_model(model, name="model", registered_model_name="credit-default-xgb")

    print(f"\nModèle      -> {MODELS / 'final_model.joblib'}")
    print(f"Carte       -> {MODELS / 'model_card.json'}")
    print("Registre    -> MLflow, modèle « credit-default-xgb »")


if __name__ == "__main__":
    main()
