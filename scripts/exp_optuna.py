"""Phase 3b — optimisation d'hyperparamètres avec Optuna.

Remplace `RandomizedSearchCV` (20 tirages purement aléatoires) par l'échantillonneur TPE
d'Optuna, qui modélise la relation entre hyperparamètres et score pour concentrer la
recherche dans les régions prometteuses. Un élagage médian arrête tôt les essais mal partis.

Protocole de comparaison, volontairement équitable :
  - même espace de recherche que RandomizedSearchCV (on isole l'effet de l'ALGORITHME)
  - même validation croisée interne (3-fold) et même métrique (average_precision)
  - le score « à 20 essais » est extrait de l'historique Optuna, ce qui donne gratuitement
    une comparaison à budget égal contre les 20 tirages de RandomizedSearchCV
  - les meilleurs paramètres sont ensuite réévalués en 5-fold out-of-fold, protocole commun
    à toutes les phases, seul chiffre directement comparable aux résultats précédents
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402
import tracking  # noqa: E402

import mlflow  # noqa: E402

SEED = 42
N_TRIALS = 100
REPORTS = ROOT / "reports"
CV_INNER = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
CV_OUTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Scores de référence obtenus par RandomizedSearchCV (20 tirages, 3-fold)
BASELINE_RANDOM_SEARCH = {"xgboost": 0.5641, "lightgbm": 0.5647}


def suggest_xgb(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def suggest_lgbm(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
    }


def build(name: str, params: dict, spw: float):
    if name == "xgboost":
        return XGBClassifier(**params, scale_pos_weight=spw, eval_metric="logloss",
                             random_state=SEED, n_jobs=2)
    return LGBMClassifier(**params, scale_pos_weight=spw, random_state=SEED,
                          n_jobs=2, verbose=-1)


def oof_score(name: str, params: dict, X, y, spw: float) -> tuple[float, float]:
    """Protocole commun : 5-fold out-of-fold, comparable à toutes les phases précédentes."""
    Xv, yv = X.to_numpy(dtype=np.float32), y.to_numpy()
    oof = np.zeros(len(yv))
    for tr, va in CV_OUTER.split(Xv, yv):
        m = build(name, params, spw)
        m.fit(Xv[tr], yv[tr])
        oof[va] = m.predict_proba(Xv[va])[:, 1]
    return average_precision_score(yv, oof), roc_auc_score(yv, oof)


def optimize(name: str, X, y, spw: float) -> dict:
    suggest = suggest_xgb if name == "xgboost" else suggest_lgbm

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        model = build(name, params, spw)
        return cross_val_score(model, X, y, cv=CV_INNER,
                               scoring="average_precision", n_jobs=1).mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - t0

    # meilleur score après 20 essais : comparaison à budget égal avec RandomizedSearchCV
    values = [t.value for t in study.trials if t.value is not None]
    best_at_20 = max(values[:20]) if len(values) >= 20 else max(values)

    print(f"\n  {name} — {len(study.trials)} essais en {elapsed:.0f}s")
    print(f"    RandomizedSearchCV (20 tirages)  : {BASELINE_RANDOM_SEARCH[name]:.4f}")
    print(f"    Optuna après 20 essais           : {best_at_20:.4f}")
    print(f"    Optuna après {N_TRIALS} essais          : {study.best_value:.4f}")
    print(f"    meilleurs paramètres : {study.best_params}")

    return {"study": study, "best_at_20": best_at_20, "elapsed": elapsed}


def main() -> None:
    tracking.setup()
    X, y, cols = dp.get_tabular()
    idx_train, _ = dp.train_test_indices(y)
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    n_neg, n_pos = np.bincount(y_train)
    spw = n_neg / n_pos

    print(f"Train {X_train.shape} | {N_TRIALS} essais Optuna par modèle "
          f"(TPE, 3-fold interne)\n")

    best_params, rows = {}, []
    for name in ("xgboost", "lightgbm"):
        out = optimize(name, X_train, y_train, spw)
        study = out["study"]
        best_params[name] = study.best_params

        ap, auc = oof_score(name, study.best_params, X_train, y_train, spw)
        print(f"    -> réévalué en 5-fold OOF : PR-AUC={ap:.4f}  ROC-AUC={auc:.4f}")

        rows.append({
            "model": name, "n_trials": len(study.trials),
            "random_search_3fold": BASELINE_RANDOM_SEARCH[name],
            "optuna_3fold_at_20": out["best_at_20"],
            "optuna_3fold_best": study.best_value,
            "oof_pr_auc": ap, "oof_roc_auc": auc, "seconds": out["elapsed"],
        })

        history = pd.DataFrame([
            {"trial": t.number, "value": t.value, **t.params}
            for t in study.trials if t.value is not None])
        history.to_csv(REPORTS / f"optuna_history_{name}.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "optuna_results.csv", index=False)
    (REPORTS / "optuna_best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8")

    tracking.set_experiment("phase3-optuna")
    for r in rows:
        with mlflow.start_run(run_name=f"{r['model']}_optuna"):
            mlflow.log_params({**{f"hp_{k}": v for k, v in best_params[r["model"]].items()},
                               "model": r["model"], "n_trials": r["n_trials"],
                               "sampler": "TPE", "pruner": "MedianPruner",
                               "search_space": "identique à RandomizedSearchCV"})
            mlflow.log_metrics({"pr_auc": r["oof_pr_auc"], "roc_auc": r["oof_roc_auc"],
                                "inner_3fold_best": r["optuna_3fold_best"],
                                "inner_3fold_at_20_trials": r["optuna_3fold_at_20"],
                                "random_search_reference": r["random_search_3fold"]})
            mlflow.set_tags({"phase": "3 - tuning Optuna",
                             "method": f"Optuna TPE, {r['n_trials']} essais",
                             "comparison": "vs RandomizedSearchCV 20 tirages"})

    print("\n" + "=" * 74)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {REPORTS / 'optuna_results.csv'}")


if __name__ == "__main__":
    main()
