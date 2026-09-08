"""Expérience 1 — modèles tabulaires : familles de boosting, tuning, ensembles.

Protocole identique pour tous les modèles (comparaison apples-to-apples) :
  - mêmes données (features de src/data_prep.py), même split stratifié (seed 42)
  - même 5-fold StratifiedKFold sur le train uniquement
  - métriques : PR-AUC (average_precision) et ROC-AUC, moyenne +/- écart-type des folds
  - le test set n'est jamais touché ici

Les modèles « tuned » sont d'abord optimisés par RandomizedSearchCV (3-fold), puis réévalués
avec le protocole commun 5-fold pour rester comparables aux autres.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

from lightgbm import LGBMClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = ["average_precision", "roc_auc"]
OUT_CSV = ROOT / "reports" / "exp_tabular_results.csv"
OUT_PARAMS = ROOT / "reports" / "best_params.json"


def evaluate(name: str, model, X, y, results: list) -> None:
    t0 = time.time()
    res = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=1)
    ap, auc = res["test_average_precision"], res["test_roc_auc"]
    row = {
        "model": name,
        "pr_auc": ap.mean(), "pr_auc_std": ap.std(),
        "roc_auc": auc.mean(), "roc_auc_std": auc.std(),
        "fit_time_s": time.time() - t0,
    }
    results.append(row)
    print(f"  {name:<34} PR-AUC={ap.mean():.4f} +/-{ap.std():.4f}   "
          f"ROC-AUC={auc.mean():.4f} +/-{auc.std():.4f}   ({row['fit_time_s']:.0f}s)", flush=True)


def main() -> None:
    X, y, cols = dp.get_tabular()
    idx_train, idx_test = dp.train_test_indices(y)
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]

    n_neg, n_pos = np.bincount(y_train)
    spw = n_neg / n_pos
    print(f"Train : {X_train.shape} | scale_pos_weight = {spw:.3f}\n")

    results: list[dict] = []

    # ---------------------------------------------------------------- modèles de référence
    print("[1/4] Modèles de référence")
    evaluate("LogisticRegression", Pipeline([
        ("sc", StandardScaler()),
        ("m", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ]), X_train, y_train, results)

    evaluate("RandomForest", RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
        X_train, y_train, results)

    evaluate("ExtraTrees", ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
        X_train, y_train, results)

    evaluate("HistGradientBoosting", HistGradientBoostingClassifier(
        max_iter=300, class_weight="balanced", random_state=SEED),
        X_train, y_train, results)

    # ---------------------------------------------------------------- boosting « par défaut »
    print("\n[2/4] Boosting (paramètres par défaut du notebook)")
    evaluate("XGBoost (baseline notebook)", XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1, scale_pos_weight=spw,
        eval_metric="logloss", random_state=SEED, n_jobs=-1),
        X_train, y_train, results)

    evaluate("LightGBM (défaut)", LGBMClassifier(
        n_estimators=300, scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbose=-1),
        X_train, y_train, results)

    evaluate("CatBoost (défaut)", CatBoostClassifier(
        iterations=500, scale_pos_weight=spw, random_seed=SEED, verbose=0, thread_count=-1),
        X_train, y_train, results)

    # ---------------------------------------------------------------- tuning
    print("\n[3/4] Tuning d'hyperparamètres (RandomizedSearchCV, 3-fold, 30 tirages)")
    search_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_params: dict[str, dict] = {}

    xgb_space = {
        "max_depth": randint(2, 9),
        "learning_rate": loguniform(0.01, 0.3),
        "n_estimators": randint(200, 1200),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.4, 0.6),
        "min_child_weight": randint(1, 20),
        "reg_lambda": loguniform(0.1, 20),
        "reg_alpha": loguniform(1e-3, 5),
        "gamma": uniform(0, 5),
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=SEED, n_jobs=-1),
        xgb_space, n_iter=30, scoring="average_precision", cv=search_cv,
        random_state=SEED, n_jobs=1, verbose=0)
    t0 = time.time()
    xgb_search.fit(X_train, y_train)
    best_params["xgboost"] = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                              for k, v in xgb_search.best_params_.items()}
    print(f"  XGBoost   -> best CV(3f) AP={xgb_search.best_score_:.4f} ({time.time()-t0:.0f}s)")
    print(f"             {best_params['xgboost']}")

    lgbm_space = {
        "num_leaves": randint(15, 150),
        "max_depth": randint(3, 12),
        "learning_rate": loguniform(0.01, 0.3),
        "n_estimators": randint(200, 1200),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.4, 0.6),
        "min_child_samples": randint(10, 120),
        "reg_lambda": loguniform(1e-3, 20),
        "reg_alpha": loguniform(1e-3, 5),
    }
    lgbm_search = RandomizedSearchCV(
        LGBMClassifier(scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbose=-1),
        lgbm_space, n_iter=30, scoring="average_precision", cv=search_cv,
        random_state=SEED, n_jobs=1, verbose=0)
    t0 = time.time()
    lgbm_search.fit(X_train, y_train)
    best_params["lightgbm"] = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                               for k, v in lgbm_search.best_params_.items()}
    print(f"  LightGBM  -> best CV(3f) AP={lgbm_search.best_score_:.4f} ({time.time()-t0:.0f}s)")
    print(f"             {best_params['lightgbm']}")

    print("\n  Réévaluation des modèles tunés avec le protocole commun (5-fold) :")
    xgb_tuned = XGBClassifier(**best_params["xgboost"], scale_pos_weight=spw,
                              eval_metric="logloss", random_state=SEED, n_jobs=-1)
    evaluate("XGBoost (tuné)", xgb_tuned, X_train, y_train, results)

    lgbm_tuned = LGBMClassifier(**best_params["lightgbm"], scale_pos_weight=spw,
                                random_state=SEED, n_jobs=-1, verbose=-1)
    evaluate("LightGBM (tuné)", lgbm_tuned, X_train, y_train, results)

    # ---------------------------------------------------------------- ensembles
    print("\n[4/4] Ensembles")
    cat = CatBoostClassifier(iterations=500, scale_pos_weight=spw, random_seed=SEED,
                             verbose=0, thread_count=-1)
    base = [("xgb", xgb_tuned), ("lgbm", lgbm_tuned), ("cat", cat)]

    evaluate("Voting (soft: XGB+LGBM+Cat)",
             VotingClassifier(base, voting="soft", n_jobs=1), X_train, y_train, results)

    evaluate("Stacking (méta = LogReg)", StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        cv=3, n_jobs=1), X_train, y_train, results)

    # ---------------------------------------------------------------- sortie
    df_res = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUT_CSV, index=False)
    OUT_PARAMS.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("CLASSEMENT (PR-AUC, 5-fold CV sur le train)")
    print("=" * 88)
    print(df_res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nRésultats -> {OUT_CSV}")
    print(f"Hyperparamètres -> {OUT_PARAMS}")


if __name__ == "__main__":
    main()
