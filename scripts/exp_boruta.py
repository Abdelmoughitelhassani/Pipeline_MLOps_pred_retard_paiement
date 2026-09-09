"""Phase 3a — sélection de features par Boruta.

Boruta compare l'importance de chaque variable à celle de « variables fantômes » obtenues
en permutant aléatoirement les colonnes. Une variable n'est retenue que si elle bat
significativement le meilleur fantôme, sur de nombreuses itérations. C'est une sélection
*all-relevant* : elle garde tout ce qui porte de l'information, pas seulement le sous-ensemble
minimal suffisant.

Ce que ce script mesure, avec le protocole commun (5-fold, out-of-fold, PR-AUC) :

    1. les 68 features actuelles          (référence)
    2. les features confirmées par Boruta
    3. les confirmées + les indécises

Hypothèse posée avant l'exécution : l'ablation de la phase 2 ayant montré que 23 features
brutes valent 68 features enrichies, réduire le nombre de variables ne devrait PAS améliorer
la performance. L'intérêt attendu est la parcimonie et l'interprétabilité, pas le score.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402
import tracking  # noqa: E402

import mlflow  # noqa: E402

SEED = 42
MAX_ITER = 80
REPORTS = ROOT / "reports"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def oof_score(X: pd.DataFrame, y: pd.Series, params: dict, spw: float) -> tuple[float, float]:
    """PR-AUC et ROC-AUC out-of-fold — protocole identique aux phases précédentes."""
    Xv, yv = X.to_numpy(dtype=np.float32), y.to_numpy()
    oof = np.zeros(len(yv))
    for tr, va in CV.split(Xv, yv):
        m = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="logloss",
                          random_state=SEED, n_jobs=2)
        m.fit(Xv[tr], yv[tr])
        oof[va] = m.predict_proba(Xv[va])[:, 1]
    return average_precision_score(yv, oof), roc_auc_score(yv, oof)


def main() -> None:
    tracking.setup()
    X, y, cols = dp.get_tabular()
    idx_train, _ = dp.train_test_indices(y)
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    n_neg, n_pos = np.bincount(y_train)
    spw = n_neg / n_pos
    params = json.loads((REPORTS / "best_params.json").read_text(encoding="utf-8"))["xgboost"]

    print(f"Train {X_train.shape} | Boruta max_iter={MAX_ITER}\n")

    # ------------------------------------------------------------------ Boruta
    # max_depth=5 : recommandation Boruta, des arbres profonds diluent les importances.
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=2,
                                class_weight="balanced", random_state=SEED)
    selector = BorutaPy(rf, n_estimators=100, max_iter=MAX_ITER, random_state=SEED, verbose=0)
    t0 = time.time()
    selector.fit(X_train.to_numpy(), y_train.to_numpy())
    elapsed = time.time() - t0

    confirmed = [c for c, keep in zip(cols, selector.support_) if keep]
    tentative = [c for c, weak in zip(cols, selector.support_weak_) if weak]
    rejected = [c for c in cols if c not in confirmed and c not in tentative]

    print(f"Boruta terminé en {elapsed:.0f}s")
    print(f"  confirmées : {len(confirmed)}/{len(cols)}")
    print(f"  indécises  : {len(tentative)}")
    print(f"  rejetées   : {len(rejected)}")
    print(f"\n  rejetées -> {', '.join(rejected) if rejected else '(aucune)'}\n")

    ranking = pd.DataFrame({
        "feature": cols,
        "rank": selector.ranking_,
        "statut": ["confirmée" if c in confirmed else "indécise" if c in tentative else "rejetée"
                   for c in cols],
    }).sort_values("rank")
    ranking.to_csv(REPORTS / "boruta_ranking.csv", index=False)

    # ------------------------------------------------------------------ comparaison
    variants = {
        "Toutes les features (référence)": cols,
        f"Boruta confirmées ({len(confirmed)})": confirmed,
        f"Boruta confirmées + indécises ({len(confirmed)+len(tentative)})": confirmed + tentative,
    }

    rows = []
    print("Évaluation (5-fold out-of-fold, XGBoost tuné) :")
    for name, subset in variants.items():
        if not subset:
            print(f"  {name:<48} (sous-ensemble vide, ignoré)")
            continue
        ap, auc = oof_score(X_train[subset], y_train, params, spw)
        rows.append({"variante": name, "n_features": len(subset), "pr_auc": ap, "roc_auc": auc})
        print(f"  {name:<48} PR-AUC={ap:.4f}  ROC-AUC={auc:.4f}")

    res = pd.DataFrame(rows)
    res["ecart_vs_reference"] = (res["pr_auc"] - res.iloc[0]["pr_auc"]).round(4)
    res.to_csv(REPORTS / "boruta_results.csv", index=False)

    (REPORTS / "boruta_features.json").write_text(
        json.dumps({"confirmed": confirmed, "tentative": tentative, "rejected": rejected},
                   indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ MLflow
    tracking.set_experiment("phase3-feature-selection")
    for r in rows:
        with mlflow.start_run(run_name=r["variante"]):
            mlflow.log_params({"n_features": r["n_features"], "model": "XGBoost tuné",
                               "selection": "Boruta" if "Boruta" in r["variante"] else "aucune",
                               "boruta_max_iter": MAX_ITER, "cv": "StratifiedKFold(5) OOF"})
            mlflow.log_metrics({"pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"]})
            mlflow.set_tags({"phase": "3 - sélection de features",
                             "method": "Boruta (all-relevant, variables fantômes)"})

    print(f"\n-> {REPORTS / 'boruta_results.csv'}")
    print(f"-> {REPORTS / 'boruta_ranking.csv'}")
    print(f"-> {REPORTS / 'boruta_features.json'}")


if __name__ == "__main__":
    main()
