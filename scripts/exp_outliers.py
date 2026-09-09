"""Phase 4 — détection et suppression d'outliers : est-ce que ça aide ?

Aucune ligne n'avait été supprimée jusqu'ici. Ce script teste si retirer les observations
atypiques améliore le modèle, avec trois détecteurs (Isolation Forest, Local Outlier Factor,
DBSCAN) et surtout DEUX protocoles, dont un volontairement incorrect.

Le point central est méthodologique. Supprimer des outliers ne doit se faire que sur les
données d'ENTRAÎNEMENT : si on les retire aussi du jeu d'évaluation, on supprime les cas
difficiles sur lesquels le modèle est jugé, et le score monte artificiellement. C'est une
erreur fréquente ; ce script la reproduit délibérément pour mesurer l'ampleur du mirage.

Contexte mesuré au préalable : les lignes flaguées par Isolation Forest contiennent
1.3 à 1.5 fois plus de défauts que les autres. Un détecteur non supervisé ignore la cible,
il supprime donc préférentiellement le signal que le modèle doit apprendre.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402
import tracking  # noqa: E402

import mlflow  # noqa: E402

SEED = 42
REPORTS = ROOT / "reports"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def detect(method: str, X: np.ndarray, contamination: float) -> np.ndarray:
    """Retourne un masque booléen : True = observation considérée comme atypique."""
    if method == "isolation_forest":
        det = IsolationForest(contamination=contamination, n_estimators=200,
                              random_state=SEED, n_jobs=2)
        return det.fit_predict(X) == -1
    if method == "lof":
        det = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=2)
        return det.fit_predict(X) == -1
    if method == "dbscan":
        # DBSCAN sur données standardisées. En grande dimension les distances se concentrent,
        # ce qui rend le paramètre eps difficile à régler et la notion de densité peu fiable.
        Xs = StandardScaler().fit_transform(X)
        labels = DBSCAN(eps=8.0, min_samples=10, n_jobs=2).fit_predict(Xs)
        return labels == -1
    raise ValueError(method)


def evaluate(params: dict, spw: float, X: pd.DataFrame, y: pd.Series,
             method: str | None, contamination: float, clean_eval: bool) -> dict:
    """Évalue en 5-fold.

    `clean_eval=False` : protocole CORRECT — suppression sur le train du pli uniquement.
    `clean_eval=True`  : protocole INCORRECT — suppression aussi sur le pli d'évaluation.
    """
    Xv, yv = X.to_numpy(dtype=np.float32), y.to_numpy()
    scores_ap, scores_auc, removed = [], [], 0

    for tr, va in CV.split(Xv, yv):
        X_tr, y_train = Xv[tr], yv[tr]
        X_va, y_va = Xv[va], yv[va]

        if method is not None:
            mask_tr = detect(method, X_tr, contamination)
            X_tr, y_train = X_tr[~mask_tr], y_train[~mask_tr]
            removed += int(mask_tr.sum())
            if clean_eval:
                mask_va = detect(method, X_va, contamination)
                X_va, y_va = X_va[~mask_va], y_va[~mask_va]

        model = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="logloss",
                              random_state=SEED, n_jobs=2)
        model.fit(X_tr, y_train)
        proba = model.predict_proba(X_va)[:, 1]
        scores_ap.append(average_precision_score(y_va, proba))
        scores_auc.append(roc_auc_score(y_va, proba))

    return {"pr_auc": float(np.mean(scores_ap)), "pr_auc_std": float(np.std(scores_ap)),
            "roc_auc": float(np.mean(scores_auc)), "lignes_retirees": removed // 5}


def main() -> None:
    tracking.setup()
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    cfg = params["final_model"]["params"]

    X, y, cols = dp.get_tabular()
    idx_train, _ = dp.train_test_indices(y)
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    n_neg, n_pos = np.bincount(y_train)
    spw = n_neg / n_pos

    print(f"Train {X_train.shape} | modèle : XGBoost tuné (Optuna)\n")

    rows = []
    base = evaluate(cfg, spw, X_train, y_train, None, 0.0, False)
    rows.append({"detecteur": "aucun (référence)", "contamination": 0.0,
                 "protocole": "—", **base})
    print(f"  {'aucun (référence)':<34} PR-AUC={base['pr_auc']:.4f} "
          f"±{base['pr_auc_std']:.4f}", flush=True)

    print("\n--- Protocole CORRECT : suppression sur le train du pli uniquement ---")
    for method in ("isolation_forest", "lof", "dbscan"):
        for cont in (0.01, 0.05, 0.10):
            if method == "dbscan" and cont != 0.05:
                continue  # DBSCAN ne prend pas de taux de contamination
            t0 = time.time()
            r = evaluate(cfg, spw, X_train, y_train, method, cont, clean_eval=False)
            label = f"{method} ({cont:.0%})" if method != "dbscan" else "dbscan (eps=8)"
            rows.append({"detecteur": method, "contamination": cont,
                         "protocole": "correct (train seul)", **r})
            print(f"  {label:<34} PR-AUC={r['pr_auc']:.4f} ±{r['pr_auc_std']:.4f}  "
                  f"({r['lignes_retirees']} lignes retirées/pli, {time.time()-t0:.0f}s)",
                  flush=True)

    print("\n--- Protocole INCORRECT : suppression aussi sur le pli d'évaluation ---")
    for cont in (0.05, 0.10):
        r = evaluate(cfg, spw, X_train, y_train, "isolation_forest", cont, clean_eval=True)
        rows.append({"detecteur": "isolation_forest", "contamination": cont,
                     "protocole": "INCORRECT (train + éval)", **r})
        print(f"  {'isolation_forest ' + f'({cont:.0%})':<34} PR-AUC={r['pr_auc']:.4f} "
              f"±{r['pr_auc_std']:.4f}   <- mirage", flush=True)

    res = pd.DataFrame(rows)
    res["ecart_vs_reference"] = (res["pr_auc"] - base["pr_auc"]).round(4)
    res.to_csv(REPORTS / "outliers_results.csv", index=False)

    # taux de défaut parmi les observations flaguées : la raison du problème
    enrich = []
    for cont in (0.01, 0.05, 0.10):
        mask = detect("isolation_forest", X_train.to_numpy(dtype=np.float32), cont)
        enrich.append({"contamination": cont, "n_flagues": int(mask.sum()),
                       "taux_defaut_outliers": float(y_train[mask].mean()),
                       "taux_defaut_conserves": float(y_train[~mask].mean())})
    pd.DataFrame(enrich).to_csv(REPORTS / "outliers_enrichment.csv", index=False)

    tracking.set_experiment("phase4-outliers")
    for r in rows:
        with mlflow.start_run(run_name=f"{r['detecteur']}_{r['contamination']:.2f}_"
                                       f"{'ok' if 'correct' in r['protocole'] else 'ref_ou_faux'}"):
            mlflow.log_params({"detecteur": r["detecteur"], "contamination": r["contamination"],
                               "protocole": r["protocole"],
                               "lignes_retirees_par_pli": r["lignes_retirees"],
                               "modele": "XGBoost tuné Optuna"})
            mlflow.log_metrics({"pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
                                "pr_auc_std": r["pr_auc_std"]})
            mlflow.set_tags({"phase": "4 - traitement des outliers",
                             "avertissement": "les protocoles marqués INCORRECT retirent des "
                                              "lignes du jeu d'évaluation : score non valide"})

    print("\n" + "=" * 78)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {REPORTS / 'outliers_results.csv'}")


if __name__ == "__main__":
    main()
