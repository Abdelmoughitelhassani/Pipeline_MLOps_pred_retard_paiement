"""Évaluation finale sur le test set + choix du seuil par coût métier.

Le test set (6 000 clients) n'a servi à aucune décision jusqu'ici : ni sélection de features,
ni choix de modèle, ni tuning, ni calibration de seuil. Il n'est utilisé qu'ici, une seule fois.

Sur le seuil : maximiser aveuglément le F2 donnait un seuil de 0.258 qui flaguait 60% du
portefeuille — inexploitable en pratique. On raisonne donc en coût métier explicite :

    coût total = C_FN x (défauts manqués) + C_FP x (fausses alertes)

et on balaye le rapport R = C_FN / C_FP, puisque seul le rapport compte pour le choix du seuil.
La banque n'a pas besoin de connaître ses coûts en valeur absolue, seulement leur rapport.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

from xgboost import XGBClassifier  # noqa: E402

SEED = 42
PARAMS_FILE = ROOT / "reports" / "best_params.json"
OUT_JSON = ROOT / "reports" / "final_evaluation.json"


def cost_threshold_table(y_true, proba, ratios=(2, 3, 5, 10, 20)) -> pd.DataFrame:
    """Pour chaque rapport de coût R = C_FN/C_FP, le seuil qui minimise le coût total."""
    grid = np.linspace(0.05, 0.95, 181)
    rows = []
    n_pos = y_true.sum()
    for r in ratios:
        best = None
        for t in grid:
            pred = (proba >= t).astype(int)
            fn = int(((pred == 0) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            cost = r * fn + fp
            if best is None or cost < best["cost"]:
                tp = int(((pred == 1) & (y_true == 1)).sum())
                best = {"ratio_FN_FP": r, "seuil": round(float(t), 3), "cost": cost,
                        "recall": tp / n_pos, "precision": tp / max(tp + fp, 1),
                        "clients_flagues": int(pred.sum()),
                        "pct_portefeuille": pred.mean()}
        rows.append(best)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    X_train, y_train, X_test, y_test, cols = dp.get_splits(
        fmt=cfg["data"].get("processed_format", "parquet"))
    params = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))["xgboost"]
    n_neg, n_pos = np.bincount(y_train)
    spw = n_neg / n_pos

    model = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="logloss",
                          random_state=SEED, n_jobs=2)

    # --- seuils choisis sur des probabilités out-of-fold du TRAIN (aucune fuite) -----------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    cost_table = cost_threshold_table(y_train.to_numpy(), oof)

    print("Seuils optimaux selon le rapport de coût (calibrés sur OOF du train) :")
    print(cost_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- évaluation unique sur le test -----------------------------------------------------
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba_test)
    roc = roc_auc_score(y_test, proba_test)

    print(f"\n=== TEST SET (n={len(y_test)}, jamais utilisé avant) ===")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"ROC-AUC : {roc:.4f}")

    results = {"model": "XGBoost tuné", "params": params,
               "test_pr_auc": float(pr_auc), "test_roc_auc": float(roc), "thresholds": {}}

    reference = {"0.5 (défaut)": 0.5}
    for _, row in cost_table.iterrows():
        reference[f"coût R={int(row['ratio_FN_FP'])}"] = row["seuil"]

    for label, thr in reference.items():
        pred = (proba_test >= thr).astype(int)
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\n--- Seuil {thr:.3f} ({label}) ---")
        print(classification_report(y_test, pred, digits=3,
                                    target_names=["Pas de défaut", "Défaut"]))
        print(f"Matrice de confusion : TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"Clients flagués : {pred.sum()} / {len(pred)} ({pred.mean():.1%} du portefeuille)")
        results["thresholds"][label] = {
            "seuil": float(thr), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "recall": float(tp / (tp + fn)), "precision": float(tp / max(tp + fp, 1)),
            "pct_flagues": float(pred.mean())}

    cost_table.to_csv(ROOT / "reports" / "cost_thresholds.csv", index=False)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Contrat stable pour la porte de qualité de la CI. Volontairement séparé du rapport
    # complet : un fichier minimal et au format figé ne casse pas le pipeline quand on
    # enrichit `final_evaluation.json`.
    scores = {
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc), 4),
        "model": "xgboost",
        "n_test": int(len(y_test)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    scores_path = ROOT / "metrics" / "scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print(f"\n-> {OUT_JSON}")
    print(f"-> {scores_path}")


if __name__ == "__main__":
    main()
