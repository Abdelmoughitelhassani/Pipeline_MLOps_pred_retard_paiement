"""Génère un lot de test simulant une récession, pour éprouver la détection de dérive.

Un détecteur qui ne se déclenche jamais est inutile. Ce script fabrique un lot dont on
sait qu'il a dérivé, afin de vérifier que `drift_report.py` le signale — et de calibrer
le seuil d'alerte sur un scénario réaliste plutôt qu'au jugé.

Scénario retenu — dégradation macroéconomique :

    plafonds de crédit      -55 %    les banques resserrent l'octroi
    statuts de retard       +1 à +2  les clients paient avec du retard
    montants remboursés     -70 %    la capacité de remboursement s'effondre
    âge                     -9 ans   la clientèle se déplace vers des profils plus jeunes

Les variables dérivées (taux d'utilisation, ratios de remboursement, agrégats de retard)
ne sont pas modifiées à la main : elles sont recalculées à partir des variables brutes,
exactement comme en production. Un lot forgé colonne par colonne serait incohérent.

Usage :
    python monitoring/make_recession_batch.py
    python monitoring/make_recession_batch.py --rows 2000 --severity 0.8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

SOURCE = ROOT / "data" / "processed" / "test.parquet"
DEFAULT_OUTPUT = ROOT / "monitoring" / "batch_recession.parquet"
SEED = 7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=1000, help="taille du lot")
    parser.add_argument("--severity", type=float, default=1.0,
                        help="intensité de la récession (1.0 = scénario nominal, "
                             "0.5 = moitié moins marqué)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not SOURCE.exists():
        raise FileNotFoundError(f"{SOURCE} introuvable — lancer `dvc repro prepare`.")

    rng = np.random.default_rng(SEED)
    sev = args.severity

    # On repart des variables BRUTES pour que le feature engineering soit rejoué ensuite.
    base = pd.read_parquet(SOURCE).tail(args.rows).reset_index(drop=True)
    raw = base[dp.RAW_FEATURES].copy()
    target = base[dp.TARGET].copy() if dp.TARGET in base.columns else None

    raw["LIMIT_BAL"] = (raw["LIMIT_BAL"] * (1 - 0.55 * sev)).round()
    raw["AGE"] = (raw["AGE"] - int(round(9 * sev))).clip(lower=18)
    for col in dp.PAY_COLS:
        aggravation = rng.integers(1, 3, len(raw)) * sev
        raw[col] = (raw[col] + aggravation).round().clip(-2, 9).astype(int)
    for col in dp.PAY_AMT_COLS:
        raw[col] = (raw[col] * (1 - 0.70 * sev)).round()

    # Les variables dérivées sont RECALCULÉES, pas bricolées : c'est la même chaîne
    # qu'à l'entraînement et qu'à l'inférence.
    features = dp.prepare_inference(raw)
    if target is not None:
        features[dp.TARGET] = target.to_numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)

    print(f"Source   : {SOURCE.relative_to(ROOT)} ({args.rows} dernières lignes)")
    print(f"Sévérité : {sev}")
    print(f"Sortie   : {args.output.relative_to(ROOT)} "
          f"({len(features)} lignes x {features.shape[1]} colonnes, "
          f"{args.output.stat().st_size / 1e3:.0f} Ko)\n")

    print("Effet sur quelques variables (moyennes) :")
    ref = dp.prepare_inference(base[dp.RAW_FEATURES])
    for col in ("LIMIT_BAL", "AGE", "PAY_1", "PAY_AMT1", "utilization_rate", "n_months_late"):
        avant, apres = ref[col].mean(), features[col].mean()
        variation = (apres - avant) / abs(avant) * 100 if avant else float("nan")
        print(f"  {col:<20} {avant:>12.2f}  ->  {apres:>12.2f}   ({variation:+.1f} %)")


if __name__ == "__main__":
    main()
