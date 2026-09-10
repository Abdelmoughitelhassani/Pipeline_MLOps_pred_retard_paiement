"""Détection de dérive des données entre le jeu de référence et un nouveau lot.

Un modèle n'est valide que tant que les données qu'il reçoit ressemblent à celles sur
lesquelles il a été entraîné. Ce script compare la distribution de chaque variable d'un
lot entrant à celle du jeu d'entraînement et produit un rapport Evidently.

Conçu pour tourner en production sur n'importe quel lot : le mode simulation (1000
dernières lignes du test set) n'est que le comportement par défaut quand aucun lot n'est
fourni. En exploitation, on passe le chemin du lot réel.

    python monitoring/drift_report.py                              # simulation
    python monitoring/drift_report.py --current lot_du_jour.parquet
    python monitoring/drift_report.py --current lot.csv --output rapports/2026-09.html
    python monitoring/drift_report.py --current lot.parquet --fail-on-drift

Le code de sortie vaut 1 si une dérive est détectée et que `--fail-on-drift` est passé,
ce qui permet de brancher le script sur une tâche planifiée ou une CI qui alerte.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MONITORING_DIR = ROOT / "monitoring"
DEFAULT_REFERENCE = ROOT / "data" / "processed" / "train.parquet"
DEFAULT_OUTPUT = MONITORING_DIR / "drift_report.html"
TARGET = "DEFAULT"
SIMULATION_ROWS = 1000

# Colonnes catégorielles du jeu de données. Les déclarer explicitement évite qu'Evidently
# les traite comme numériques : un test de dérive sur des codes 1/2/3/4 n'a pas le même
# sens qu'un test sur un montant, et le choix du test statistique en dépend.
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def rel(path: Path) -> str:
    """Chemin affiché relativement à la racine du projet quand c'est possible.

    Un chemin fourni en ligne de commande peut pointer n'importe où — y compris hors du
    dépôt. `Path.relative_to()` lève dans ce cas, d'où ce repli sur le chemin complet.
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_table(path: Path) -> pd.DataFrame:
    """Lit un lot au format Parquet ou CSV, selon son extension."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable — lancer `dvc repro prepare` ou vérifier le chemin.")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"format non supporté : {path.suffix} (attendu .parquet ou .csv)")


def align_columns(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Restreint les deux jeux aux colonnes communes, cible exclue.

    En production, un lot entrant n'a pas encore de label : la cible ne peut donc pas être
    comparée. On la retire des deux côtés pour ne mesurer que la dérive des variables
    d'entrée, qui est ce qui compte pour savoir si le modèle opère hors de son domaine.
    """
    ref = reference.drop(columns=[TARGET], errors="ignore")
    cur = current.drop(columns=[TARGET], errors="ignore")

    common = [c for c in ref.columns if c in cur.columns]
    manquantes = [c for c in ref.columns if c not in cur.columns]
    inconnues = [c for c in cur.columns if c not in ref.columns]

    if manquantes:
        print(f"  ATTENTION — absentes du lot, ignorées : {manquantes}")
    if inconnues:
        print(f"  ATTENTION — inconnues de la référence, ignorées : {inconnues}")
    if not common:
        raise ValueError("aucune colonne commune entre la référence et le lot")

    return ref[common], cur[common], common


def build_definition(columns: list[str]) -> DataDefinition:
    categorical = [c for c in CATEGORICAL if c in columns]
    numerical = [c for c in columns if c not in categorical]
    return DataDefinition(numerical_columns=numerical, categorical_columns=categorical)


def extract_summary(snapshot, drift_share_threshold: float = 0.5) -> dict:
    """Extrait les indicateurs de dérive du rapport, pour l'alerte et l'historisation.

    Le HTML est fait pour être lu par un humain ; cette synthèse chiffrée est ce qu'on
    branche sur une supervision automatique.
    """
    payload = snapshot.dict()
    summary = {"drift_detected": None, "n_drifted_columns": None,
               "share_drifted_columns": None, "drifted_columns": []}

    for metric in payload.get("metrics", []):
        name = str(metric.get("metric_name", ""))
        config = metric.get("config", {}) or {}
        value = metric.get("value")

        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            summary["n_drifted_columns"] = int(value.get("count", 0))
            summary["share_drifted_columns"] = float(value.get("share", 0.0))

        elif name.startswith("ValueDrift") and isinstance(value, (int, float)):
            # Une métrique par colonne : le score est comparé au seuil porté par sa config,
            # car le test statistique — et donc le seuil — dépend du type de la variable.
            column = config.get("column")
            threshold = config.get("threshold")
            if column is None or threshold is None:
                continue
            if float(value) > float(threshold):
                summary["drifted_columns"].append({
                    "column": column,
                    "score": round(float(value), 4),
                    "threshold": float(threshold),
                    "method": config.get("method"),
                })

    summary["drifted_columns"].sort(key=lambda d: d["score"], reverse=True)

    if summary["share_drifted_columns"] is not None:
        summary["drift_share_threshold"] = drift_share_threshold
        summary["drift_detected"] = bool(
            summary["share_drifted_columns"] > drift_share_threshold)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                        help="jeu de référence (défaut : data/processed/train.parquet)")
    parser.add_argument("--current", type=Path, default=None,
                        help="lot à contrôler ; sans cet argument, simulation sur les "
                             f"{SIMULATION_ROWS} dernières lignes du test set")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="chemin du rapport HTML")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="code de sortie 1 si une dérive est détectée (pour CI/cron)")
    parser.add_argument("--drift-share-threshold", type=float, default=0.5,
                        help="part de colonnes dérivées au-delà de laquelle on considère que "
                             "le jeu a dérivé (défaut : 0.5, convention Evidently). Sur un jeu "
                             "à nombreuses variables dérivées, un seuil plus bas alerte plus tôt")
    args = parser.parse_args()

    # Résolus en absolu dès l'entrée : un chemin relatif fourni en ligne de commande
    # dépend du répertoire courant, ce qui rend le comportement imprévisible en cron.
    args.reference = args.reference.resolve()
    args.output = args.output.resolve()
    if args.current is not None:
        args.current = args.current.resolve()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    reference = read_table(args.reference)
    if args.current is None:
        test_path = ROOT / "data" / "processed" / "test.parquet"
        current = read_table(test_path).tail(SIMULATION_ROWS)
        source = f"SIMULATION — {SIMULATION_ROWS} dernières lignes de {test_path.name}"
    else:
        current = read_table(args.current)
        source = rel(args.current)

    print(f"Référence : {rel(args.reference)}  ({len(reference)} lignes)")
    print(f"Lot       : {source}  ({len(current)} lignes)\n")

    ref, cur, columns = align_columns(reference, current)
    definition = build_definition(columns)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        current_data=Dataset.from_pandas(cur, data_definition=definition),
        reference_data=Dataset.from_pandas(ref, data_definition=definition),
    )
    snapshot.save_html(str(args.output))

    summary = extract_summary(snapshot, args.drift_share_threshold)
    summary.update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": rel(args.reference),
        "reference_rows": len(reference),
        "current_source": source,
        "current_rows": len(current),
        "n_columns_compared": len(columns),
    })
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"  colonnes comparées   : {len(columns)}")
    if summary["n_drifted_columns"] is not None:
        print(f"  colonnes dérivées    : {summary['n_drifted_columns']} "
              f"({summary['share_drifted_columns']:.1%})")
    print(f"  dérive détectée      : {summary['drift_detected']}")
    if summary["drifted_columns"]:
        print("\n  variables ayant dérivé (score > seuil) :")
        for d in summary["drifted_columns"][:10]:
            print(f"    {d['column']:<24} score={d['score']:.4f}  seuil={d['threshold']}")
        reste = len(summary["drifted_columns"]) - 10
        if reste > 0:
            print(f"    ... et {reste} autres")
    print(f"\n  rapport  -> {rel(args.output)}")
    print(f"  synthèse -> {rel(summary_path)}")

    if args.fail_on_drift and summary["drift_detected"]:
        print("\n  DÉRIVE DÉTECTÉE — code de sortie 1")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
