"""Étape `prepare` du pipeline : data/raw/ -> data/processed/

Lit les données brutes, applique le nettoyage et le feature engineering, découpe en
train/test de façon stratifiée, puis écrit les deux jeux sur disque.

Pourquoi matérialiser plutôt que tout garder en mémoire : cela découple la préparation
de l'entraînement. Les étapes en aval lisent un fichier déjà prêt au lieu de reconstruire
les features à chaque exécution, et DVC ne relance cette étape que si les données brutes,
le code de préparation ou les paramètres du découpage ont réellement changé.

Le format est réglé par `data.processed_format` dans params.yaml. Parquet par défaut :
colonnaire, compressé et typé, il reste lisible par morceaux quand le volume grossit.

Usage :
    python scripts/prepare_data.py
    dvc repro prepare
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    fmt = params["data"].get("processed_format", "parquet")
    split_cfg = params["split"]

    print(f"Source  : {dp.DEFAULT_DATA_PATH.relative_to(ROOT)}")
    print(f"Format  : {fmt}")
    print(f"Découpe : test_size={split_cfg['test_size']}, "
          f"random_state={split_cfg['random_state']}, stratifié\n")

    info = dp.save_processed(fmt=fmt, test_size=split_cfg["test_size"],
                             random_state=split_cfg["random_state"])

    for split in ("train", "test"):
        d = info[split]
        print(f"  {split:<6} {d['rows']:>6} lignes x {d['cols']:>3} colonnes  "
              f"| défauts {d['default_rate']:.4f} | {d['size_mb']:.2f} Mo  "
              f"-> {d['path'].relative_to(ROOT)}")

    # Le taux de défaut doit être préservé par la stratification : c'est la garantie que
    # train et test restent comparables, et donc que les métriques le sont aussi.
    ecart = abs(info["train"]["default_rate"] - info["test"]["default_rate"])
    assert ecart < 0.005, f"stratification suspecte : écart de {ecart:.4f} entre train et test"
    print(f"\n  stratification vérifiée : écart de taux de défaut = {ecart:.5f}")

    summary = {
        "format": fmt,
        "n_features": len(info["features"]),
        "features": info["features"],
        "train": {k: v for k, v in info["train"].items() if k != "path"},
        "test": {k: v for k, v in info["test"].items() if k != "path"},
    }
    out = ROOT / "reports" / "data_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  résumé -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
