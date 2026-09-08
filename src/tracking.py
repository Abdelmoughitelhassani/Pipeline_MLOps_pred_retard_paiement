"""Configuration centralisée du suivi d'expériences MLflow.

Backend SQLite (`mlflow.db`) plutôt que le store fichier : ce dernier est en mode
maintenance depuis MLflow 3.x et ne supporte pas le registre de modèles. Un fichier
unique est aussi plus simple à versionner avec DVC qu'une arborescence de milliers
de petits fichiers.

Tout script qui enregistre une expérience doit passer par `setup()` pour que tous
les runs atterrissent au même endroit.
"""
from __future__ import annotations

from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"

TRACKING_URI = f"sqlite:///{TRACKING_DB.as_posix()}"


def setup() -> str:
    """Pointe MLflow vers la base du projet. Retourne l'URI de tracking."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    return TRACKING_URI


def set_experiment(name: str) -> str:
    """Sélectionne (ou crée) une expérience en fixant son dossier d'artefacts."""
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            name, artifact_location=(ARTIFACTS_DIR / name).as_uri())
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id
