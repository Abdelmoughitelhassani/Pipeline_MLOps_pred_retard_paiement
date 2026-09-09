"""Configuration centralisée du suivi d'expériences MLflow.

Backend SQLite (`mlflow.db`) plutôt que le store fichier : ce dernier est en mode
maintenance depuis MLflow 3.x et ne supporte pas le registre de modèles. Un fichier
unique est aussi plus simple à versionner avec DVC qu'une arborescence de milliers
de petits fichiers.

Tout script qui enregistre une expérience doit passer par `setup()` pour que tous
les runs atterrissent au même endroit.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"

LOCAL_TRACKING_URI = f"sqlite:///{TRACKING_DB.as_posix()}"

# Capturé à l'import, avant tout appel à MLflow : `mlflow.set_tracking_uri()` écrit
# lui-même dans MLFLOW_TRACKING_URI, donc lire la variable après coup ferait croire
# à tort qu'un serveur distant est configuré.
_CONFIGURED_URI = os.environ.get("MLFLOW_TRACKING_URI") or None


def is_remote() -> bool:
    """Vrai si un serveur MLflow distant était configuré au lancement du script."""
    return _CONFIGURED_URI is not None and not _CONFIGURED_URI.startswith("sqlite:")


def setup() -> str:
    """Pointe MLflow vers la base du projet, ou vers un serveur distant si configuré.

    Définir `MLFLOW_TRACKING_URI` (par exemple vers un serveur DagsHub) redirige tous les
    runs vers ce serveur, avec `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`
    pour l'authentification. Sans cette variable, tout reste en local dans `mlflow.db`.
    Les identifiants ne doivent jamais être écrits dans le code : ils passent uniquement
    par l'environnement, qui n'est pas versionné.
    """
    uri = _CONFIGURED_URI or LOCAL_TRACKING_URI
    if not is_remote():
        ARTIFACTS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return uri


def set_experiment(name: str) -> str:
    """Sélectionne (ou crée) une expérience.

    En local, les artefacts sont rangés par expérience dans `mlartifacts/`. Sur un serveur
    distant, c'est le serveur qui décide de l'emplacement : on ne l'impose pas.
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        exp_id = exp.experiment_id
    elif is_remote():
        exp_id = mlflow.create_experiment(name)
    else:
        exp_id = mlflow.create_experiment(
            name, artifact_location=(ARTIFACTS_DIR / name).as_uri())
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id
