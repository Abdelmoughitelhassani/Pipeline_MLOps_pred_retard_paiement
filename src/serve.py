"""Service d'inférence FastAPI pour le scoring de défaut de paiement.

Trois points de terminaison :

    GET  /health           état du service et identité du modèle chargé
    POST /predict          probabilité de défaut et niveau de risque d'un client
    GET  /predict/example  un corps de requête valide, prêt à copier-coller

Choix de conception important : l'API attend les **23 variables brutes** du dossier client,
pas les 68 features dérivées. Le calcul des features est fait côté serveur via
`data_prep.prepare_inference()`, c'est-à-dire exactement le même code qu'à l'entraînement.
Demander à l'appelant de fournir les features dérivées l'obligerait à réimplémenter cette
logique, et le moindre écart fausserait les prédictions sans qu'aucune erreur ne soit levée
— c'est le *training/serving skew*, l'une des défaillances les plus coûteuses en production.

Lancer le service :
    uvicorn src.serve:app --reload --port 8000
    puis http://localhost:8000/docs pour la documentation interactive
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

MODEL_PATH = ROOT / "models" / "final_model.joblib"
CARD_PATH = ROOT / "models" / "model_card.json"


# --------------------------------------------------------------------------- chargement
class ModelBundle:
    """Modèle, features attendues et seuil de décision, chargés une fois au démarrage."""

    def __init__(self) -> None:
        self.model = None
        self.feature_cols: list[str] = []
        self.threshold: float = 0.5
        self.card: dict = {}
        self.loaded_at: str | None = None
        self.error: str | None = None

    def load(self) -> None:
        try:
            bundle = joblib.load(MODEL_PATH)
            self.model = bundle["model"]
            self.feature_cols = list(bundle["feature_cols"])
            self.card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
            card_threshold = float(self.card["threshold"]["threshold"])

            # Le seuil vit à deux endroits : la carte du modèle fait foi, mais un écart
            # avec celui sérialisé signale un artefact incohérent — mieux vaut le savoir
            # au démarrage qu'après avoir servi des décisions erronées.
            bundle_threshold = float(bundle.get("threshold", card_threshold))
            if abs(card_threshold - bundle_threshold) > 1e-9:
                raise ValueError(
                    f"seuils incohérents : {bundle_threshold} dans le modèle sérialisé "
                    f"contre {card_threshold} dans model_card.json")

            self.threshold = card_threshold
            self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.error = None
        except Exception as exc:                                  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self.model is not None and self.error is None


bundle = ModelBundle()


# --------------------------------------------------------------------------- schémas
class ClientFeatures(BaseModel):
    """Dossier client brut — les 23 variables du jeu de données UCI."""

    LIMIT_BAL: float = Field(..., gt=0, description="Plafond de crédit accordé (NT$)")
    SEX: int = Field(..., ge=1, le=2, description="1 = homme, 2 = femme")
    EDUCATION: int = Field(..., ge=0, le=6, description="1=universitaire … 4=autre (0/5/6 recodés en 4)")
    MARRIAGE: int = Field(..., ge=0, le=3, description="1=marié, 2=célibataire, 3=autre (0 recodé en 3)")
    AGE: int = Field(..., ge=18, le=120, description="Âge en années")

    PAY_1: int = Field(..., ge=-2, le=9, description="Statut de retard, mois le plus récent")
    PAY_2: int = Field(..., ge=-2, le=9)
    PAY_3: int = Field(..., ge=-2, le=9)
    PAY_4: int = Field(..., ge=-2, le=9)
    PAY_5: int = Field(..., ge=-2, le=9)
    PAY_6: int = Field(..., ge=-2, le=9, description="Statut de retard, mois le plus ancien")

    BILL_AMT1: float = Field(..., description="Facture du mois le plus récent (peut être négative)")
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(..., ge=0, description="Montant remboursé le mois le plus récent")
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Probabilité de défaut le mois suivant")
    risk_level: Literal["HIGH", "LOW"] = Field(..., description="HIGH si probabilité >= seuil")
    threshold: float = Field(..., description="Seuil de décision appliqué")
    model_name: str
    model_trained_at: str


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    model_name: str | None = None
    model_trained_at: str | None = None
    threshold: float | None = None
    n_features: int | None = None
    loaded_at: str | None = None
    detail: str | None = None


# --------------------------------------------------------------------------- application
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Charge le modèle une seule fois au démarrage, pas à chaque requête."""
    bundle.load()
    yield


app = FastAPI(
    title="Scoring de défaut de paiement",
    description="Estime la probabilité qu'un client de carte de crédit fasse défaut "
                "le mois suivant, à partir de son dossier et de 6 mois d'historique.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Supervision"])
def health() -> HealthResponse:
    """État du service et identité du modèle chargé."""
    if not bundle.ready:
        return HealthResponse(status="error", detail=bundle.error or "modèle non chargé")
    return HealthResponse(
        status="ok",
        model_name=bundle.card.get("model", "inconnu"),
        model_trained_at=bundle.card.get("trained_at"),
        threshold=bundle.threshold,
        n_features=len(bundle.feature_cols),
        loaded_at=bundle.loaded_at,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(client: ClientFeatures) -> PredictionResponse:
    """Probabilité de défaut et niveau de risque pour un client.

    Le niveau est HIGH dès que la probabilité atteint le seuil de décision. Ce seuil n'est
    pas 0.5 : il a été calibré sur un rapport de coût métier (coût d'un défaut manqué contre
    coût d'une fausse alerte), voir la section 17 du rapport.
    """
    if not bundle.ready:
        raise HTTPException(status_code=503,
                            detail=f"modèle indisponible — {bundle.error}")

    raw = pd.DataFrame([client.model_dump()])
    try:
        features = dp.prepare_inference(raw, bundle.feature_cols)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    proba = float(bundle.model.predict_proba(features)[0, 1])
    return PredictionResponse(
        default_probability=round(proba, 4),
        risk_level="HIGH" if proba >= bundle.threshold else "LOW",
        threshold=bundle.threshold,
        model_name=bundle.card.get("model", "inconnu"),
        model_trained_at=bundle.card.get("trained_at", "inconnu"),
    )


@app.get("/predict/example", tags=["Scoring"])
def predict_example() -> dict:
    """Un corps de requête valide, prêt à copier-coller vers POST /predict.

    Deux profils sont fournis : un client au comportement de paiement dégradé et un client
    régulier, pour vérifier d'un coup d'œil que le service discrimine bien les deux.
    """
    risque = {
        "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_1": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
        "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
        "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
        "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
    }
    sain = {
        "LIMIT_BAL": 500000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 45,
        "PAY_1": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 12000, "BILL_AMT2": 15000, "BILL_AMT3": 11000,
        "BILL_AMT4": 9000, "BILL_AMT5": 13000, "BILL_AMT6": 10000,
        "PAY_AMT1": 15000, "PAY_AMT2": 11000, "PAY_AMT3": 9000,
        "PAY_AMT4": 13000, "PAY_AMT5": 10000, "PAY_AMT6": 12000,
    }
    return {
        "description": "Copier l'un des deux objets ci-dessous dans le corps de POST /predict",
        "profil_a_risque": risque,
        "profil_sain": sain,
        "curl": "curl -X POST http://localhost:8000/predict "
                "-H 'Content-Type: application/json' "
                "-d @- <<< \"$(curl -s http://localhost:8000/predict/example "
                "| python -c 'import json,sys; print(json.dumps(json.load(sys.stdin)[\\\"profil_a_risque\\\"]))')\"",
    }
