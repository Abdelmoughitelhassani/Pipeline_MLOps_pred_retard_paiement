"""Tests du service d'inférence.

Le service est le seul composant exposé à l'extérieur : ses garde-fous doivent être
vérifiés automatiquement, pas constatés à la main une fois en production.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "models" / "final_model.joblib"
CARD_PATH = ROOT / "models" / "model_card.json"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="modèle absent — lancer `dvc pull` ou `dvc repro train`")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import serve
    with TestClient(serve.app) as c:
        yield c


@pytest.fixture(scope="module")
def exemples(client):
    return client.get("/predict/example").json()


def test_health_expose_l_identite_du_modele(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["model_name"]
    assert d["n_features"] == 68
    assert 0 < d["threshold"] < 1


def test_seuils_coherents_entre_artefacts():
    """Le seuil vit dans le joblib et dans la carte du modèle. Une divergence ferait
    prendre des décisions de crédit avec le mauvais seuil, sans signe extérieur."""
    import joblib
    bundle = joblib.load(MODEL_PATH)
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    assert bundle["threshold"] == pytest.approx(card["threshold"]["threshold"])


def test_example_fournit_des_charges_utiles_valides(client, exemples):
    for nom in ("profil_a_risque", "profil_sain"):
        assert nom in exemples
        assert client.post("/predict", json=exemples[nom]).status_code == 200


def test_predict_retourne_une_probabilite_et_un_niveau(client, exemples):
    d = client.post("/predict", json=exemples["profil_sain"]).json()
    assert 0.0 <= d["default_probability"] <= 1.0
    assert d["risk_level"] in ("HIGH", "LOW")


def test_niveau_de_risque_coherent_avec_le_seuil(client, exemples):
    """HIGH si et seulement si la probabilité atteint le seuil — sinon le niveau
    retourné ne voudrait rien dire."""
    for nom in ("profil_a_risque", "profil_sain"):
        d = client.post("/predict", json=exemples[nom]).json()
        attendu = "HIGH" if d["default_probability"] >= d["threshold"] else "LOW"
        assert d["risk_level"] == attendu


def test_le_modele_discrimine(client, exemples):
    """Un profil dégradé doit scorer nettement plus haut qu'un profil régulier.
    Sans ce test, un modèle cassé renvoyant une constante passerait inaperçu."""
    risque = client.post("/predict", json=exemples["profil_a_risque"]).json()
    sain = client.post("/predict", json=exemples["profil_sain"]).json()
    assert risque["default_probability"] > sain["default_probability"] + 0.2


@pytest.mark.parametrize("champ,valeur", [
    ("LIMIT_BAL", -500),      # plafond négatif
    ("AGE", 5),               # âge hors bornes
    ("PAY_1", 99),            # statut de retard invalide
    ("SEX", 7),               # code inexistant
    ("PAY_AMT1", -100),       # remboursement négatif
])
def test_entrees_invalides_rejetees(client, exemples, champ, valeur):
    charge = dict(exemples["profil_sain"])
    charge[champ] = valeur
    assert client.post("/predict", json=charge).status_code == 422


def test_champs_manquants_rejetes(client):
    assert client.post("/predict", json={"LIMIT_BAL": 50000}).status_code == 422
