"""Tests de la préparation des données.

Ces tests ne sont pas décoratifs : chacun verrouille un problème réellement rencontré
pendant le projet. Les commentaires indiquent lequel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

pytestmark = pytest.mark.skipif(
    not dp.DEFAULT_DATA_PATH.exists(),
    reason="données brutes absentes — lancer `dvc pull`")


@pytest.fixture(scope="module")
def tabular():
    return dp.get_tabular()


def test_remap_fusionne_les_codes_non_documentes():
    """EDUCATION 0/5/6 et MARRIAGE 0 doivent rejoindre la catégorie « autre »."""
    brut = pd.DataFrame({"EDUCATION": [0, 1, 2, 3, 4, 5, 6],
                         "MARRIAGE": [0, 1, 2, 3, 1, 2, 3]})
    out = dp.apply_category_remap(brut)
    assert set(out["EDUCATION"]) <= {1, 2, 3, 4}
    assert set(out["MARRIAGE"]) <= {1, 2, 3}
    assert (out["EDUCATION"].iloc[[0, 5, 6]] == 4).all()
    assert out["MARRIAGE"].iloc[0] == 3


def test_aucune_valeur_non_finie(tabular):
    """Régression : `pay_ratio` divisait par une facture pouvant valoir zéro et
    produisait des infinis, ce qui faisait planter StandardScaler en aval."""
    X, _, _ = tabular
    assert np.isfinite(X.to_numpy()).all()


def test_dimensions_et_cible(tabular):
    X, y, cols = tabular
    assert X.shape == (30000, 68)
    assert len(cols) == 68
    assert set(y.unique()) == {0, 1}
    assert 0.21 < y.mean() < 0.23          # ~22.12 % de défauts


def test_split_disjoint_et_stratifie(tabular):
    """Un chevauchement train/test invaliderait toutes les métriques du projet."""
    _, y, _ = tabular
    idx_train, idx_test = dp.train_test_indices(y)
    assert not set(idx_train) & set(idx_test)
    assert len(set(idx_train) | set(idx_test)) == len(y)
    ecart = abs(y.iloc[idx_train].mean() - y.iloc[idx_test].mean())
    assert ecart < 0.005


def test_inference_identique_a_entrainement(tabular):
    """Invariant central : le chemin d'inférence doit produire exactement les mêmes
    features que l'entraînement. Tout écart serait un training/serving skew, c'est-à-dire
    des prédictions fausses sans la moindre erreur levée."""
    X, _, cols = tabular
    brut = dp.load_clean().head(50)[dp.RAW_FEATURES]
    depuis_inference = dp.prepare_inference(brut, cols)

    assert list(depuis_inference.columns) == cols
    np.testing.assert_allclose(depuis_inference.to_numpy(), X.head(50).to_numpy())


def test_inference_refuse_les_colonnes_manquantes():
    incomplet = pd.DataFrame({"LIMIT_BAL": [50000], "AGE": [30]})
    with pytest.raises(ValueError, match="colonnes manquantes"):
        dp.prepare_inference(incomplet)


def test_sequences_en_ordre_chronologique():
    """Le fichier numérote les mois à l'envers (indice 1 = le plus récent). Les donner
    tels quels à un RNN lui ferait lire l'histoire à rebours."""
    X_seq, X_static, y = dp.get_sequences()
    assert X_seq.shape == (30000, 6, 5)
    assert X_static.shape == (30000, 11)
    assert len(y) == 30000

    brut = dp.load_clean()
    # Le dernier pas de temps doit correspondre au mois le plus récent (PAY_1)
    np.testing.assert_allclose(X_seq[:100, -1, 0], brut["PAY_1"].head(100).to_numpy())
    # Le premier pas de temps au mois le plus ancien (PAY_6)
    np.testing.assert_allclose(X_seq[:100, 0, 0], brut["PAY_6"].head(100).to_numpy())
