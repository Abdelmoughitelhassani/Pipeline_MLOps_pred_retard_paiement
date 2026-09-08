"""Préparation des données — module réutilisable (notebooks, scripts d'expérimentation, production).

Deux vues des mêmes données sont exposées :

- `get_tabular()`   : matrice plate (n, k) pour les modèles tabulaires (XGBoost, LightGBM, ...)
- `get_sequences()` : tenseur (n, 6, f) + statiques (n, s) pour les modèles séquentiels (LSTM/GRU)

La vue séquentielle exploite la structure réelle du dataset : chaque client est décrit par
6 mois consécutifs de (statut de retard, facture, paiement). Les colonnes du fichier sont
numérotées à l'envers (indice 1 = mois le plus récent), donc on les remet en ordre
chronologique (t=0 = mois le plus ancien) avant de les donner à un RNN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Chemins relatifs à la racine du projet : le dépôt doit être clonable et exécutable
# sur n'importe quelle machine, sans chemin absolu codé en dur.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "default_of_credit_card_clients.xls"

BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BASE_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
RAW_FEATURES = BASE_COLS + PAY_COLS + BILL_COLS + PAY_AMT_COLS
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------------------------------- chargement
def load_clean(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Charge le fichier UCI, normalise les noms de colonnes et fusionne les codes non documentés.

    `EDUCATION` 0/5/6 et `MARRIAGE` 0 ne figurent pas dans la documentation UCI : ils sont
    fusionnés dans la catégorie « autre » déjà prévue par le schéma (4 et 3 respectivement),
    ce qui évite à la fois de perdre des lignes et d'inventer une imputation non vérifiable.
    """
    df = pd.read_excel(path, sheet_name=0, header=1)
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "DEFAULT"})
    df.columns = df.columns.str.strip()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


# --------------------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features dérivées (ratios métier, agrégats, dynamique temporelle).

    Note : `pay_ratio_*` est calculé conditionnellement (`BILL > 0`) et borné à [0, 2].
    Une première version divisait par `BILL.clip(lower=0) + 1`, ce qui transformait le ratio
    en montant brut dès que la facture approchait 0 et détruisait la feature
    (corrélation avec la cible ~0.005 au lieu de ~0.10).
    """
    df = df.copy()
    limit = df["LIMIT_BAL"] + 1

    # taux d'utilisation du crédit
    df["utilization_rate"] = df["BILL_AMT1"] / limit
    df["avg_utilization_rate"] = df[BILL_COLS].mean(axis=1) / limit
    df["max_utilization"] = df[BILL_COLS].max(axis=1) / limit

    # comportement de remboursement mois par mois
    # PAY_AMT_i règle la facture du mois précédent (BILL_AMT_{i+1})
    for i in range(1, 6):
        bill = df[f"BILL_AMT{i+1}"]
        paid = df[f"PAY_AMT{i}"]
        ratio = np.where(bill > 0, paid / bill.where(bill > 0, 1), 1.0)
        df[f"pay_ratio_{i}"] = np.clip(ratio, 0, 2)
        df[f"paid_full_{i}"] = ((bill <= 0) | (paid >= bill)).astype(int)
        df[f"unpaid_gap_{i}"] = (bill - paid).clip(lower=0)

    ratio_cols = [f"pay_ratio_{i}" for i in range(1, 6)]
    gap_cols = [f"unpaid_gap_{i}" for i in range(1, 6)]
    df["avg_pay_ratio"] = df[ratio_cols].mean(axis=1)
    df["n_months_paid_full"] = df[[f"paid_full_{i}" for i in range(1, 6)]].sum(axis=1)
    df["total_unpaid_gap"] = df[gap_cols].sum(axis=1)
    df["unpaid_to_limit"] = df["total_unpaid_gap"] / limit
    df["pay_to_limit"] = df[PAY_AMT_COLS].mean(axis=1) / limit

    # historique de retard : sévérité, fréquence, dynamique
    df["pay_delay_max"] = df[PAY_COLS].max(axis=1)
    df["pay_delay_mean"] = df[PAY_COLS].mean(axis=1)
    df["pay_delay_std"] = df[PAY_COLS].std(axis=1)
    df["n_months_late"] = (df[PAY_COLS] > 0).sum(axis=1)
    df["ever_serious_delay"] = (df["pay_delay_max"] >= 2).astype(int)
    df["recent_vs_past_delay"] = df["PAY_1"] - df[PAY_COLS[1:]].mean(axis=1)
    df["pay_delay_last3"] = df[["PAY_1", "PAY_2", "PAY_3"]].mean(axis=1)
    df["pay_delay_first3"] = df[["PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
    df["delay_acceleration"] = df["pay_delay_last3"] - df["pay_delay_first3"]
    for i in range(1, 6):
        df[f"delta_pay_{i}"] = df[f"PAY_{i}"] - df[f"PAY_{i+1}"]

    # montants : niveau, tendance, volatilité
    df["avg_bill_amt"] = df[BILL_COLS].mean(axis=1)
    df["avg_pay_amt"] = df[PAY_AMT_COLS].mean(axis=1)
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["bill_std"] = df[BILL_COLS].std(axis=1)
    df["pay_amt_std"] = df[PAY_AMT_COLS].std(axis=1)
    df["log_limit"] = np.log1p(df["LIMIT_BAL"])
    df["log_avg_bill"] = np.log1p(df["avg_bill_amt"].clip(lower=0))
    df["log_avg_pay"] = np.log1p(df["avg_pay_amt"])

    return df


ENGINEERED_FEATURES = (
    ["utilization_rate", "avg_utilization_rate", "max_utilization"]
    + [f"pay_ratio_{i}" for i in range(1, 6)]
    + [f"paid_full_{i}" for i in range(1, 6)]
    + [f"unpaid_gap_{i}" for i in range(1, 6)]
    + [f"delta_pay_{i}" for i in range(1, 6)]
    + ["avg_pay_ratio", "n_months_paid_full", "total_unpaid_gap", "unpaid_to_limit",
       "pay_to_limit", "pay_delay_max", "pay_delay_mean", "pay_delay_std", "n_months_late",
       "ever_serious_delay", "recent_vs_past_delay", "pay_delay_last3", "pay_delay_first3",
       "delay_acceleration", "avg_bill_amt", "avg_pay_amt", "bill_trend", "bill_std",
       "pay_amt_std", "log_limit", "log_avg_bill", "log_avg_pay"]
)


# --------------------------------------------------------------------------- vues
def get_tabular(path: str | Path = DEFAULT_DATA_PATH, engineered: bool = True):
    """Retourne (X, y, feature_cols) pour les modèles tabulaires."""
    df = add_features(load_clean(path))
    cols = RAW_FEATURES + (ENGINEERED_FEATURES if engineered else [])
    X = df[cols].astype(float)
    y = df["DEFAULT"].astype(int)
    assert np.isfinite(X.to_numpy()).all(), "valeurs non finies dans les features"
    return X, y, cols


def get_sequences(path: str | Path = DEFAULT_DATA_PATH):
    """Retourne (X_seq, X_static, y) pour les modèles séquentiels.

    X_seq    : (n, 6, 5) — 6 mois en ordre chronologique (t=0 le plus ancien)
               features par pas de temps : statut de retard, utilisation, part du plafond
               remboursée, log-facture, log-paiement.
    X_static : (n, 11)   — log(plafond), âge normalisé, SEX/EDUCATION/MARRIAGE en one-hot.
    """
    df = load_clean(path)
    limit = (df["LIMIT_BAL"] + 1).to_numpy()[:, None]

    # indices 1..6 = du plus récent au plus ancien -> on inverse pour l'ordre chronologique
    order = list(range(6, 0, -1))
    pay = df[[f"PAY_{i}" for i in order]].to_numpy(dtype=float)
    bill = df[[f"BILL_AMT{i}" for i in order]].to_numpy(dtype=float)
    amt = df[[f"PAY_AMT{i}" for i in order]].to_numpy(dtype=float)

    X_seq = np.stack(
        [
            pay,                                  # statut de retard du mois
            bill / limit,                         # taux d'utilisation
            amt / limit,                          # remboursement rapporté au plafond
            np.log1p(np.clip(bill, 0, None)),     # niveau de facture
            np.log1p(amt),                        # niveau de remboursement
        ],
        axis=-1,
    ).astype(np.float32)

    static = [np.log1p(df["LIMIT_BAL"].to_numpy(dtype=float)), df["AGE"].to_numpy(dtype=float)]
    for col, n_cat in [("SEX", 2), ("EDUCATION", 4), ("MARRIAGE", 3)]:
        for k in range(1, n_cat + 1):
            static.append((df[col] == k).to_numpy(dtype=float))
    X_static = np.stack(static, axis=-1).astype(np.float32)

    y = df["DEFAULT"].to_numpy(dtype=np.int64)
    assert np.isfinite(X_seq).all() and np.isfinite(X_static).all()
    return X_seq, X_static, y


def train_test_indices(y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Split stratifié partagé par toutes les expériences (indices positionnels)."""
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=test_size, stratify=y, random_state=random_state)
