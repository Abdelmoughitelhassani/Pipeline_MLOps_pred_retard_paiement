"""Diagnostic : où est le plafond de performance ?

Compare plusieurs jeux de features avec le MEME modele (XGBoost + scale_pos_weight)
pour isoler l'effet du feature engineering de l'effet du modele.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "default_of_credit_card_clients.xls"

# ---------------------------------------------------------------- chargement + nettoyage
df = pd.read_excel(DATA, sheet_name=0, header=1)
df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "DEFAULT"})
df.columns = df.columns.str.strip()
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
pay_cols = [f"PAY_{i}" for i in range(1, 7)]

RAW = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + pay_cols + bill_cols + pay_amt_cols

# ---------------------------------------------------------------- FE v1 (celui du notebook)
df["utilization_rate"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)
df["avg_utilization_rate"] = df[bill_cols].mean(axis=1) / (df["LIMIT_BAL"] + 1)
for i in range(1, 6):
    df[f"payment_ratio_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i+1}"].clip(lower=0) + 1)
df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
df["pay_delay_max"] = df[pay_cols].max(axis=1)
df["pay_delay_mean"] = df[pay_cols].mean(axis=1)
df["n_months_late"] = (df[pay_cols] > 0).sum(axis=1)
df["recent_vs_past_delay"] = df["PAY_1"] - df[pay_cols[1:]].mean(axis=1)
df["ever_serious_delay"] = (df["pay_delay_max"] >= 2).astype(int)
df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
df["avg_pay_amt"] = df[pay_amt_cols].mean(axis=1)

FE_V1_NEW = ["utilization_rate", "avg_utilization_rate"] + [f"payment_ratio_{i}" for i in range(1, 6)] + [
    "bill_trend", "pay_delay_max", "pay_delay_mean", "n_months_late",
    "recent_vs_past_delay", "ever_serious_delay", "avg_bill_amt", "avg_pay_amt"]

SET_V1 = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_1", "BILL_AMT1", "PAY_AMT1"] + FE_V1_NEW

# ---------------------------------------------------------------- FE v2 (corrige + enrichi)
# Probleme identifie en v1 : payment_ratio explose quand BILL_AMT ~ 0 (denominateur = 1),
# le "ratio" devient un montant brut -> feature incoherente, correlation ~ 0.
for i in range(1, 6):
    bill = df[f"BILL_AMT{i+1}"]
    paid = df[f"PAY_AMT{i}"]
    ratio = np.where(bill > 0, paid / bill.where(bill > 0, 1), 1.0)  # rien du -> considere paye
    df[f"pay_ratio_v2_{i}"] = np.clip(ratio, 0, 2)                    # borne les outliers
    df[f"paid_full_{i}"] = ((bill <= 0) | (paid >= bill)).astype(int)
    df[f"unpaid_gap_{i}"] = (bill - paid).clip(lower=0)

df["n_months_paid_full"] = df[[f"paid_full_{i}" for i in range(1, 6)]].sum(axis=1)
df["avg_pay_ratio"] = df[[f"pay_ratio_v2_{i}" for i in range(1, 6)]].mean(axis=1)
df["total_unpaid_gap"] = df[[f"unpaid_gap_{i}" for i in range(1, 6)]].sum(axis=1)

# volatilite / dynamique
df["bill_std"] = df[bill_cols].std(axis=1)
df["pay_amt_std"] = df[pay_amt_cols].std(axis=1)
df["max_utilization"] = df[bill_cols].max(axis=1) / (df["LIMIT_BAL"] + 1)
df["pay_to_limit"] = df[pay_amt_cols].mean(axis=1) / (df["LIMIT_BAL"] + 1)
df["unpaid_to_limit"] = df["total_unpaid_gap"] / (df["LIMIT_BAL"] + 1)
for i in range(1, 6):
    df[f"delta_pay_{i}"] = df[f"PAY_{i}"] - df[f"PAY_{i+1}"]
df["pay_delay_std"] = df[pay_cols].std(axis=1)
df["pay_delay_last3"] = df[["PAY_1", "PAY_2", "PAY_3"]].mean(axis=1)
df["pay_delay_first3"] = df[["PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
df["delay_acceleration"] = df["pay_delay_last3"] - df["pay_delay_first3"]
df["log_limit"] = np.log1p(df["LIMIT_BAL"])
df["log_avg_bill"] = np.log1p(df[bill_cols].mean(axis=1).clip(lower=0))
df["log_avg_pay"] = np.log1p(df[pay_amt_cols].mean(axis=1))

FE_V2_NEW = (
    [f"pay_ratio_v2_{i}" for i in range(1, 6)]
    + [f"paid_full_{i}" for i in range(1, 6)]
    + [f"unpaid_gap_{i}" for i in range(1, 6)]
    + [f"delta_pay_{i}" for i in range(1, 6)]
    + ["n_months_paid_full", "avg_pay_ratio", "total_unpaid_gap", "bill_std", "pay_amt_std",
       "max_utilization", "pay_to_limit", "unpaid_to_limit", "pay_delay_std",
       "pay_delay_last3", "pay_delay_first3", "delay_acceleration",
       "log_limit", "log_avg_bill", "log_avg_pay",
       "utilization_rate", "avg_utilization_rate", "bill_trend", "pay_delay_max",
       "pay_delay_mean", "n_months_late", "recent_vs_past_delay", "ever_serious_delay",
       "avg_bill_amt", "avg_pay_amt"]
)

FEATURE_SETS = {
    "A. Brut seul (23 feat.)": RAW,
    "B. FE v1 seul = notebook (23 feat.)": SET_V1,
    "C. Brut + FE v1 (38 feat.)": RAW + FE_V1_NEW,
    "D. Brut + FE v2 corrige (%d feat.)" % len(RAW + FE_V2_NEW): RAW + FE_V2_NEW,
}

y = df["DEFAULT"]
idx_train, idx_test = train_test_split(df.index, test_size=0.20, stratify=y, random_state=42)

n_neg, n_pos = np.bincount(y.loc[idx_train])
spw = n_neg / n_pos
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"scale_pos_weight = {spw:.3f}\n")
print(f"{'Jeu de features':<42} {'PR-AUC':<18} {'ROC-AUC':<18}")
print("-" * 80)

rows = []
for name, cols in FEATURE_SETS.items():
    X = df.loc[idx_train, cols]
    assert np.isfinite(X.to_numpy(dtype=float)).all(), f"valeurs non finies dans {name}"
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                          scale_pos_weight=spw, eval_metric="logloss",
                          random_state=42, n_jobs=-1)
    res = cross_validate(model, X, y.loc[idx_train], cv=cv,
                         scoring=["average_precision", "roc_auc"], n_jobs=1)
    ap, auc = res["test_average_precision"], res["test_roc_auc"]
    print(f"{name:<42} {ap.mean():.4f} +/- {ap.std():.4f}   {auc.mean():.4f} +/- {auc.std():.4f}")
    rows.append({"features": name, "n_features": len(cols),
                 "pr_auc": ap.mean(), "pr_auc_std": ap.std(),
                 "roc_auc": auc.mean(), "roc_auc_std": auc.std()})

pd.DataFrame(rows).to_csv(ROOT / "reports" / "ablation_results.csv", index=False)

# correlation des ratios corriges vs v1
print("\nCorrelation avec DEFAULT — payment_ratio v1 (casse) vs v2 (corrige) :")
for i in range(1, 6):
    c1 = df[f"payment_ratio_{i}"].corr(df["DEFAULT"])
    c2 = df[f"pay_ratio_v2_{i}"].corr(df["DEFAULT"])
    c3 = df[f"paid_full_{i}"].corr(df["DEFAULT"])
    print(f"  mois {i}: v1={c1:+.4f}   v2={c2:+.4f}   paid_full={c3:+.4f}")

print("\nTop correlations des nouvelles features v2 :")
new_corr = df[FE_V2_NEW + ["DEFAULT"]].corr()["DEFAULT"].drop("DEFAULT")
print(new_corr.sort_values(key=abs, ascending=False).head(15).to_string())
