"""Expérimentation complète, robuste aux interruptions.

Principe : pour chaque modèle on calcule les probabilités **out-of-fold** (OOF) sur le train
(5-fold stratifié, protocole commun) et on les sauvegarde sur disque, une par modèle.

Avantages de ce design :
  - reprise après crash : un modèle déjà calculé est ignoré au relancement
  - les métriques se recalculent depuis les OOF, sans réentraîner
  - le blending (moyenne de modèles) devient gratuit : il suffit de moyenner les fichiers OOF
  - les mêmes OOF serviront à calibrer le seuil de décision sans fuite

Le test set n'est jamais touché ici : il est réservé à l'évaluation finale (scripts/final_eval.py).

Usage :
    python scripts/run_experiments.py            # tout (reprend là où ça s'est arrêté)
    python scripts/run_experiments.py --only histgb catboost
    python scripts/run_experiments.py --stage tune|oof|analyze
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

SEED = 42
N_JOBS = 2                       # bridé : la machine a 4 cœurs et peu de RAM
OOF_DIR = ROOT / "reports" / "oof"
PARAMS_FILE = ROOT / "reports" / "best_params.json"
METRICS_FILE = ROOT / "reports" / "model_ranking.csv"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

TABULAR_MODELS = ["logreg", "rf", "histgb", "histgb_tuned", "xgb_base", "xgb_tuned",
                  "lgbm_tuned", "catboost"]
DEEP_MODELS = ["mlp", "lstm", "gru", "bilstm", "transformer"]
ALL_MODELS = TABULAR_MODELS + DEEP_MODELS


# =========================================================================== modèles tabulaires
def build_tabular(name: str, spw: float, params: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    if name == "logreg":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=SEED))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      n_jobs=N_JOBS, random_state=SEED)
    if name == "histgb":
        return HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                              random_state=SEED)
    if name == "histgb_tuned":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED,
                                              **params.get("histgb", {}))
    if name == "xgb_base":
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                             scale_pos_weight=spw, eval_metric="logloss",
                             random_state=SEED, n_jobs=N_JOBS)
    if name == "xgb_tuned":
        return XGBClassifier(**params.get("xgboost", {}), scale_pos_weight=spw,
                             eval_metric="logloss", random_state=SEED, n_jobs=N_JOBS)
    if name == "lgbm_tuned":
        return LGBMClassifier(**params.get("lightgbm", {}), scale_pos_weight=spw,
                              random_state=SEED, n_jobs=N_JOBS, verbose=-1)
    if name == "catboost":
        return CatBoostClassifier(iterations=500, scale_pos_weight=spw, random_seed=SEED,
                                  verbose=0, thread_count=N_JOBS)
    raise ValueError(name)


def oof_tabular(name: str, X, y, idx_train, spw, params) -> np.ndarray:
    """Probabilités out-of-fold sur le train (chaque ligne prédite par un modèle qui ne l'a pas vue)."""
    X_tr = X.iloc[idx_train].to_numpy(dtype=np.float32)
    y_tr = y.iloc[idx_train].to_numpy()
    oof = np.zeros(len(idx_train), dtype=np.float64)

    for tr, va in CV.split(X_tr, y_tr):
        model = build_tabular(name, spw, params)
        model.fit(X_tr[tr], y_tr[tr])
        oof[va] = model.predict_proba(X_tr[va])[:, 1]
        del model
        gc.collect()
    return oof


# =========================================================================== modèles profonds
def oof_deep(name: str, y, idx_train) -> np.ndarray:
    import torch
    import torch.nn as nn

    torch.set_num_threads(N_JOBS)
    X_seq, X_static, y_np = dp.get_sequences()
    # la matrice tabulaire n'est chargée que pour le MLP (économie de mémoire : la machine
    # a peu de RAM et les processus ont déjà été tués une fois pour cette raison)
    if name == "mlp":
        X_tab = dp.get_tabular()[0].to_numpy(dtype=np.float32)
    else:
        X_tab = None

    class SequenceNet(nn.Module):
        def __init__(self, kind, n_seq, n_static, hidden=64, dropout=0.3):
            super().__init__()
            self.kind = kind
            if kind == "lstm":
                self.enc = nn.LSTM(n_seq, hidden, batch_first=True)
            elif kind == "gru":
                self.enc = nn.GRU(n_seq, hidden, batch_first=True)
            elif kind == "bilstm":
                self.enc = nn.LSTM(n_seq, hidden // 2, batch_first=True, bidirectional=True)
            elif kind == "transformer":
                self.proj = nn.Linear(n_seq, hidden)
                self.pos = nn.Parameter(torch.randn(1, 6, hidden) * 0.02)
                layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2,
                                                   dropout=dropout, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.static_net = nn.Sequential(nn.Linear(n_static, 32), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(hidden + 32, 64), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))

        def forward(self, seq, static):
            if self.kind == "transformer":
                h = self.enc(self.proj(seq) + self.pos).mean(dim=1)
            else:
                out, _ = self.enc(seq)
                h = out[:, -1, :]
            return self.head(torch.cat([h, self.static_net(static)], dim=1)).squeeze(1)

    class TabularMLP(nn.Module):
        def __init__(self, n_feat, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1))

        def forward(self, x):
            return self.net(x).squeeze(1)

    def train_one(model, tr_pack, es_pack, pos_weight, max_epochs=80, patience=10, bs=256):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
        n = len(tr_pack[-1])
        best_ap, best_state, bad = -np.inf, None, 0
        for _ in range(max_epochs):
            model.train()
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss_fn(model(*[t[b] for t in tr_pack[:-1]]), tr_pack[-1][b]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                ap = average_precision_score(es_pack[-1].numpy(),
                                             model(*es_pack[:-1]).numpy())
            if ap > best_ap + 1e-5:
                best_ap, bad = ap, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            model.load_state_dict(best_state)
        return model

    y_tr = y.iloc[idx_train].to_numpy()
    oof = np.zeros(len(idx_train), dtype=np.float64)

    for fold, (tr, va) in enumerate(CV.split(np.zeros(len(y_tr)), y_tr)):
        torch.manual_seed(SEED + fold)
        abs_tr, abs_va = idx_train[tr], idx_train[va]
        sub_tr, sub_es = train_test_split(abs_tr, test_size=0.15,
                                          stratify=y_np[abs_tr], random_state=SEED)
        pos_weight = (y_np[sub_tr] == 0).sum() / (y_np[sub_tr] == 1).sum()

        if name == "mlp":
            sc = StandardScaler().fit(X_tab[sub_tr])
            pack = lambda idx: (torch.tensor(sc.transform(X_tab[idx]), dtype=torch.float32),
                                torch.tensor(y_np[idx], dtype=torch.float32))
            model = TabularMLP(X_tab.shape[1])
        else:
            nf = X_seq.shape[-1]
            sc_s = StandardScaler().fit(X_seq[sub_tr].reshape(-1, nf))
            sc_t = StandardScaler().fit(X_static[sub_tr])
            pack = lambda idx: (
                torch.tensor(sc_s.transform(X_seq[idx].reshape(-1, nf)).reshape(len(idx), 6, nf),
                             dtype=torch.float32),
                torch.tensor(sc_t.transform(X_static[idx]), dtype=torch.float32),
                torch.tensor(y_np[idx], dtype=torch.float32))
            model = SequenceNet(name, nf, X_static.shape[1])

        model = train_one(model, pack(sub_tr), pack(sub_es), pos_weight)
        model.eval()
        with torch.no_grad():
            oof[va] = torch.sigmoid(model(*pack(abs_va)[:-1])).numpy()
        del model
        gc.collect()
    return oof


# =========================================================================== étapes
def stage_tune(X, y, idx_train, spw) -> dict:
    """RandomizedSearchCV (3-fold, 20 tirages) pour XGBoost, LightGBM et HistGradientBoosting."""
    if PARAMS_FILE.exists():
        print(f"Hyperparamètres déjà présents ({PARAMS_FILE.name}), étape ignorée.")
        return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))

    from scipy.stats import loguniform, randint, uniform
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    spaces = {
        "xgboost": (
            XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=SEED,
                          n_jobs=N_JOBS),
            {"max_depth": randint(2, 9), "learning_rate": loguniform(0.01, 0.3),
             "n_estimators": randint(200, 900), "subsample": uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.4, 0.6), "min_child_weight": randint(1, 20),
             "reg_lambda": loguniform(0.1, 20), "reg_alpha": loguniform(1e-3, 5),
             "gamma": uniform(0, 5)}),
        "lightgbm": (
            LGBMClassifier(scale_pos_weight=spw, random_state=SEED, n_jobs=N_JOBS, verbose=-1),
            {"num_leaves": randint(15, 150), "max_depth": randint(3, 12),
             "learning_rate": loguniform(0.01, 0.3), "n_estimators": randint(200, 900),
             "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.4, 0.6),
             "min_child_samples": randint(10, 120), "reg_lambda": loguniform(1e-3, 20),
             "reg_alpha": loguniform(1e-3, 5)}),
        "histgb": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
            {"max_iter": randint(150, 700), "learning_rate": loguniform(0.01, 0.3),
             "max_leaf_nodes": randint(10, 80), "min_samples_leaf": randint(10, 150),
             "l2_regularization": loguniform(1e-3, 20), "max_bins": randint(64, 255)}),
    }

    best = {}
    for key, (estimator, space) in spaces.items():
        t0 = time.time()
        search = RandomizedSearchCV(estimator, space, n_iter=20, scoring="average_precision",
                                    cv=cv3, random_state=SEED, n_jobs=1)
        search.fit(X_tr, y_tr)
        best[key] = {k: (int(v) if isinstance(v, (np.integer, int)) else float(v))
                     for k, v in search.best_params_.items()}
        print(f"  {key:<10} AP(3-fold)={search.best_score_:.4f}  ({time.time()-t0:.0f}s)")
        print(f"             {best[key]}", flush=True)
        gc.collect()

    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def stage_oof(models, X, y, idx_train, spw, params) -> None:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    y_tr = y.iloc[idx_train].to_numpy()

    for name in models:
        path = OOF_DIR / f"{name}.npy"
        if path.exists():
            print(f"  {name:<14} déjà calculé, ignoré.")
            continue
        t0 = time.time()
        oof = oof_deep(name, y, idx_train) if name in DEEP_MODELS \
            else oof_tabular(name, X, y, idx_train, spw, params)
        np.save(path, oof)
        ap, auc = average_precision_score(y_tr, oof), roc_auc_score(y_tr, oof)
        print(f"  {name:<14} PR-AUC={ap:.4f}  ROC-AUC={auc:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)
        gc.collect()


def stage_analyze(y, idx_train) -> None:
    """Classement des modèles + recherche du meilleur blend (moyenne de rangs)."""
    from itertools import combinations
    from scipy.stats import rankdata

    y_tr = y.iloc[idx_train].to_numpy()
    oofs = {p.stem: np.load(p) for p in sorted(OOF_DIR.glob("*.npy"))}
    if not oofs:
        print("Aucun OOF trouvé.")
        return

    rows = [{"model": n, "pr_auc": average_precision_score(y_tr, o),
             "roc_auc": roc_auc_score(y_tr, o)} for n, o in oofs.items()]
    ranking = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)

    print("\n" + "=" * 74)
    print("CLASSEMENT INDIVIDUEL (OOF sur le train)")
    print("=" * 74)
    print(ranking.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # blending : moyenne des rangs (robuste aux échelles de probabilité différentes)
    ranked = {n: rankdata(o) / len(o) for n, o in oofs.items()}
    names = list(ranking["model"])
    blends = []
    for k in (2, 3, 4, 5):
        for combo in combinations(names, k):
            blend = np.mean([ranked[n] for n in combo], axis=0)
            blends.append({"blend": " + ".join(combo), "k": k,
                           "pr_auc": average_precision_score(y_tr, blend),
                           "roc_auc": roc_auc_score(y_tr, blend)})
    blends_df = pd.DataFrame(blends).sort_values("pr_auc", ascending=False)

    print("\n" + "=" * 74)
    print("TOP 10 BLENDS (moyenne des rangs des OOF)")
    print("=" * 74)
    print(blends_df.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ranking.to_csv(METRICS_FILE, index=False)
    blends_df.head(50).to_csv(ROOT / "reports" / "blend_ranking.csv", index=False)
    print(f"\n-> {METRICS_FILE}")
    print(f"-> {ROOT / 'reports' / 'blend_ranking.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--stage", choices=["tune", "oof", "analyze", "all"], default="all")
    args = ap.parse_args()

    X, y, cols = dp.get_tabular()
    idx_train, idx_test = dp.train_test_indices(y)
    n_neg, n_pos = np.bincount(y.iloc[idx_train])
    spw = n_neg / n_pos
    print(f"Train {X.iloc[idx_train].shape} | scale_pos_weight={spw:.3f}\n")

    params = {}
    if args.stage in ("tune", "all"):
        print("[Étape 1] Tuning d'hyperparamètres")
        params = stage_tune(X, y, idx_train, spw)
    elif PARAMS_FILE.exists():
        params = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))

    if args.stage in ("oof", "all"):
        models = args.only or ALL_MODELS
        print("\n[Étape 2] Prédictions out-of-fold")
        stage_oof(models, X, y, idx_train, spw, params)

    if args.stage in ("analyze", "all"):
        stage_analyze(y, idx_train)


if __name__ == "__main__":
    main()
