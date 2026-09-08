"""Expérience 2 — modèles de deep learning (MLP, LSTM, GRU, Transformer).

Le dataset a une vraie structure séquentielle : chaque client est une série de 6 mois
(statut de retard, facture, remboursement) + des attributs statiques. Les modèles récurrents
sont donc appliqués sur cette séquence en ordre chronologique, avec une seconde branche
pour les variables statiques :

    séquence (B, 6, 5) --> LSTM/GRU/Transformer --> représentation (B, H) --.
                                                                            |--> MLP --> logit
    statiques (B, 11) ---> Linear+ReLU --------> représentation (B, 32) ----'

Protocole identique à l'expérience tabulaire (5-fold stratifié sur le train, PR-AUC/ROC-AUC),
pour que les chiffres soient directement comparables à XGBoost & co.
Le déséquilibre est géré par `pos_weight` dans la BCE — l'équivalent exact de
`scale_pos_weight`, la stratégie qui s'est révélée la meilleure sur les modèles à arbres.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

SEED = 42
DEVICE = torch.device("cpu")
OUT_CSV = ROOT / "reports" / "exp_deep_results.csv"

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))


# --------------------------------------------------------------------------- modèles
class SequenceNet(nn.Module):
    """Encodeur séquentiel (LSTM / GRU / Transformer) + branche statique."""

    def __init__(self, kind: str, n_seq_feat: int, n_static: int, hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.encoder = nn.LSTM(n_seq_feat, hidden, num_layers=1, batch_first=True)
        elif kind == "gru":
            self.encoder = nn.GRU(n_seq_feat, hidden, num_layers=1, batch_first=True)
        elif kind == "bilstm":
            self.encoder = nn.LSTM(n_seq_feat, hidden // 2, num_layers=1, batch_first=True,
                                   bidirectional=True)
        elif kind == "transformer":
            self.input_proj = nn.Linear(n_seq_feat, hidden)
            self.pos = nn.Parameter(torch.randn(1, 6, hidden) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
                dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        else:
            raise ValueError(kind)

        self.static_net = nn.Sequential(nn.Linear(n_static, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, seq, static):
        if self.kind == "transformer":
            h = self.encoder(self.input_proj(seq) + self.pos).mean(dim=1)
        else:
            out, _ = self.encoder(seq)
            h = out[:, -1, :]          # dernier pas de temps = mois le plus récent
        return self.head(torch.cat([h, self.static_net(static)], dim=1)).squeeze(1)


class TabularMLP(nn.Module):
    """MLP de référence sur les 68 features tabulaires (le deep learning aide-t-il ici ?)."""

    def __init__(self, n_feat: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


# --------------------------------------------------------------------------- entraînement
def train_one(model, tensors_tr, tensors_va, pos_weight, max_epochs=80, patience=10,
              batch_size=256, lr=1e-3):
    """Entraîne avec early stopping sur la PR-AUC de validation. Retourne le meilleur état."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    n = len(tensors_tr[-1])

    best_ap, best_state, bad_epochs = -np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch = perm[start:start + batch_size]
            if len(batch) < 2:      # BatchNorm exige au moins 2 échantillons
                continue
            opt.zero_grad()
            inputs = [t[batch] for t in tensors_tr[:-1]]
            loss = loss_fn(model(*inputs), tensors_tr[-1][batch])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(*tensors_va[:-1])
            ap = average_precision_score(tensors_va[-1].numpy(), logits.numpy())
        if ap > best_ap + 1e-5:
            best_ap, bad_epochs = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, tensors):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(*tensors)).numpy()


# --------------------------------------------------------------------------- évaluation CV
def run_cv(name: str, build_fn, data_fn, y, idx_train, results: list):
    """5-fold stratifié ; dans chaque fold, 15% du train sert à l'early stopping."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_tr_all = y[idx_train]
    aps, aucs = [], []
    t0 = time.time()

    for fold, (tr, va) in enumerate(cv.split(np.zeros(len(y_tr_all)), y_tr_all)):
        torch.manual_seed(SEED + fold)
        abs_tr, abs_va = idx_train[tr], idx_train[va]
        sub_tr, sub_es = train_test_split(
            abs_tr, test_size=0.15, stratify=y[abs_tr], random_state=SEED)

        tensors_tr, tensors_es, tensors_va = data_fn(sub_tr, sub_es, abs_va)
        pos_weight = (y[sub_tr] == 0).sum() / (y[sub_tr] == 1).sum()

        model = build_fn()
        model = train_one(model, tensors_tr, tensors_es, pos_weight)

        proba = predict(model, tensors_va[:-1])
        aps.append(average_precision_score(y[abs_va], proba))
        aucs.append(roc_auc_score(y[abs_va], proba))

    ap, auc = np.array(aps), np.array(aucs)
    results.append({"model": name, "pr_auc": ap.mean(), "pr_auc_std": ap.std(),
                    "roc_auc": auc.mean(), "roc_auc_std": auc.std(),
                    "fit_time_s": time.time() - t0})
    print(f"  {name:<34} PR-AUC={ap.mean():.4f} +/-{ap.std():.4f}   "
          f"ROC-AUC={auc.mean():.4f} +/-{auc.std():.4f}   ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    X_tab, y_series, cols = dp.get_tabular()
    X_seq, X_static, y = dp.get_sequences()
    X_tab = X_tab.to_numpy(dtype=np.float32)
    idx_train, idx_test = dp.train_test_indices(y)

    print(f"Séquences {X_seq.shape} | statiques {X_static.shape} | tabulaire {X_tab.shape}")
    print(f"Train n={len(idx_train)}\n")
    results: list[dict] = []

    def make_seq_data(sub_tr, sub_es, va):
        """Scale la séquence et les statiques sur le sous-train uniquement (pas de fuite)."""
        n_feat = X_seq.shape[-1]
        sc_seq = StandardScaler().fit(X_seq[sub_tr].reshape(-1, n_feat))
        sc_stat = StandardScaler().fit(X_static[sub_tr])

        def pack(idx):
            seq = sc_seq.transform(X_seq[idx].reshape(-1, n_feat)).reshape(len(idx), 6, n_feat)
            return (torch.tensor(seq, dtype=torch.float32),
                    torch.tensor(sc_stat.transform(X_static[idx]), dtype=torch.float32),
                    torch.tensor(y[idx], dtype=torch.float32))

        return pack(sub_tr), pack(sub_es), pack(va)

    def make_tab_data(sub_tr, sub_es, va):
        sc = StandardScaler().fit(X_tab[sub_tr])

        def pack(idx):
            return (torch.tensor(sc.transform(X_tab[idx]), dtype=torch.float32),
                    torch.tensor(y[idx], dtype=torch.float32))

        return pack(sub_tr), pack(sub_es), pack(va)

    print("[1/2] MLP sur features tabulaires")
    run_cv("MLP (tabulaire, 68 feat.)", lambda: TabularMLP(X_tab.shape[1]),
           make_tab_data, y, idx_train, results)

    print("\n[2/2] Modèles séquentiels (6 mois en ordre chronologique)")
    for kind, label in [("lstm", "LSTM"), ("gru", "GRU"),
                        ("bilstm", "BiLSTM"), ("transformer", "Transformer")]:
        run_cv(f"{label} (séquence + statiques)",
               lambda k=kind: SequenceNet(k, X_seq.shape[-1], X_static.shape[1]),
               make_seq_data, y, idx_train, results)

    df_res = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 88)
    print("CLASSEMENT DEEP LEARNING (PR-AUC, 5-fold CV sur le train)")
    print("=" * 88)
    print(df_res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nRésultats -> {OUT_CSV}")


if __name__ == "__main__":
    main()
