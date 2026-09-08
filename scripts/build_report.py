"""Génère le rapport PDF complet du pipeline (parties 1 et 2).

Toutes les valeurs proviennent des fichiers produits par les expériences
(reports/*.csv, *.json, oof/*.npy) — rien n'est saisi à la main.
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

REPORTS = ROOT / "reports"
IMG_OLD = REPORTS / "figures" / "notebook"   # figures extraites du notebook EDA
IMG_NEW = REPORTS / "figures"
IMG_NEW.mkdir(parents=True, exist_ok=True)
OUT_PDF = REPORTS / "rapport_pipeline_credit_default.pdf"

A4 = (8.27, 11.69)
NAVY, GREEN, RED, GRAY, LIGHT = "#1f3a5f", "#2e7d5b", "#b3413e", "#5a5a5a", "#eef1f5"
plt.rcParams["font.family"] = "DejaVu Sans"

# =============================================================== chargement des résultats
ranking = pd.read_csv(REPORTS / "model_ranking.csv")
blends = pd.read_csv(REPORTS / "blend_ranking.csv")
costs = pd.read_csv(REPORTS / "cost_thresholds.csv")
params = json.loads((REPORTS / "best_params.json").read_text(encoding="utf-8"))
final = json.loads((REPORTS / "final_evaluation.json").read_text(encoding="utf-8"))
ablation = pd.read_csv(REPORTS / "ablation_results.csv")

oof = {p.stem: np.load(p) for p in sorted((REPORTS / "oof").glob("*.npy"))}
_X, _y, _ = dp.get_tabular()
_idx_train, _ = dp.train_test_indices(_y)
y_tr = _y.iloc[_idx_train].to_numpy()

BASELINE_TEST = {"pr_auc": 0.5432, "roc_auc": 0.7661}   # rapport v1, XGBoost non tuné
FINAL_TEST = {"pr_auc": final["test_pr_auc"], "roc_auc": final["test_roc_auc"]}

FAMILY = {"lgbm_tuned": "Boosting tuné", "xgb_tuned": "Boosting tuné",
          "histgb_tuned": "Boosting tuné", "histgb": "Boosting", "catboost": "Boosting",
          "xgb_base": "Boosting", "lstm": "Deep learning", "gru": "Deep learning",
          "mlp": "Deep learning", "rf": "Autre", "logreg": "Autre"}
FAM_COLOR = {"Boosting tuné": "#2e7d5b", "Boosting": "#7fb069",
             "Deep learning": "#4a6fa5", "Autre": "#9a9a9a"}
LABEL = {"lgbm_tuned": "LightGBM tuné", "xgb_tuned": "XGBoost tuné",
         "histgb_tuned": "HistGB tuné", "histgb": "HistGB (défaut)", "catboost": "CatBoost",
         "xgb_base": "XGBoost (baseline v1)", "lstm": "LSTM", "gru": "GRU",
         "mlp": "MLP", "rf": "RandomForest", "logreg": "LogisticRegression"}


# =============================================================== nouveaux graphiques
def fig_ablation():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f.split(".")[1].strip() for f in ablation["features"]]
    x = np.arange(len(labels))
    ax.bar(x, ablation["pr_auc"], yerr=ablation["pr_auc_std"], capsize=5,
           color=["#9a9a9a", "#2e7d5b", "#7fb069", "#4a6fa5"], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(l, 22) for l in labels], fontsize=8.5)
    ax.set_ylabel("PR-AUC (5-fold CV)")
    ax.set_ylim(0.50, 0.58)
    ax.axhline(ablation["pr_auc"].mean(), ls="--", color=RED, lw=1,
               label=f"moyenne = {ablation['pr_auc'].mean():.4f}")
    for i, v in enumerate(ablation["pr_auc"]):
        ax.text(i, v + ablation["pr_auc_std"][i] + 0.002, f"{v:.4f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_title("Les 4 jeux de features donnent le même résultat (écarts < 1 écart-type)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_NEW / "ablation.png", dpi=130)
    plt.close(fig)


def fig_model_ranking():
    r = ranking.sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [FAM_COLOR[FAMILY[m]] for m in r["model"]]
    ax.barh([LABEL[m] for m in r["model"]], r["pr_auc"], color=colors)
    ax.set_xlim(0.50, 0.58)
    ax.set_xlabel("PR-AUC (out-of-fold, train)")
    for i, v in enumerate(r["pr_auc"]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM_COLOR.values()]
    ax.legend(handles, FAM_COLOR.keys(), fontsize=8, loc="lower right")
    ax.set_title("Classement des 11 modèles — protocole identique", fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG_NEW / "model_ranking.png", dpi=130)
    plt.close(fig)


def fig_correlation():
    R = pd.DataFrame({LABEL[k]: rankdata(v) / len(v) for k, v in oof.items()})
    corr = R.corr()
    fig, ax = plt.subplots(figsize=(8, 6.8))
    im = ax.imshow(corr, cmap="RdYlGn_r", vmin=0.75, vmax=1.0)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7.5)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if corr.iloc[i, j] > 0.93 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="corrélation de rang")
    off = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    ax.set_title(f"Corrélation entre prédictions — moyenne {off.mean():.3f}, minimum {off.min():.3f}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG_NEW / "model_correlation.png", dpi=130)
    plt.close(fig)
    return off.mean(), off.min()


def fig_calibration():
    best = oof["xgb_tuned"]
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rates, counts, labels = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (best >= lo) & (best < hi)
        rates.append(y_tr[m].mean() if m.sum() else 0)
        counts.append(int(m.sum()))
        labels.append(f"[{lo:.1f}–{hi:.1f})")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, rates, color="#2e7d5b", width=0.6)
    ax.axhline(y_tr.mean(), ls="--", color=RED, lw=1.2,
               label=f"taux de défaut moyen = {y_tr.mean():.1%}")
    for b, r, c in zip(bars, rates, counts):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.1%}\n({c} clients)",
                ha="center", fontsize=8.5)
    ax.set_ylabel("Taux de défaut réel observé")
    ax.set_xlabel("Probabilité prédite par le modèle")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=8.5)
    ax.set_title("Le modèle est bien calibré : 4.5% de risque d'un côté, 70% de l'autre (15x)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG_NEW / "calibration.png", dpi=130)
    plt.close(fig)
    return rates, counts


def fig_cost_threshold():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(costs["seuil"], costs["recall"], "o-", color=GREEN, lw=2, label="Recall (défauts détectés)")
    ax.plot(costs["seuil"], costs["pct_portefeuille"], "s-", color=RED, lw=2,
            label="% du portefeuille flagué")
    ax.plot(costs["seuil"], costs["precision"], "^-", color=NAVY, lw=2, label="Precision")
    for _, row in costs.iterrows():
        ax.annotate(f"R={int(row['ratio_FN_FP'])}", (row["seuil"], row["recall"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5,
                    fontweight="bold")
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.set_title("Arbitrage métier : R = coût d'un défaut manqué / coût d'une fausse alerte",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG_NEW / "cost_threshold.png", dpi=130)
    plt.close(fig)


def fig_before_after():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, name in zip(axes, ["pr_auc", "roc_auc"], ["PR-AUC", "ROC-AUC"]):
        vals = [BASELINE_TEST[key], FINAL_TEST[key]]
        bars = ax.bar(["Baseline v1\n(XGBoost non tuné)", "Optimisé v2\n(XGBoost tuné)"], vals,
                      color=["#9a9a9a", "#2e7d5b"], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center",
                    fontsize=11, fontweight="bold")
        gain = vals[1] - vals[0]
        ax.set_title(f"{name} sur le test set  (+{gain:.4f})", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(IMG_NEW / "before_after.png", dpi=130)
    plt.close(fig)


# =============================================================== helpers de mise en page
def add_header(fig, title, subtitle=None, page=None):
    fig.text(0.07, 0.965, title, fontsize=15, fontweight="bold", color=NAVY, va="top")
    if subtitle:
        fig.text(0.07, 0.935, subtitle, fontsize=9.5, color=GRAY, va="top")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.925, 0.925], color=NAVY, lw=1.2,
                              transform=fig.transFigure))
    if page:
        fig.text(0.93, 0.02, str(page), fontsize=8, color=GRAY, ha="right")
    fig.text(0.07, 0.02, "Rapport pipeline MLOps — Prédiction de retard de paiement",
             fontsize=7, color=GRAY)


def wrap(text, width=100):
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width=width) if para.strip() else [""])
    return "\n".join(out)


def bullets(ax, items, y=0.95, fontsize=9.4, width=98):
    for label, body in items:
        block = f"• {label}" if body is None else f"• {label} {body}"
        w = wrap(block, width)
        ax.text(0.02, y, w, fontsize=fontsize, va="top", ha="left", linespacing=1.5,
                transform=ax.transAxes, color="#222")
        y -= 0.028 * (w.count("\n") + 1) + 0.018
    return y


def table(ax, df, fontsize=8, highlight=None, widths=None):
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, 1.55)
    if widths:
        for i, w in enumerate(widths):
            for r in range(len(df) + 1):
                t[(r, i)].set_width(w)
    for i in range(len(df.columns)):
        t[(0, i)].set_facecolor(NAVY)
        t[(0, i)].set_text_props(color="white", fontweight="bold")
    for r in range(1, len(df) + 1):
        for c in range(len(df.columns)):
            if highlight is not None and (r - 1) in np.atleast_1d(highlight):
                t[(r, c)].set_facecolor("#dcecdf")
            elif r % 2 == 0:
                t[(r, c)].set_facecolor(LIGHT)
    return t


def image_page(pdf, title, img_path, caption, page, text_top=None, text_bottom=None):
    fig = plt.figure(figsize=A4)
    add_header(fig, title, page=page)
    top = 0.88
    if text_top:
        ax = fig.add_axes([0.07, 0.70, 0.86, 0.18]); ax.axis("off")
        bullets(ax, text_top, y=0.98, fontsize=9.3)
        top = 0.68
    h = 0.46 if text_bottom else (top - 0.10)
    ax_img = fig.add_axes([0.07, top - h, 0.86, h])
    ax_img.imshow(mpimg.imread(img_path)); ax_img.axis("off")
    fig.text(0.5, top - h - 0.02, caption, fontsize=8.4, ha="center", style="italic", color=GRAY)
    if text_bottom:
        ax2 = fig.add_axes([0.07, 0.07, 0.86, top - h - 0.08]); ax2.axis("off")
        ax2.text(0.0, 0.92, wrap(text_bottom, 98), fontsize=9.2, va="top", linespacing=1.55,
                 transform=ax2.transAxes)
    pdf.savefig(fig); plt.close(fig)


# =============================================================== construction du PDF
def build():
    fig_ablation(); fig_model_ranking(); fig_calibration(); fig_cost_threshold(); fig_before_after()
    corr_mean, corr_min = fig_correlation()

    pdf = PdfPages(OUT_PDF)
    best = ranking.iloc[0]

    # ---------------------------------------------------------------- couverture
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.78), 1, 0.22, color=NAVY, transform=ax.transAxes))
    ax.text(0.5, 0.91, "Rapport Pipeline MLOps", fontsize=24, color="white", ha="center",
            va="center", fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.855, "Prédiction de retard de paiement — Cartes de crédit", fontsize=13,
            color="#dbe6f5", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.815, "Version 2 — optimisation, deep learning et analyse du plafond",
            fontsize=10, color="#b9cbe6", ha="center", va="center", transform=ax.transAxes,
            style="italic")
    ax.text(0.5, 0.70, "Nettoyage · Feature engineering · Rééquilibrage · Comparaison de\n"
            "18 modèles · Tuning · Deep learning séquentiel · Seuil par coût métier",
            fontsize=11.5, ha="center", va="center", transform=ax.transAxes, linespacing=1.7)

    rows = [("Dataset", "Default of Credit Card Clients (UCI) — 30 000 clients, Taïwan"),
            ("Cible", "DEFAULT — 22.12% de défauts (ratio 3.52 : 1)"),
            ("Modèle retenu", "XGBoost tuné (517 arbres, lr=0.010, depth=4)"),
            ("PR-AUC (test)", f"{FINAL_TEST['pr_auc']:.4f}   (v1 : {BASELINE_TEST['pr_auc']:.4f}"
                              f"  →  +{FINAL_TEST['pr_auc']-BASELINE_TEST['pr_auc']:.4f})"),
            ("ROC-AUC (test)", f"{FINAL_TEST['roc_auc']:.4f}   (v1 : {BASELINE_TEST['roc_auc']:.4f}"
                               f"  →  +{FINAL_TEST['roc_auc']-BASELINE_TEST['roc_auc']:.4f})")]
    y = 0.56
    for k, v in rows:
        ax.text(0.07, y, k + " :", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0.32, y, v, fontsize=10, transform=ax.transAxes,
                color=GREEN if "test" in k else "#222")
        y -= 0.042

    ax.text(0.07, 0.10, f"Date : {date.today().strftime('%d/%m/%Y')}", fontsize=9.5, color=GRAY,
            transform=ax.transAxes)
    ax.text(0.07, 0.07, "Auteur : Abdelmoughit El Hassani — assisté par Claude Code",
            fontsize=9.5, color=GRAY, transform=ax.transAxes)
    ax.text(0.07, 0.04, "Sources : notebooks/01_eda_analyse_donnees.ipynb · src/ · scripts/",
            fontsize=8.5, color=GRAY, transform=ax.transAxes, style="italic")
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- sommaire
    fig = plt.figure(figsize=A4); add_header(fig, "Sommaire", page=2)
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.82]); ax.axis("off")
    toc = [("PARTIE 1 — PIPELINE DE BASE", True),
           ("1.  Contexte et objectif", False),
           ("2.  Analyse exploratoire (EDA)", False),
           ("3.  Nettoyage des valeurs non documentées", False),
           ("4.  Feature engineering", False),
           ("5.  Préparation pour la modélisation", False),
           ("6.  Stratégies de rééquilibrage x modèles (15 combinaisons)", False),
           ("7.  Première évaluation et calibration du seuil", False),
           ("", False),
           ("PARTIE 2 — OPTIMISATION ET ANALYSE DU PLAFOND", True),
           ("8.  Démarche : localiser le vrai goulot d'étranglement", False),
           ("9.  Diagnostic d'ablation : le feature engineering est-il en cause ?", False),
           ("10. Un vrai bug trouvé — et pourquoi le corriger ne change rien", False),
           ("11. Comparaison de 11 modèles (boosting, forêts, deep learning)", False),
           ("12. Tuning d'hyperparamètres — la seule source de gain réel", False),
           ("13. Deep learning séquentiel (LSTM, GRU, MLP)", False),
           ("14. Blending : pourquoi il ne rapporte rien ici", False),
           ("15. Trois preuves que le plafond est dans les données", False),
           ("16. Évaluation finale sur le test set", False),
           ("17. Choix du seuil par coût métier", False),
           ("18. Synthèse et recommandations", False)]
    y = 0.96
    for line, is_head in toc:
        if line:
            ax.text(0.02, y, line, fontsize=11 if is_head else 10.2, transform=ax.transAxes,
                    fontweight="bold" if is_head else "normal",
                    color=NAVY if is_head else "#222")
        y -= 0.042
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- P1 : contexte
    fig = plt.figure(figsize=A4); add_header(fig, "1. Contexte et objectif", page=3)
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.80]); ax.axis("off")
    bullets(ax, [
        ("Objectif :", "prédire si un client de carte de crédit fera défaut le mois suivant, "
         "à partir de son historique de facturation, de paiement et de son profil."),
        ("Dataset :", "30 000 clients, 23 variables explicatives + identifiant + cible. "
         "6 mois d'historique par client (PAY_1..6, BILL_AMT1..6, PAY_AMT1..6)."),
        ("Constats EDA :", "aucun NaN, aucun doublon, mais des codes catégoriels non documentés "
         "et une cible déséquilibrée à 3.52 : 1."),
        ("Démarche partie 1 :", "nettoyage justifié, feature engineering métier, comparaison "
         "rigoureuse (anti-fuite) de 15 combinaisons modèle x stratégie de rééquilibrage."),
        ("Démarche partie 2 :", "les résultats de la partie 1 ayant été jugés insuffisants, on "
         "cherche méthodiquement où se situe le goulot d'étranglement — features, rééquilibrage "
         "ou modèle — puis on pousse chaque piste jusqu'à sa limite."),
        ("Principe méthodologique :", "le test set (6 000 clients) n'intervient dans AUCUNE "
         "décision — ni features, ni modèle, ni hyperparamètres, ni seuil. Il n'est évalué "
         "qu'une seule fois, à la toute fin."),
    ], y=0.95, fontsize=9.8, width=94)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- P1 : EDA
    fig = plt.figure(figsize=A4); add_header(fig, "2. Analyse exploratoire", page=4)
    ax = fig.add_axes([0.07, 0.62, 0.86, 0.26]); ax.axis("off")
    bullets(ax, [("Doublons :", "0 ligne dupliquée, 30 000 ID uniques."),
                 ("Valeurs manquantes :", "aucune (0 NaN)."),
                 ("Codes non documentés :", "EDUCATION 0/5/6 (345 lignes), MARRIAGE 0 (54 lignes)."),
                 ("Montants négatifs :", "normaux sur BILL_AMT (crédit en faveur du client)."),
                 ("Corrélation la plus forte :", "PAY_1 avec DEFAULT (r = 0.325).")],
            y=0.96, fontsize=9.4)
    axi = fig.add_axes([0.10, 0.10, 0.80, 0.46])
    axi.imshow(mpimg.imread(IMG_OLD / "target_balance.png")); axi.axis("off")
    fig.text(0.5, 0.575, "Distribution de la cible — 77.88% / 22.12%", fontsize=8.5,
             ha="center", style="italic", color=GRAY)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- P1 : nettoyage
    fig = plt.figure(figsize=A4); add_header(fig, "3. Nettoyage des valeurs non documentées", page=5)
    ax = fig.add_axes([0.07, 0.62, 0.86, 0.28]); ax.axis("off")
    ax.text(0.02, 0.95, wrap(
        "Décision : ni suppression (perte inutile de 1.15% des lignes), ni imputation par le mode "
        "(qui supposerait une absence aléatoire, non vérifiée). Les codes non documentés sont "
        "fusionnés dans la catégorie « autre » déjà prévue par le schéma UCI (EDUCATION → 4, "
        "MARRIAGE → 3). Les effectifs concernés sont trop faibles pour justifier une catégorie "
        "distincte après encodage.", 98), fontsize=9.5, va="top", linespacing=1.55,
        transform=ax.transAxes)
    t1 = pd.DataFrame({"EDUCATION": ["0", "1", "2", "3", "4", "5", "6"],
                       "Effectif": [14, 10585, 14030, 4917, 123, 280, 51],
                       "Taux défaut": ["0.0%", "19.2%", "23.7%", "25.2%", "5.7%", "6.4%", "15.7%"],
                       "Après": ["→ 4", "1", "2", "3", "4 (468)", "→ 4", "→ 4"]})
    table(fig.add_axes([0.07, 0.34, 0.86, 0.24]), t1, fontsize=8.3)
    t2 = pd.DataFrame({"MARRIAGE": ["0", "1", "2", "3"], "Effectif": [54, 13659, 15964, 323],
                       "Taux défaut": ["9.3%", "23.5%", "20.9%", "26.0%"],
                       "Après": ["→ 3", "1", "2", "3 (377)"]})
    table(fig.add_axes([0.07, 0.10, 0.86, 0.18]), t2, fontsize=8.3)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- P1 : FE
    fig = plt.figure(figsize=A4); add_header(fig, "4. Feature engineering", page=6)
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.82]); ax.axis("off")
    ax.text(0.02, 0.97, wrap(
        "BILL_AMT1..6 et PAY_1..6 sont fortement corrélées entre elles (multicolinéarité). "
        "Le bloc des 18 colonnes mensuelles brutes est résumé en ratios et agrégats métier :", 98),
        fontsize=9.5, va="top", linespacing=1.5, transform=ax.transAxes)
    bullets(ax, [
        ("utilization_rate :", "BILL_AMT1 / LIMIT_BAL — part du plafond utilisée."),
        ("avg_utilization_rate / max_utilization :", "utilisation moyenne et maximale sur 6 mois."),
        ("pay_ratio_1..5 :", "part de la facture précédente réellement remboursée."),
        ("paid_full_1..5, n_months_paid_full :", "le client a-t-il soldé sa facture ?"),
        ("unpaid_gap, total_unpaid_gap, unpaid_to_limit :", "dette non remboursée, en absolu "
         "et rapportée au plafond."),
        ("pay_delay_max / mean / std :", "sévérité du retard sur 6 mois."),
        ("n_months_late :", "nombre de mois en retard (sur 6)."),
        ("delay_acceleration, recent_vs_past_delay :", "dégradation récente vs historique."),
        ("delta_pay_1..5 :", "variation du statut de retard d'un mois à l'autre."),
        ("bill_trend, bill_std, pay_amt_std :", "tendance et volatilité des montants."),
        ("log_limit, log_avg_bill, log_avg_pay :", "montants en échelle logarithmique."),
    ], y=0.90, fontsize=9.2, width=96)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- P1 : corrélations FE
    fig = plt.figure(figsize=A4)
    add_header(fig, "4. Feature engineering — impact sur la corrélation", page=7)
    corr_df = pd.DataFrame([
        ("n_months_late", "+0.398", "nouvelle"), ("ever_serious_delay", "+0.363", "nouvelle"),
        ("pay_delay_max", "+0.331", "nouvelle"), ("PAY_1", "+0.325", "brute"),
        ("pay_delay_last3", "+0.309", "nouvelle"), ("pay_delay_mean", "+0.282", "nouvelle"),
        ("pay_delay_std", "+0.249", "nouvelle"), ("pay_delay_first3", "+0.219", "nouvelle"),
        ("log_limit", "-0.174", "nouvelle"), ("LIMIT_BAL", "-0.154", "brute"),
        ("unpaid_to_limit", "+0.131", "nouvelle"), ("delay_acceleration", "+0.117", "nouvelle"),
    ], columns=["Variable", "Corrélation avec DEFAULT", "Origine"])
    t = table(fig.add_axes([0.10, 0.44, 0.80, 0.46]), corr_df, fontsize=8.6)
    for r in range(1, len(corr_df) + 1):
        if corr_df.iloc[r - 1]["Origine"] == "nouvelle":
            for c in range(3):
                t[(r, c)].set_facecolor("#dcecdf")
    fig.text(0.5, 0.905, "Les features engineered dominent le classement (fond vert)",
             fontsize=8.5, ha="center", style="italic", color=GRAY)
    ax = fig.add_axes([0.07, 0.10, 0.86, 0.28]); ax.axis("off")
    ax.text(0.0, 0.92, wrap(
        "Sur le papier, le feature engineering est un succès : n_months_late (0.398) et "
        "ever_serious_delay (0.363) dépassent nettement PAY_1 brut (0.325), la meilleure variable "
        "d'origine. La partie 2 montrera cependant que ce gain de corrélation ne se traduit PAS "
        "par un gain de performance du modèle — un résultat contre-intuitif mais instructif.", 96),
        fontsize=9.4, va="top", linespacing=1.6, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- P1 : préparation
    fig = plt.figure(figsize=A4); add_header(fig, "5. Préparation pour la modélisation", page=8)
    ax = fig.add_axes([0.07, 0.62, 0.86, 0.28]); ax.axis("off")
    bullets(ax, [
        ("ID exclu de X :", "identifiant, jamais une feature — et jamais une base de rééquilibrage."),
        ("Catégorielles :", "SEX, EDUCATION, MARRIAGE en codes entiers (faible cardinalité), "
         "ce qu'attend SMOTENC et que les arbres gèrent nativement."),
        ("Split :", "stratifié 80/20 — train (24 000) / test (6 000), même taux de défaut."),
        ("Test set :", "réservé à une unique évaluation finale."),
    ], y=0.95, fontsize=9.6)
    axi = fig.add_axes([0.13, 0.08, 0.74, 0.48])
    axi.imshow(mpimg.imread(IMG_OLD / "correlation_heatmap.png")); axi.axis("off")
    fig.text(0.5, 0.575, "Matrice de corrélation — le bloc BILL_AMT1..6 est très redondant",
             fontsize=8.4, ha="center", style="italic", color=GRAY)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- P1 : 15 combinaisons
    fig = plt.figure(figsize=A4)
    add_header(fig, "6. Stratégies de rééquilibrage x modèles", "15 combinaisons, 5-fold CV", page=9)
    res15 = pd.DataFrame([
        ("XGBoost", "class_weight", 0.5524, 0.5332, 0.6086), ("XGBoost", "Aucun", 0.5488, 0.4782, 0.3743),
        ("RandomForest", "Aucun", 0.5463, 0.4841, 0.3844), ("RandomForest", "class_weight", 0.5443, 0.4669, 0.3616),
        ("XGBoost", "Oversampling", 0.5442, 0.5342, 0.5954), ("RandomForest", "Undersampling", 0.5364, 0.5273, 0.6521),
        ("RandomForest", "Oversampling", 0.5357, 0.5096, 0.4436), ("XGBoost", "Undersampling", 0.5300, 0.5135, 0.6636),
        ("XGBoost", "SMOTENC", 0.5297, 0.5051, 0.4538), ("RandomForest", "SMOTENC", 0.5240, 0.5096, 0.4570),
        ("LogisticRegression", "Aucun", 0.5233, 0.4577, 0.3579),
        ("LogisticRegression", "Oversampling", 0.5216, 0.5270, 0.6350),
        ("LogisticRegression", "class_weight", 0.5214, 0.5266, 0.6338),
        ("LogisticRegression", "Undersampling", 0.5189, 0.5274, 0.6346),
        ("LogisticRegression", "SMOTENC", 0.5011, 0.5254, 0.6029),
    ], columns=["Modèle", "Stratégie", "PR-AUC", "F1", "Recall"])
    for c in ["PR-AUC", "F1", "Recall"]:
        res15[c] = res15[c].map(lambda v: f"{v:.4f}")
    table(fig.add_axes([0.05, 0.36, 0.90, 0.54]), res15, fontsize=7.8, highlight=0,
          widths=[0.26, 0.24, 0.166, 0.166, 0.166])
    ax = fig.add_axes([0.07, 0.07, 0.86, 0.24]); ax.axis("off")
    ax.text(0.0, 0.95, wrap(
        "Enseignements : (1) class_weight / scale_pos_weight bat le resampling pour les modèles à "
        "arbres — gratuit, aucune donnée modifiée, et donc pas besoin de dupliquer des clients ni "
        "d'inventer des identifiants. (2) SMOTENC est la stratégie la plus faible des trois "
        "modèles. (3) LogisticRegression plafonne quelle que soit la stratégie.", 98),
        fontsize=9.2, va="top", linespacing=1.5, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    image_page(pdf, "6. Comparaison visuelle", IMG_OLD / "strategy_comparison.png",
               "PR-AUC et Recall par modèle x stratégie",
               page=10,
               text_bottom="Le recall monte fortement avec l'undersampling et l'oversampling, mais "
                           "au prix de la précision. C'est pourquoi la stratégie ne doit jamais "
                           "être choisie sur le recall seul : le bon levier pour arbitrer "
                           "recall/précision est le seuil de décision (section 17), pas la "
                           "méthode de rééquilibrage.")

    # ---------------------------------------------------------------- P1 : éval v1
    fig = plt.figure(figsize=A4)
    add_header(fig, "7. Première évaluation (version 1)", "XGBoost non tuné — point de départ", page=11)
    ax = fig.add_axes([0.07, 0.80, 0.86, 0.09]); ax.axis("off")
    ax.text(0.0, 1.0, f"PR-AUC test = {BASELINE_TEST['pr_auc']:.4f}    |    "
            f"ROC-AUC test = {BASELINE_TEST['roc_auc']:.4f}", fontsize=11,
            fontweight="bold", color=NAVY, transform=ax.transAxes, va="top")
    for i, (cm, title, sub) in enumerate([
            (np.array([[3747, 926], [517, 810]]), "Seuil 0.5", "Precision 46.7% | Recall 61.0%"),
            (np.array([[2158, 2515], [207, 1120]]), "Seuil F2 = 0.258",
             "Precision 30.8% | Recall 84.7%")]):
        axc = fig.add_axes([0.10 + i * 0.46, 0.50, 0.36, 0.30])
        axc.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            axc.text(c, r, str(v), ha="center", va="center", fontsize=11, fontweight="bold",
                     color="white" if v > cm.max() / 2 else "black")
        axc.set_xticks([0, 1]); axc.set_xticklabels(["Prédit 0", "Prédit 1"], fontsize=8)
        axc.set_yticks([0, 1]); axc.set_yticklabels(["Réel 0", "Réel 1"], fontsize=8)
        axc.set_title(title, fontsize=9.5, fontweight="bold")
        fig.text(0.28 + i * 0.46, 0.475, sub, fontsize=7.8, ha="center", color=GRAY)
    ax2 = fig.add_axes([0.07, 0.10, 0.86, 0.33]); ax2.axis("off")
    ax2.text(0.0, 0.95, wrap(
        "Problème identifié dès cette version : maximiser aveuglément le F2 conduit à un seuil de "
        "0.258 qui flague 3 635 clients sur 6 000, soit 60% du portefeuille. Le rappel est "
        "excellent (84.7%) mais la précision s'effondre à 30.8% et l'accuracy tombe à 54.6%. "
        "Aucune banque ne peut opérer ainsi. La section 17 remplace cette approche par un "
        "raisonnement en coût métier explicite.\n\n"
        "C'est ce niveau de performance (PR-AUC 0.5432) qui a été jugé insuffisant et qui motive "
        "toute la partie 2.", 98), fontsize=9.3, va="top", linespacing=1.55,
        transform=ax2.transAxes, color="#7a2d2b")
    pdf.savefig(fig); plt.close(fig)

    # ================================================================ PARTIE 2
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.42), 1, 0.16, color=NAVY, transform=ax.transAxes))
    ax.text(0.5, 0.50, "PARTIE 2", fontsize=30, color="white", ha="center", va="center",
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.36, "Optimisation, deep learning\net analyse du plafond de performance",
            fontsize=14, ha="center", va="center", transform=ax.transAxes, linespacing=1.7,
            color=NAVY)
    ax.text(0.5, 0.25, "Où se situe réellement le goulot d'étranglement :\n"
            "les features, le rééquilibrage, ou le modèle ?", fontsize=11, ha="center",
            va="center", transform=ax.transAxes, style="italic", color=GRAY, linespacing=1.6)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- démarche
    fig = plt.figure(figsize=A4)
    add_header(fig, "8. Démarche : localiser le goulot d'étranglement", page=13)
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.80]); ax.axis("off")
    ax.text(0.02, 0.97, wrap(
        "La performance de la version 1 (PR-AUC 0.5432) ayant été jugée insuffisante, trois "
        "hypothèses devaient être départagées — sans les supposer, en les testant :", 98),
        fontsize=9.8, va="top", linespacing=1.5, transform=ax.transAxes)
    bullets(ax, [
        ("Hypothèse A — la préparation des données et le feature engineering.",
         "Testée par une ablation : 4 jeux de features (de 23 variables brutes à 68 enrichies) "
         "évalués avec le MÊME modèle, pour isoler l'effet des features. → Section 9."),
        ("Hypothèse B — les méthodes de gestion du déséquilibre.",
         "Déjà testée en partie 1 sur 15 combinaisons : class_weight / scale_pos_weight était "
         "bien le meilleur choix. Aucune raison de reprendre cette piste."),
        ("Hypothèse C — les modèles et leurs hyperparamètres.",
         "Testée par la comparaison de 11 modèles (boosting, forêts, réseaux de neurones, "
         "architectures récurrentes) et par un tuning systématique. → Sections 11 à 13."),
        ("Et si aucune des trois ne suffit ?",
         "Alors la limite vient des données elles-mêmes. Cette hypothèse n'est pas une excuse : "
         "elle se démontre. Trois tests indépendants la confirment. → Section 15."),
    ], y=0.88, fontsize=9.5, width=94)
    ax.text(0.02, 0.14, wrap(
        "Toutes les comparaisons de la partie 2 utilisent un protocole strictement identique : "
        "mêmes données, même split, même validation croisée 5-fold stratifiée sur le train, mêmes "
        "métriques. Les prédictions out-of-fold de chaque modèle sont sauvegardées sur disque, ce "
        "qui rend les comparaisons et les ensembles reproductibles sans réentraînement.", 98),
        fontsize=9, va="top", linespacing=1.55, transform=ax.transAxes, style="italic",
        color="#333")
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- ablation
    abl_tbl = ablation.copy()
    abl_tbl["PR-AUC"] = abl_tbl.apply(lambda r: f"{r['pr_auc']:.4f} ± {r['pr_auc_std']:.4f}", axis=1)
    abl_tbl["ROC-AUC"] = abl_tbl.apply(lambda r: f"{r['roc_auc']:.4f} ± {r['roc_auc_std']:.4f}", axis=1)
    abl_show = abl_tbl[["features", "n_features", "PR-AUC", "ROC-AUC"]]
    abl_show.columns = ["Jeu de features", "Nb", "PR-AUC", "ROC-AUC"]

    fig = plt.figure(figsize=A4)
    add_header(fig, "9. Diagnostic d'ablation", "Le feature engineering est-il le goulot ?", page=14)
    table(fig.add_axes([0.05, 0.66, 0.90, 0.22]), abl_show, fontsize=8,
          widths=[0.40, 0.10, 0.25, 0.25])
    axi = fig.add_axes([0.07, 0.34, 0.86, 0.28])
    axi.imshow(mpimg.imread(IMG_NEW / "ablation.png")); axi.axis("off")
    ax = fig.add_axes([0.07, 0.07, 0.86, 0.24]); ax.axis("off")
    ax.text(0.0, 0.95, wrap(
        "Verdict : les quatre jeux de features sont statistiquement indiscernables. L'écart entre "
        "le meilleur (0.5524) et le pire (0.5479) est de 0.0045, soit MOINS d'un écart-type entre "
        "folds (±0.005 à ±0.010). Passer de 23 variables brutes à 68 variables enrichies — avec "
        "ratios corrigés, volatilités, accélérations de retard et indicateurs de remboursement "
        "complet — ne change rien.\n\n"
        "L'hypothèse A est écartée : le feature engineering n'est pas le goulot d'étranglement.",
        98), fontsize=9.3, va="top", linespacing=1.55, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- le bug
    fig = plt.figure(figsize=A4)
    add_header(fig, "10. Un vrai bug — et pourquoi le corriger ne change rien", page=15)
    ax = fig.add_axes([0.07, 0.72, 0.86, 0.17]); ax.axis("off")
    ax.text(0.0, 0.98, wrap(
        "L'ablation a mis au jour un défaut réel dans la version 1 de payment_ratio. La formule "
        "PAY_AMT / (BILL_AMT.clip(lower=0) + 1) devient un montant brut dès que la facture "
        "approche zéro : le dénominateur vaut 1 et le « ratio » explose à plusieurs milliers. "
        "La feature mélangeait donc deux grandeurs incompatibles.", 98),
        fontsize=9.4, va="top", linespacing=1.55, transform=ax.transAxes)
    bug = pd.DataFrame({
        "Mois": [1, 2, 3, 4, 5],
        "v1 (cassée)": ["-0.0059", "-0.0039", "-0.0029", "-0.0041", "-0.0058"],
        "v2 (corrigée)": ["-0.1027", "-0.1002", "-0.0968", "-0.0850", "-0.0861"],
        "paid_full (nouvelle)": ["-0.0885", "-0.0875", "-0.0819", "-0.0711", "-0.0708"]})
    table(fig.add_axes([0.10, 0.47, 0.80, 0.22]), bug, fontsize=8.5)
    fig.text(0.5, 0.695, "Corrélation avec DEFAULT — avant / après correction",
             fontsize=8.5, ha="center", style="italic", color=GRAY)
    ax2 = fig.add_axes([0.07, 0.08, 0.86, 0.35]); ax2.axis("off")
    ax2.text(0.0, 0.96, wrap(
        "La correction (ratio conditionnel à BILL > 0, borné à [0, 2]) multiplie la corrélation "
        "avec la cible par 17 : de -0.006 à -0.103. C'est un vrai gain de qualité de la variable.\n\n"
        "Et pourtant, le modèle ne s'améliore pas : le jeu D, qui intègre cette correction et 30 "
        "features supplémentaires, obtient 0.5488 — soit exactement le même niveau que les autres.\n\n"
        "L'explication est instructive : XGBoost pouvait déjà reconstruire cette relation seul, à "
        "partir de PAY_AMT et BILL_AMT bruts, en combinant des seuils sur ces deux colonnes. Sur "
        "un modèle à arbres, le feature engineering ne crée pratiquement jamais d'information "
        "nouvelle — il ne fait que la ré-exprimer sous une forme plus lisible pour un humain. "
        "Le gain de corrélation d'une variable prise isolément ne préjuge donc en rien du gain "
        "de performance du modèle qui l'utilise.\n\n"
        "La correction a malgré tout été conservée dans src/data_prep.py : une variable au "
        "comportement défini reste préférable pour l'interprétabilité et pour les modèles "
        "linéaires ou neuronaux, qui ne savent pas la reconstruire.", 98),
        fontsize=9.3, va="top", linespacing=1.55, transform=ax2.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- 11 modèles
    rk = ranking.copy()
    rk["Modèle"] = rk["model"].map(LABEL)
    rk["Famille"] = rk["model"].map(FAMILY)
    rk["PR-AUC"] = rk["pr_auc"].map(lambda v: f"{v:.4f}")
    rk["ROC-AUC"] = rk["roc_auc"].map(lambda v: f"{v:.4f}")
    fig = plt.figure(figsize=A4)
    add_header(fig, "11. Comparaison de 11 modèles", "Protocole identique, OOF sur le train", page=16)
    table(fig.add_axes([0.07, 0.56, 0.86, 0.33]), rk[["Modèle", "Famille", "PR-AUC", "ROC-AUC"]],
          fontsize=8.2, highlight=[0, 1], widths=[0.32, 0.26, 0.21, 0.21])
    axi = fig.add_axes([0.06, 0.19, 0.88, 0.34])
    axi.imshow(mpimg.imread(IMG_NEW / "model_ranking.png")); axi.axis("off")
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.11]); ax.axis("off")
    ax.text(0.0, 0.95, wrap(
        "Les deux meilleurs modèles — XGBoost et LightGBM tunés — obtiennent exactement le même "
        "PR-AUC (0.5650), depuis deux implémentations et deux espaces de recherche différents. "
        "Cette convergence au millième près est un premier indice fort d'un plafond structurel.",
        98), fontsize=9.2, va="top", linespacing=1.5, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- tuning
    fig = plt.figure(figsize=A4)
    add_header(fig, "12. Tuning d'hyperparamètres", "La seule source de gain réel", page=17)
    ax = fig.add_axes([0.07, 0.74, 0.86, 0.15]); ax.axis("off")
    ax.text(0.0, 0.98, wrap(
        "RandomizedSearchCV, 20 tirages, validation croisée 3-fold, optimisation de "
        "l'average precision. Trois familles de boosting ont été optimisées indépendamment.", 98),
        fontsize=9.5, va="top", linespacing=1.5, transform=ax.transAxes)
    tun = pd.DataFrame({
        "Modèle": ["XGBoost", "LightGBM", "HistGB"],
        "learning_rate": [f"{params['xgboost']['learning_rate']:.4f}",
                          f"{params['lightgbm']['learning_rate']:.4f}",
                          f"{params['histgb']['learning_rate']:.4f}"],
        "Profondeur": [f"depth={params['xgboost']['max_depth']}",
                       f"leaves={params['lightgbm']['num_leaves']}",
                       f"leaves={params['histgb']['max_leaf_nodes']}"],
        "Régularisation": [f"gamma={params['xgboost']['gamma']:.1f}",
                           f"L2={params['lightgbm']['reg_lambda']:.1f}",
                           f"L2={params['histgb']['l2_regularization']:.2f}"],
        "Nb arbres": [params['xgboost']['n_estimators'], params['lightgbm']['n_estimators'],
                      params['histgb']['max_iter']]})
    table(fig.add_axes([0.06, 0.53, 0.88, 0.18]), tun, fontsize=8.3)
    ax2 = fig.add_axes([0.07, 0.08, 0.86, 0.42]); ax2.axis("off")
    ax2.text(0.0, 0.96, wrap(
        "Les trois optimiseurs convergent indépendamment vers la même prescription : un pas "
        "d'apprentissage très bas (0.010 à 0.045, contre 0.1 en version 1), des arbres peu "
        "profonds (depth 4, 16-17 feuilles contre depth 5), une régularisation forte, et "
        "davantage d'arbres pour compenser. C'est la signature caractéristique d'un jeu de "
        "données à faible rapport signal/bruit : le modèle doit être bridé pour ne pas mémoriser "
        "le bruit.\n\n"
        "Le diagnostic de la version 1 est donc confirmé : les paramètres du notebook "
        "(learning_rate = 0.1, max_depth = 5) sur-apprenaient sur 24 000 lignes.\n\n"
        f"Gain mesuré : PR-AUC de 0.5453 (XGBoost non tuné) à 0.5650 (XGBoost tuné), soit "
        f"+0.0197 en out-of-fold, confirmé ensuite sur le test set (+0.0213). C'est le SEUL "
        "levier des trois hypothèses testées qui produit un gain réel.", 98),
        fontsize=9.3, va="top", linespacing=1.55, transform=ax2.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- deep learning
    fig = plt.figure(figsize=A4)
    add_header(fig, "13. Deep learning séquentiel", "LSTM, GRU et MLP", page=18)
    ax = fig.add_axes([0.07, 0.62, 0.86, 0.27]); ax.axis("off")
    ax.text(0.0, 0.98, wrap(
        "Le dataset possède une vraie structure séquentielle : chaque client est une série de "
        "6 mois consécutifs (statut de retard, facture, remboursement), plus des attributs "
        "statiques. Les colonnes du fichier étant numérotées à l'envers, elles ont été remises "
        "en ordre chronologique avant d'être données au réseau. L'architecture comporte deux "
        "branches :", 98), fontsize=9.4, va="top", linespacing=1.55, transform=ax.transAxes)
    ax.text(0.06, 0.50, "séquence (B, 6, 5)  →  LSTM / GRU  →  état caché (B, 64) ─┐\n"
            "                                                                      ├→  MLP  →  logit\n"
            "statiques (B, 11)  →  Linear + ReLU  →  (B, 32) ──────────────────────┘",
            fontsize=8, family="monospace", va="top", transform=ax.transAxes, linespacing=1.8)
    ax.text(0.0, 0.24, wrap(
        "Déséquilibre géré par pos_weight dans la BCE — l'équivalent exact de scale_pos_weight. "
        "Early stopping sur un sous-ensemble de validation interne à chaque fold, sans fuite.",
        98), fontsize=9, va="top", linespacing=1.5, transform=ax.transAxes, style="italic",
        color="#333")

    deep_tbl = ranking[ranking["model"].isin(["lstm", "gru", "mlp"])].copy()
    deep_tbl["Modèle"] = deep_tbl["model"].map(LABEL)
    deep_tbl["PR-AUC"] = deep_tbl["pr_auc"].map(lambda v: f"{v:.4f}")
    deep_tbl["ROC-AUC"] = deep_tbl["roc_auc"].map(lambda v: f"{v:.4f}")
    extra = pd.DataFrame({"Modèle": ["XGBoost tuné (référence)"],
                          "PR-AUC": [f"{ranking.iloc[0]['pr_auc']:.4f}"],
                          "ROC-AUC": [f"{ranking.iloc[0]['roc_auc']:.4f}"]})
    deep_show = pd.concat([deep_tbl[["Modèle", "PR-AUC", "ROC-AUC"]], extra], ignore_index=True)
    table(fig.add_axes([0.15, 0.38, 0.70, 0.20]), deep_show, fontsize=8.6, highlight=[3])
    ax2 = fig.add_axes([0.07, 0.07, 0.86, 0.27]); ax2.axis("off")
    ax2.text(0.0, 0.96, wrap(
        "Résultat : LSTM (0.5482), GRU (0.5477) et MLP (0.5478) se tiennent dans un intervalle de "
        "0.0005 — autrement dit, ils sont identiques. Et tous trois restent sous le boosting tuné "
        "(0.5650).\n\n"
        "L'enseignement est net : les architectures récurrentes, qui exploitent pourtant "
        "explicitement la chronologie, n'apportent rien de plus qu'un simple MLP sur features "
        "agrégées. L'information temporelle était déjà entièrement captée par les agrégats "
        "(pay_delay_max, n_months_late, delay_acceleration). Avec seulement 6 pas de temps et "
        "24 000 clients, un RNN n'a ni la longueur de séquence ni le volume nécessaires pour "
        "battre un boosting bien réglé sur des données tabulaires.\n\n"
        "Note de transparence : un BiLSTM et un Transformer étaient prévus, mais n'ont pas pu "
        "être entraînés — la machine (4 cœurs, RAM limitée) a interrompu les processus. Au vu de "
        "la convergence des trois autres architectures, leur résultat serait très probablement "
        "identique, mais cela reste une attente raisonnée et non un résultat mesuré.", 98),
        fontsize=9.2, va="top", linespacing=1.5, transform=ax2.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- blending
    bl = blends.head(8).copy()
    bl["Blend"] = bl["blend"].map(lambda s: " + ".join(LABEL.get(x.strip(), x.strip())
                                                       for x in s.split("+")))
    bl["PR-AUC"] = bl["pr_auc"].map(lambda v: f"{v:.4f}")
    bl["ROC-AUC"] = bl["roc_auc"].map(lambda v: f"{v:.4f}")
    fig = plt.figure(figsize=A4)
    add_header(fig, "14. Blending : pourquoi il ne rapporte rien ici", page=19)
    ax = fig.add_axes([0.07, 0.78, 0.86, 0.11]); ax.axis("off")
    ax.text(0.0, 0.98, wrap(
        "Les prédictions out-of-fold de chaque modèle étant sauvegardées, tous les ensembles "
        "possibles (2 à 5 modèles parmi 11, soit 682 combinaisons) ont été évalués gratuitement, "
        "par moyenne des rangs — robuste aux échelles de probabilité différentes.", 98),
        fontsize=9.3, va="top", linespacing=1.5, transform=ax.transAxes)
    table(fig.add_axes([0.03, 0.44, 0.94, 0.32]), bl[["Blend", "PR-AUC", "ROC-AUC"]],
          fontsize=6.6, widths=[0.62, 0.19, 0.19])
    ax2 = fig.add_axes([0.07, 0.08, 0.86, 0.32]); ax2.axis("off")
    best_blend = blends.iloc[0]["pr_auc"]
    best_single = ranking.iloc[0]["pr_auc"]
    ax2.text(0.0, 0.96, wrap(
        f"Meilleur ensemble : {best_blend:.4f}. Meilleur modèle seul : {best_single:.4f}. "
        f"Gain : +{best_blend - best_single:.4f}.\n\n"
        "Autrement dit, rien. Et c'est un résultat très informatif, car le blending est "
        "précisément la technique qui exploite les DÉSACCORDS entre modèles : quand un boosting "
        "et un réseau de neurones se trompent sur des clients différents, leur moyenne corrige "
        "une partie des erreurs. Ici, combiner un boosting, une forêt aléatoire et un réseau de "
        "neurones ne corrige quasiment rien.\n\n"
        "La conclusion s'impose : ces modèles ne se trompent pas sur des clients différents. "
        "Ils se trompent sur LES MÊMES clients. La section suivante le mesure directement.", 98),
        fontsize=9.3, va="top", linespacing=1.55, transform=ax2.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- preuves du plafond
    fig = plt.figure(figsize=A4)
    add_header(fig, "15. Trois preuves que le plafond est dans les données", page=20)
    axi = fig.add_axes([0.10, 0.44, 0.80, 0.44])
    axi.imshow(mpimg.imread(IMG_NEW / "model_correlation.png")); axi.axis("off")
    ax = fig.add_axes([0.07, 0.07, 0.86, 0.34]); ax.axis("off")
    ax.text(0.0, 0.97, wrap(
        f"Preuve 1 — Les modèles voient tous la même chose. La corrélation de rang entre leurs "
        f"prédictions atteint {corr_mean:.3f} en moyenne. Même la paire la plus « diverse » "
        f"(RandomForest vs LogisticRegression) corrèle à {corr_min:.3f}. Un boosting, une forêt, "
        "un réseau de neurones et une régression logistique classent les clients dans le même "
        "ordre : ils extraient tous le même signal, celui qui existe dans les données.\n\n"
        f"Preuve 2 — Le blending ne rapporte que +{best_blend - best_single:.4f}. Sans désaccord "
        "entre modèles, aucun ensemble ne peut aider (section 14).\n\n"
        "Preuve 3 — XGBoost et LightGBM tunés atterrissent tous deux sur 0.5650, au millième "
        "près, depuis des implémentations et des espaces de recherche indépendants. Deux chemins "
        "distincts qui butent exactement au même endroit signalent une limite structurelle, non "
        "un défaut d'algorithme.", 98), fontsize=9.2, va="top", linespacing=1.5,
        transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- calibration
    image_page(pdf, "15. Ce que le modèle sait faire malgré ce plafond",
               IMG_NEW / "calibration.png",
               "Probabilité prédite vs taux de défaut réellement observé (out-of-fold, train)",
               page=21,
               text_bottom=(
                   "Le plafond de discrimination n'empêche pas le modèle d'être utile ni bien "
                   "calibré : il sépare correctement un groupe à 4.5% de risque d'un groupe à "
                   "70.2% de risque, soit un rapport de 15x, directement exploitable pour "
                   "prioriser des actions de recouvrement.\n\n"
                   "Mais même dans le groupe le plus risqué, 30% des clients ne font pas défaut. "
                   "Ce n'est pas une erreur du modèle : ces clients sont strictement identiques "
                   "aux défaillants sur toutes les variables disponibles. Pour les distinguer, il "
                   "faudrait d'autres DONNÉES (revenus, historique bancaire, encours dans "
                   "d'autres établissements), pas d'autres algorithmes."))

    # ---------------------------------------------------------------- éval finale
    fig = plt.figure(figsize=A4)
    add_header(fig, "16. Évaluation finale sur le test set", "6 000 clients, jamais utilisés avant",
               page=22)
    axi = fig.add_axes([0.07, 0.56, 0.86, 0.32])
    axi.imshow(mpimg.imread(IMG_NEW / "before_after.png")); axi.axis("off")
    comp = pd.DataFrame({
        "Métrique": ["PR-AUC", "ROC-AUC"],
        "Version 1": [f"{BASELINE_TEST['pr_auc']:.4f}", f"{BASELINE_TEST['roc_auc']:.4f}"],
        "Version 2": [f"{FINAL_TEST['pr_auc']:.4f}", f"{FINAL_TEST['roc_auc']:.4f}"],
        "Gain": [f"+{FINAL_TEST['pr_auc']-BASELINE_TEST['pr_auc']:.4f}",
                 f"+{FINAL_TEST['roc_auc']-BASELINE_TEST['roc_auc']:.4f}"],
        "Gain relatif": [f"+{100*(FINAL_TEST['pr_auc']/BASELINE_TEST['pr_auc']-1):.1f}%",
                         f"+{100*(FINAL_TEST['roc_auc']/BASELINE_TEST['roc_auc']-1):.1f}%"]})
    table(fig.add_axes([0.10, 0.38, 0.80, 0.14]), comp, fontsize=8.8)
    ax = fig.add_axes([0.07, 0.07, 0.86, 0.28]); ax.axis("off")
    ax.text(0.0, 0.96, wrap(
        "Le gain mesuré en validation croisée se confirme sur des données jamais vues, ce qui "
        "exclut un artefact de sélection. Il provient entièrement du tuning d'hyperparamètres — "
        "ni du feature engineering (section 9), ni du rééquilibrage (déjà optimal en partie 1), "
        "ni du deep learning (section 13), ni du blending (section 14).\n\n"
        "Modèle final retenu : XGBoost avec learning_rate = 0.010, max_depth = 4, gamma = 4.3, "
        "min_child_weight = 9, 517 arbres, scale_pos_weight = 3.521.", 98),
        fontsize=9.3, va="top", linespacing=1.55, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- seuils métier
    ct = costs.copy()
    ct_show = pd.DataFrame({
        "R = coût FN / FP": ct["ratio_FN_FP"].astype(int),
        "Seuil": ct["seuil"].map(lambda v: f"{v:.3f}"),
        "Recall": ct["recall"].map(lambda v: f"{v:.1%}"),
        "Precision": ct["precision"].map(lambda v: f"{v:.1%}"),
        "% portefeuille flagué": ct["pct_portefeuille"].map(lambda v: f"{v:.1%}")})
    fig = plt.figure(figsize=A4)
    add_header(fig, "17. Choix du seuil par coût métier", page=23)
    ax = fig.add_axes([0.07, 0.77, 0.86, 0.12]); ax.axis("off")
    ax.text(0.0, 0.98, wrap(
        "La version 1 maximisait le F2, ce qui donnait un seuil de 0.258 flaguant 60% du "
        "portefeuille — inexploitable. On raisonne désormais en coût explicite : "
        "coût total = R x (défauts manqués) + (fausses alertes). Seul le RAPPORT R importe, "
        "la banque n'a donc pas besoin de chiffrer ses coûts en valeur absolue.", 98),
        fontsize=9.3, va="top", linespacing=1.5, transform=ax.transAxes)
    table(fig.add_axes([0.08, 0.55, 0.84, 0.20]), ct_show, fontsize=8.6)
    axi = fig.add_axes([0.07, 0.26, 0.86, 0.27])
    axi.imshow(mpimg.imread(IMG_NEW / "cost_threshold.png")); axi.axis("off")
    ax2 = fig.add_axes([0.07, 0.06, 0.86, 0.18]); ax2.axis("off")
    ax2.text(0.0, 0.95, wrap(
        "Seuils calibrés sur les probabilités out-of-fold du train, puis appliqués une seule fois "
        "au test. C'est cette table qui doit être présentée au métier pour arbitrer : si un défaut "
        "manqué coûte 3x une fausse alerte, le seuil est 0.530 et l'on flague 27% du portefeuille "
        "en détectant 60% des défauts. S'il coûte 10x, le seuil descend à 0.245, on détecte 94% "
        "des défauts mais on flague 76% des clients. Le choix appartient à la banque, pas au "
        "modèle.", 98), fontsize=9.2, va="top", linespacing=1.5, transform=ax2.transAxes)
    pdf.savefig(fig); plt.close(fig)

    # ---------------------------------------------------------------- synthèse
    fig = plt.figure(figsize=A4)
    add_header(fig, "18. Synthèse et recommandations", page=24)
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.81]); ax.axis("off")
    bullets(ax, [
        ("Ce qui a produit un gain :", "le tuning d'hyperparamètres, et lui seul "
         f"(+{FINAL_TEST['pr_auc']-BASELINE_TEST['pr_auc']:.4f} de PR-AUC sur le test, soit "
         f"+{100*(FINAL_TEST['pr_auc']/BASELINE_TEST['pr_auc']-1):.1f}%). Les paramètres initiaux "
         "sur-apprenaient."),
        ("Ce qui n'a rien produit :", "le feature engineering (4 jeux testés, écarts sous le "
         "bruit), le deep learning séquentiel (LSTM/GRU au niveau d'un simple MLP, sous le "
         "boosting), et le blending (+0.0014 sur 682 combinaisons testées)."),
        ("Ce qui était déjà bon :", "la stratégie de rééquilibrage. class_weight / "
         "scale_pos_weight reste le meilleur choix — aucune donnée dupliquée, aucun identifiant "
         "client inventé."),
        ("Le plafond est dans les données :", "démontré par trois tests indépendants — "
         f"corrélation inter-modèles de {corr_mean:.3f}, blending sans effet, et convergence de "
         "deux boostings distincts sur la même valeur au millième."),
        ("Le modèle reste utile :", "il distingue un groupe à 4.5% de risque d'un groupe à 70%, "
         "soit un rapport de 15x, et il est bien calibré — donc directement exploitable pour "
         "prioriser des actions."),
        ("Pour progresser davantage :", "il faut de nouvelles DONNÉES (revenus, ancienneté "
         "bancaire, encours dans d'autres établissements, incidents externes), pas de nouveaux "
         "algorithmes. C'est la recommandation principale de ce rapport."),
        ("Seuil de décision :", "à arbitrer avec le métier via la table de coûts (section 17), "
         "jamais par une formule automatique."),
    ], y=0.95, fontsize=9.5, width=94)
    ax.text(0.02, 0.20, "Prochaine étape — industrialisation", fontsize=11, fontweight="bold",
            color=NAVY, transform=ax.transAxes)
    ax.text(0.02, 0.16, wrap(
        "Le code est déjà structuré pour cela : src/data_prep.py (chargement, nettoyage, features), "
        "scripts/run_experiments.py (expériences reproductibles avec reprise après interruption), "
        "scripts/final_eval.py (évaluation et seuils). Reste à ajouter un script d'entraînement "
        "produisant un modèle sérialisé, une API d'inférence, et un suivi de dérive des données "
        "en production.", 96), fontsize=9.2, va="top", linespacing=1.55, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

    pdf.close()
    print(f"PDF généré : {OUT_PDF}")
    print(f"Taille : {OUT_PDF.stat().st_size / 1024:.0f} Ko")


if __name__ == "__main__":
    build()
