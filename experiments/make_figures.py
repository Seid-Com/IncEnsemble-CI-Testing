"""Regenerate study figures from results/experiment_results.json.

Figures (IET revision, no 'EMLAST'):
  fig2_sensitivity.png   - hyperparameter sensitivity
  fig3_fdr_budget.png    - FDR vs budget, mean +/- SD
  fig4_atd_budget.png    - ATD vs budget
  fig5_adaptation.png    - per-commit FDR under drift (Lang, 25%)
  fig6_shap.png          - SHAP feature importance (top-10)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

METHOD_ORDER = ["IncEnsemble", "XGBoost", "L2R", "NonIncRF", "Static", "Random"]
COLORS = {
    "IncEnsemble": "#C0392B",
    "XGBoost": "#2980B9",
    "L2R": "#27AE60",
    "NonIncRF": "#8E44AD",
    "Static": "#E67E22",
    "Random": "#7F8C8D",
}
MARKERS = {
    "IncEnsemble": "o",
    "XGBoost": "s",
    "L2R": "^",
    "NonIncRF": "D",
    "Static": "v",
    "Random": "x",
}


def style_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig3_fdr(main_df):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    budgets = sorted(main_df["budget"].unique())
    for m in METHOD_ORDER:
        sub = main_df[main_df["method"] == m]
        means, sds = [], []
        for b in budgets:
            vals = sub[sub["budget"] == b]["fdr"].values
            means.append(vals.mean())
            sds.append(vals.std(ddof=1))
        means, sds = np.array(means), np.array(sds)
        ax.errorbar(budgets, means, yerr=sds, label=m, color=COLORS[m],
                    marker=MARKERS[m], capsize=3, linewidth=1.8, markersize=7)
    style_axes(ax, "CI execution budget (%)", "Fault-detection rate (FDR)",
               "FDR vs execution budget (mean $\\pm$ SD, 5 projects x 10 seeds)")
    ax.set_xticks(budgets)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_fdr_budget.png"), dpi=200)
    plt.close(fig)
    print("wrote fig3_fdr_budget.png")


def fig4_atd(main_df):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    budgets = sorted(main_df["budget"].unique())
    for m in METHOD_ORDER:
        sub = main_df[main_df["method"] == m]
        means, sds = [], []
        for b in budgets:
            vals = sub[sub["budget"] == b]["atd"].values
            means.append(vals.mean())
            sds.append(vals.std(ddof=1))
        means, sds = np.array(means), np.array(sds)
        ax.errorbar(budgets, means, yerr=sds, label=m, color=COLORS[m],
                    marker=MARKERS[m], capsize=3, linewidth=1.8, markersize=7)
    style_axes(ax, "CI execution budget (%)", "Average time-to-detect (s)",
               "ATD vs execution budget (mean $\\pm$ SD)")
    ax.set_xticks(budgets)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_atd_budget.png"), dpi=200)
    plt.close(fig)
    print("wrote fig4_atd_budget.png")


def fig5_adaptation():
    path = os.path.join(RES, "commit_fdr.csv")
    if not os.path.exists(path):
        print("SKIP fig5: commit_fdr.csv missing")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for m in ["IncEnsemble", "NonIncRF", "L2R"]:
        sub = df[df["method"] == m].sort_values("commit")
        if sub.empty:
            continue
        # rolling mean over commits for readability
        y = sub["fdr"].rolling(5, min_periods=1).mean()
        ax.plot(sub["commit"], y, label=m, color=COLORS[m], linewidth=1.8)
    # mark drift point if present
    if "drift" in df.columns:
        drift_commits = df.loc[df["drift"] == True, "commit"]  # noqa: E712
        if len(drift_commits):
            d0 = drift_commits.min()
            ax.axvline(d0, color="black", linestyle="--", linewidth=1, alpha=0.6)
            ax.text(d0 + 1, ax.get_ylim()[1] * 0.95, "concept drift",
                    fontsize=9, va="top")
    style_axes(ax, "Commit index (rolling mean, w=5)", "Fault-detection rate",
               "Online adaptation under concept drift (Lang, 25% budget)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_adaptation.png"), dpi=200)
    plt.close(fig)
    print("wrote fig5_adaptation.png")


def fig2_sensitivity():
    path = os.path.join(RES, "sensitivity.csv")
    if not os.path.exists(path):
        print("SKIP fig2: sensitivity.csv missing")
        return
    df = pd.read_csv(path)
    # column may be 'parameter' or 'param'
    col = "parameter" if "parameter" in df.columns else "param"
    params = df[col].unique()
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, param in zip(axes, params):
        sub = df[df[col] == param].sort_values("value")
        ax.plot(sub["value"], sub["fdr"], "o-", color=COLORS["IncEnsemble"],
                linewidth=1.8, markersize=6)
        style_axes(ax, param, "FDR")
    fig.suptitle("Hyperparameter sensitivity (FDR, Lang, 25% budget)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_sensitivity.png"), dpi=200)
    plt.close(fig)
    print("wrote fig2_sensitivity.png")


def fig6_shap():
    path = os.path.join(RES, "shap.csv")
    if not os.path.exists(path):
        print("SKIP fig6: shap.csv missing")
        return
    df = pd.read_csv(path).sort_values("mean_abs_shap", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(df["feature"], df["mean_abs_shap"], color="#2C3E50")
    style_axes(ax, "Mean |SHAP value|", "",
               "Top-10 predictive features (IncEnsemble)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_shap.png"), dpi=200)
    plt.close(fig)
    print("wrote fig6_shap.png")


def main():
    with open(os.path.join(RES, "experiment_results.json")) as f:
        results = json.load(f)
    main_df = pd.DataFrame(results["main"])
    fig2_sensitivity()
    fig3_fdr(main_df)
    fig4_atd(main_df)
    fig5_adaptation()
    fig6_shap()
    print("All figures in", OUT)


if __name__ == "__main__":
    main()
