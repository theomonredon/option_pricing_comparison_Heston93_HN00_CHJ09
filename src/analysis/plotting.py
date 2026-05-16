"""Standardised plots for the analysis notebook. Each function returns a
matplotlib Figure for easy saving in the notebook."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLORS = {
    "heston93": "#1f77b4",
    "hn2000":   "#ff7f0e",
    "chj2009":  "#2ca02c",
}
MODEL_NAMES = {"heston93": "Heston 1993", "hn2000": "HN 2000", "chj2009": "CHJ 2009"}


def plot_in_sample_heatmap(calib_df: pd.DataFrame, metric: str = "in_sample_iv_vega_rmse"):
    """Heatmap (model x ticker) of median in-sample metric."""
    pivot = calib_df.groupby(["model", "ticker"])[metric].median().unstack("ticker")
    fig, ax = plt.subplots(figsize=(1.5 * pivot.shape[1] + 2, 0.7 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center",
                    color="black", fontsize=9)
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"Median {metric} — in-sample")
    plt.tight_layout()
    return fig


def plot_loss_timeseries(calib_df: pd.DataFrame, metric: str = "in_sample_iv_vega_rmse"):
    """Time series of in-sample loss, one panel per ticker, 3 models superposed."""
    tickers = sorted(calib_df.ticker.unique())
    n = len(tickers)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, t in zip(axes, tickers):
        for m in sorted(calib_df.model.unique()):
            sub = calib_df[(calib_df.ticker == t) & (calib_df.model == m)].sort_values("date")
            ax.plot(sub.date, sub[metric], "o-", markersize=3,
                    color=MODEL_COLORS.get(m), label=MODEL_NAMES.get(m, m))
        ax.set_title(t); ax.tick_params(axis="x", rotation=45)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"In-sample {metric} — time series", y=1.02)
    plt.tight_layout()
    return fig


def plot_oos_by_horizon(oos_df: pd.DataFrame, metric_template: str = "oos_iv_vega_rmse_J{h}"):
    """Distribution of OOS error per horizon, per model."""
    horizons = [1, 2, 5]
    rows = []
    for h in horizons:
        col = metric_template.format(h=h)
        if col not in oos_df.columns:
            continue
        for _, r in oos_df.iterrows():
            if pd.notna(r[col]):
                rows.append({"horizon": h, "model": r["model"], "loss": r[col]})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 4), sharey=True)
    for ax, h in zip(np.atleast_1d(axes).flatten(), horizons):
        sub = df[df.horizon == h]
        models = sorted(sub.model.unique())
        data = [sub[sub.model == m].loss.to_numpy() for m in models]
        bp = ax.boxplot(data, labels=[MODEL_NAMES.get(m, m) for m in models], showfliers=False)
        ax.set_title(f"OOS J+{h}"); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("OOS error distribution by horizon", y=1.02)
    plt.tight_layout()
    return fig


def plot_oos_by_regime(oos_df: pd.DataFrame, horizon: int = 1,
                       metric_template: str = "oos_iv_vega_rmse_J{h}"):
    """OOS error distribution per regime, with 3 models per regime."""
    col = metric_template.format(h=horizon)
    if col not in oos_df.columns or "regime" not in oos_df.columns:
        return None
    regimes = ["calm", "normal", "stressed"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, reg in zip(axes, regimes):
        sub = oos_df[oos_df.regime == reg]
        models = sorted(sub.model.unique())
        data = [sub[sub.model == m][col].dropna().to_numpy() for m in models]
        ax.boxplot(data, labels=[MODEL_NAMES.get(m, m) for m in models], showfliers=False)
        ax.set_title(f"Regime: {reg} ({len(sub)} obs)")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(f"OOS J+{horizon} — by regime", y=1.02)
    plt.tight_layout()
    return fig


def plot_compute_cost(calib_df: pd.DataFrame):
    """Bar chart of median calibration time per model."""
    med = calib_df.groupby("model")["calibration_time_sec"].median()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [MODEL_COLORS.get(m) for m in med.index]
    ax.bar([MODEL_NAMES.get(m, m) for m in med.index], med.values, color=colors)
    ax.set_ylabel("Median calibration time (s)")
    ax.set_title("Computational cost")
    plt.tight_layout()
    return fig
