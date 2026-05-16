"""Aggregation metrics over the calibration / OOS result tables."""

from __future__ import annotations
import numpy as np
import pandas as pd


def bucket_maturity(dte: pd.Series, edges=(7, 30, 90, 365)) -> pd.Series:
    """Returns 'short', 'medium', 'long', 'other'."""
    labels = ["short", "medium", "long"]
    out = pd.cut(dte, bins=edges, labels=labels, include_lowest=True)
    return out.astype(object).fillna("other")


def bucket_moneyness(m: pd.Series, edges=(0.0, 0.95, 1.05, np.inf)) -> pd.Series:
    """Returns 'otm_put', 'atm', 'otm_call'."""
    labels = ["otm_put", "atm", "otm_call"]
    return pd.cut(m, bins=edges, labels=labels, include_lowest=True).astype(object)


def classify_regime(realized_vol_ann: float,
                    calm_thr: float = 0.15,
                    stressed_thr: float = 0.25) -> str:
    """Classify a single date by its realized vol level."""
    if realized_vol_ann < calm_thr:
        return "calm"
    if realized_vol_ann < stressed_thr:
        return "normal"
    return "stressed"


def realized_vol_series(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualised realised vol on rolling window (21j by default)."""
    return returns.rolling(window).std() * np.sqrt(252)


def parameter_stability(
    calib_df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("ticker", "model"),
    param_cols: tuple[str, ...] = ("param_value",),
) -> pd.DataFrame:
    """Compute coefficient of variation and lag-1 autocorrelation
    per (ticker, model, param)."""
    rows = []
    for keys, sub in calib_df.groupby(list(group_cols) + ["param_name"]):
        x = sub.sort_values("date")["param_value"].to_numpy()
        if len(x) < 3:
            continue
        cv = float(np.std(x) / np.abs(np.mean(x))) if np.mean(x) != 0 else np.nan
        try:
            ac = float(pd.Series(x).autocorr(lag=1))
        except Exception:
            ac = np.nan
        rows.append({
            **dict(zip(list(group_cols) + ["param_name"], keys)),
            "n_obs": len(x),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "cv": cv,
            "autocorr_lag1": ac,
        })
    return pd.DataFrame(rows)
