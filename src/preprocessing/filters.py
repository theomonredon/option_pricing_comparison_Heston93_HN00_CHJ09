"""Option-chain filtering for calibration. Six filters + audit trail."""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class FilterConfig:
    min_price: float = 0.10
    max_rel_spread: float = 0.25
    min_dte: int = 7
    max_dte: int = 365
    min_moneyness: float = 0.7
    max_moneyness: float = 1.3
    min_volume: int = 1
    r_annual: float = 0.0


def _mask_no_crossed(df: pd.DataFrame) -> pd.Series:
    return (df["bid"] > 0) & (df["ask"] > df["bid"])


def _mask_min_price(df: pd.DataFrame, m: float) -> pd.Series:
    return df["mid"] >= m


def _mask_max_rel_spread(df: pd.DataFrame, m: float) -> pd.Series:
    rel = df["spread"] / df["mid"].replace(0, np.nan)
    return rel <= m


def _mask_dte(df: pd.DataFrame, mn: int, mx: int) -> pd.Series:
    return (df["dte"] >= mn) & (df["dte"] <= mx)


def _mask_moneyness(df: pd.DataFrame, mn: float, mx: float) -> pd.Series:
    return (df["moneyness"] >= mn) & (df["moneyness"] <= mx)


def _mask_volume(df: pd.DataFrame, m: int) -> pd.Series:
    return df["volume"] >= m


def _mask_intrinsic(df: pd.DataFrame, r: float) -> pd.Series:
    T = df["dte"] / 365.0
    disc_K = df["strike"] * np.exp(-r * T)
    is_call = df["right"].str.upper() == "CALL"
    intrinsic_call = np.maximum(df["spot"] - disc_K, 0.0)
    intrinsic_put = np.maximum(disc_K - df["spot"], 0.0)
    intrinsic = np.where(is_call, intrinsic_call, intrinsic_put)
    return df["mid"] >= intrinsic - 1e-6


def filter_chain(
    df: pd.DataFrame,
    config: FilterConfig | None = None,
    return_audit: bool = False,
):
    """Apply all filters. Returns clean DataFrame (+ audit dict if requested)."""
    cfg = config or FilterConfig()
    masks = {
        "no_crossed":     _mask_no_crossed(df),
        "min_price":      _mask_min_price(df, cfg.min_price),
        "max_rel_spread": _mask_max_rel_spread(df, cfg.max_rel_spread),
        "dte":            _mask_dte(df, cfg.min_dte, cfg.max_dte),
        "moneyness":      _mask_moneyness(df, cfg.min_moneyness, cfg.max_moneyness),
        "min_volume":     _mask_volume(df, cfg.min_volume),
        "intrinsic":      _mask_intrinsic(df, cfg.r_annual),
    }
    combined = pd.Series(True, index=df.index)
    audit = {"n_input": len(df), "per_filter": {}}
    for name, m in masks.items():
        audit["per_filter"][name] = {"pass": int(m.sum()), "reject": int((~m).sum())}
        combined &= m
    out = df.loc[combined].copy()
    audit["n_output"] = len(out)
    if return_audit:
        return out, audit
    return out


def add_implied_vol(df: pd.DataFrame, r_annual: float = 0.0) -> pd.DataFrame:
    """Adds 'iv' and 'vega' columns (market). Uses vectorized Newton-Raphson for speed."""
    from ..models import implied_vol_vec, bs_vega_vec
    import numpy as np
    df = df.copy()
    T = (df["dte"] / 365).to_numpy()
    df["iv"] = implied_vol_vec(
        df["mid"].to_numpy(), df["spot"].to_numpy(), df["strike"].to_numpy(),
        T, r_annual, df["right"].to_numpy(),
    )
    df = df.dropna(subset=["iv"]).reset_index(drop=True)
    T2 = (df["dte"] / 365).to_numpy()
    df["vega"] = bs_vega_vec(
        df["spot"].to_numpy(), df["strike"].to_numpy(), T2, r_annual, df["iv"].to_numpy()
    )
    return df
