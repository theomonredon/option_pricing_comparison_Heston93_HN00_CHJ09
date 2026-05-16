"""Loss functions for option-pricing model calibration."""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..models import BasePricer, implied_vol, bs_vega


def _model_prices(pricer: BasePricer, options: pd.DataFrame) -> np.ndarray:
    return pricer.price_chain(
        S=options["spot"].to_numpy(),
        K=options["strike"].to_numpy(),
        T_years=(options["dte"] / 365).to_numpy(),
        rights=options["right"].to_numpy(),
    )


def loss_price_rmse(pricer: BasePricer, options: pd.DataFrame) -> float:
    model = _model_prices(pricer, options)
    market = options["mid"].to_numpy()
    return float(np.sqrt(np.mean((model - market) ** 2)))


def loss_iv_vega_rmse(pricer: BasePricer, options: pd.DataFrame) -> float:
    """RMSE on IV, weighted by vega. Christoffersen-Jacobs (2004) standard."""
    model_prices = _model_prices(pricer, options)
    market_prices = options["mid"].to_numpy()
    spots = options["spot"].to_numpy()
    strikes = options["strike"].to_numpy()
    T = (options["dte"] / 365).to_numpy()
    r = pricer.r_f
    rights = options["right"].str.lower().to_numpy()

    sq = []
    for i in range(len(options)):
        iv_mkt = implied_vol(market_prices[i], spots[i], strikes[i], T[i], r, rights[i])
        iv_mod = implied_vol(model_prices[i], spots[i], strikes[i], T[i], r, rights[i])
        if np.isnan(iv_mkt) or np.isnan(iv_mod):
            continue
        v = bs_vega(spots[i], strikes[i], T[i], r, iv_mkt)
        sq.append((iv_mod - iv_mkt) ** 2 * v ** 2)
    if not sq:
        return np.inf
    return float(np.sqrt(np.mean(sq)))


def evaluate_metrics(
    pricer: BasePricer,
    options: pd.DataFrame,
    r_annual: float = 0.0,
) -> dict:
    """Compute several metrics for one (pricer, panel). Used both in-sample
    and out-of-sample."""
    model = _model_prices(pricer, options)
    market = options["mid"].to_numpy()
    spots = options["spot"].to_numpy()
    strikes = options["strike"].to_numpy()
    T = (options["dte"] / 365).to_numpy()
    rights = options["right"].str.lower().to_numpy()

    iv_mkt = np.array([
        implied_vol(market[i], spots[i], strikes[i], T[i], r_annual, rights[i])
        for i in range(len(options))
    ])
    iv_mod = np.array([
        implied_vol(model[i], spots[i], strikes[i], T[i], r_annual, rights[i])
        for i in range(len(options))
    ])
    valid = ~(np.isnan(iv_mkt) | np.isnan(iv_mod))
    vegas = np.array([
        bs_vega(spots[i], strikes[i], T[i], r_annual, iv_mkt[i]) if valid[i] else 0.0
        for i in range(len(options))
    ])

    iv_err = iv_mod - iv_mkt
    price_err = model - market

    return {
        "n_options": int(valid.sum()),
        "rmse_price": float(np.sqrt(np.mean(price_err ** 2))),
        "mae_price": float(np.mean(np.abs(price_err))),
        "rmse_iv": float(np.sqrt(np.nanmean(iv_err[valid] ** 2))) if valid.any() else np.nan,
        "mae_iv": float(np.nanmean(np.abs(iv_err[valid]))) if valid.any() else np.nan,
        "iv_vega_rmse": float(np.sqrt(np.nanmean((iv_err[valid] * vegas[valid]) ** 2))) if valid.any() else np.nan,
    }
