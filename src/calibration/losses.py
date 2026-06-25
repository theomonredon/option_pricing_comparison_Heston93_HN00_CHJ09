"""Loss functions for option-pricing model calibration."""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..models import BasePricer, implied_vol_vec, bs_vega_vec


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
    """RMSE on IV, weighted by vega. Christoffersen-Jacobs (2004) standard.

    Uses precomputed market IV and vega (columns 'iv' and 'vega') when available,
    falling back to on-the-fly vectorized computation otherwise.
    """
    model_prices = _model_prices(pricer, options)
    spots = options["spot"].to_numpy()
    strikes = options["strike"].to_numpy()
    T = (options["dte"] / 365).to_numpy()
    r = pricer.r_f
    rights = options["right"].to_numpy()

    # Market IV and vega: use precomputed columns if present (set by add_implied_vol)
    if "iv" in options.columns:
        iv_mkt = options["iv"].to_numpy()
    else:
        iv_mkt = implied_vol_vec(options["mid"].to_numpy(), spots, strikes, T, r, rights)

    if "vega" in options.columns:
        vegas = options["vega"].to_numpy()
    else:
        vegas = bs_vega_vec(spots, strikes, T, r, iv_mkt)

    # Model IV: must be recomputed at each calibration step (vectorized)
    iv_mod = implied_vol_vec(model_prices, spots, strikes, T, r, rights)

    valid = ~(np.isnan(iv_mkt) | np.isnan(iv_mod))
    if not valid.any():
        return np.inf
    sq = (iv_mod[valid] - iv_mkt[valid]) ** 2 * vegas[valid] ** 2
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
    rights = options["right"].to_numpy()

    if "iv" in options.columns:
        iv_mkt = options["iv"].to_numpy()
    else:
        iv_mkt = implied_vol_vec(market, spots, strikes, T, r_annual, rights)

    iv_mod = implied_vol_vec(model, spots, strikes, T, r_annual, rights)

    valid = ~(np.isnan(iv_mkt) | np.isnan(iv_mod))

    if "vega" in options.columns:
        vegas = options["vega"].to_numpy()
    else:
        vegas = bs_vega_vec(spots, strikes, T, r_annual, iv_mkt)

    iv_err = iv_mod - iv_mkt
    price_err = model - market

    return {
        "n_options": int(valid.sum()),
        "rmse_price": float(np.sqrt(np.nanmean(price_err ** 2))),
        "mae_price": float(np.nanmean(np.abs(price_err))),
        "rmse_iv": float(np.sqrt(np.nanmean(iv_err[valid] ** 2))) if valid.any() else np.nan,
        "mae_iv": float(np.nanmean(np.abs(iv_err[valid]))) if valid.any() else np.nan,
        "iv_vega_rmse": float(np.sqrt(np.nanmean((iv_err[valid] * vegas[valid]) ** 2))) if valid.any() else np.nan,
    }
