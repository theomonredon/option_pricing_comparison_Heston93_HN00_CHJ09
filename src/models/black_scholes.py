"""
Mini-Black-Scholes pour comparaison et extraction de volatilité implicite.

Convention : sigma est la volatilité ANNUALISÉE, T est la maturité en
ANNÉES, r est le taux continu annuel.

Pour matcher le modèle Heston-Nandi journalier :
    sigma_annual = sigma_daily * sqrt(252)
    T_annual    = n_steps_daily / 252
    r_annual    = r_f_daily * 252
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1_d2(S, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega = dC/dsigma. Identique pour call et put."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    right: str = "call",
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Volatilité implicite par bissection (Brent).

    Retourne np.nan si le prix est hors des bornes d'arbitrage ou si la
    bissection ne converge pas.
    """
    if T <= 0 or price <= 0:
        return np.nan

    # Bornes d'arbitrage
    intrinsic = max(S - K * np.exp(-r * T), 0.0) if right == "call" else max(K * np.exp(-r * T) - S, 0.0)
    upper_bound = S if right == "call" else K * np.exp(-r * T)
    if price < intrinsic - 1e-10 or price > upper_bound + 1e-10:
        return np.nan

    if right == "call":
        f = lambda sig: bs_call(S, K, T, r, sig) - price
    else:
        f = lambda sig: bs_put(S, K, T, r, sig) - price

    try:
        return brentq(f, lo, hi, xtol=1e-6, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_vec(
    prices: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    rights: np.ndarray,
    n_iter: int = 50,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 10.0,
) -> np.ndarray:
    """Vectorized implied vol via Newton-Raphson over all options simultaneously.

    ~20-50x faster than calling implied_vol() in a Python loop.
    Returns NaN for options where inversion fails (no convergence or bad inputs).
    """
    prices = np.asarray(prices, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    rights = np.asarray([str(r_).lower() for r_ in rights])
    is_call = np.isin(rights, ("c", "call"))

    valid = (T > 0) & (prices > 0) & np.isfinite(prices)

    # Initial guess: flat sigma=0.3 is a reliable starting point across moneyness
    sigma = np.where(valid, 0.3, 1.0)

    bs_price = np.zeros_like(sigma)
    vega_arr = np.zeros_like(sigma)

    for _ in range(n_iter):
        with np.errstate(divide="ignore", invalid="ignore"):
            sqrt_T = np.sqrt(np.maximum(T, 1e-10))
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            disc = np.exp(-r * T)

            bs_price = np.where(
                is_call,
                S * norm.cdf(d1) - K * disc * norm.cdf(d2),
                K * disc * norm.cdf(-d2) - S * norm.cdf(-d1),
            )
            vega_arr = S * norm.pdf(d1) * sqrt_T

            step = np.where(vega_arr > 1e-12, (bs_price - prices) / vega_arr, 0.0)
            sigma = np.where(valid, sigma - step, sigma)
            sigma = np.clip(sigma, sigma_lo, sigma_hi)

    # Accept if |BS(sigma) - price| < 1% of spot (generous tolerance for model prices)
    final_err = np.abs(bs_price - prices)
    out = np.where(valid & (final_err < 0.01 * S), sigma, np.nan)
    return out


def bs_vega_vec(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Vectorized BS vega."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_T = np.sqrt(np.maximum(T, 1e-10))
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return np.where((T > 0) & (sigma > 0), S * norm.pdf(d1) * sqrt_T, 0.0)
