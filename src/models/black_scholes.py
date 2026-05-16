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
