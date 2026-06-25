"""
BasePricer abstrait : machinerie commune de pricing par inversion de Fourier.

Tous les modèles (Heston 1993, Heston-Nandi 2000, CHJ 2009) admettent une
fonction caractéristique sous Q de la forme :

    phi(u, tau) = E^Q[ exp( i u log S_T ) | F_0 ]
                = exp( i u (log S_0 + r tau) + g_model(u, tau) )

où g_model(u, tau) est la partie "variance-only" qui dépend du modèle (Riccati
pour Heston, récursion A/B pour HN, somme pour CHJ).

La formule de Heston (1993) qui donne le prix d'option à partir de phi est
universelle :

    C = S * P_1 - K e^{-r tau} P_2

    P_2 = 1/2 + (1/pi) ∫_0^∞ Re[ exp(-i u log K) phi(u, tau) / (i u) ] du
    P_1 = 1/2 + (1/pi) ∫_0^∞ Re[ exp(-i u log K) phi(u - i, tau) / (i u phi(-i, tau)) ] du

où phi(-i, tau) = E^Q[S_T] = S_0 e^{r tau} (propriété de martingale).

Convention de temps : on standardise sur **années** au niveau de l'API
(tau = T_days / 252). Chaque pricer convertit en interne si son modèle est
en temps discret journalier (HN).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence


# ============================================================================
# Quadrature Gauss-Legendre cachée (calculée une fois)
# ============================================================================

_QUAD_CACHE: dict = {}


def _gauss_nodes(n: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    key = (n, lo, hi)
    if key not in _QUAD_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        # Rescale [-1, 1] -> [lo, hi]
        x_scaled = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        w_scaled = 0.5 * (hi - lo) * w
        _QUAD_CACHE[key] = (x_scaled, w_scaled)
    return _QUAD_CACHE[key]


# ============================================================================
# BasePricer
# ============================================================================

class BasePricer(ABC):
    """Classe de base pour tout pricer par inversion de Fourier.

    Les sous-classes implémentent ``char_fn_log`` (le log de la CF de log(S_T)
    sous Q). La méthode ``price_call`` est commune.

    Attributs
    ---------
    r_f : float
        Taux sans-risque annualisé continu.
    """

    r_f: float = 0.0

    # ----- Méthode à implémenter dans chaque sous-classe ------------------

    @abstractmethod
    def char_fn_log(self, u: np.ndarray, T_years: float, log_S: float) -> np.ndarray:
        """log phi(u, T) = log E^Q[ exp(i u log S_T) | S_0 ].

        Doit être vectorisée en ``u`` (numpy complex array).
        ``T_years`` est en années.
        ``log_S`` = log(S_0).
        """
        ...

    # ----- Inversion de Fourier (commun à tous les modèles) ---------------

    def _P_integral(
        self,
        S: float,
        K: float,
        T_years: float,
        which: int,
        n_quad: int = 64,
        upper: float = 200.0,
    ) -> float:
        """Calcule l'intégrale pour P_1 (which=1) ou P_2 (which=2)."""
        u, w = _gauss_nodes(n_quad, 1e-8, upper)
        log_S = np.log(S)
        log_K = np.log(K)

        # phi(u, T) en complexe
        if which == 2:
            phi_u = np.exp(self.char_fn_log(u + 0j, T_years, log_S))
            integrand = np.real(np.exp(-1j * u * log_K) * phi_u / (1j * u))
        else:  # which == 1
            # phi(u - i, T) / phi(-i, T)
            phi_u_minus_i = np.exp(self.char_fn_log(u - 1j, T_years, log_S))
            phi_minus_i = np.exp(self.char_fn_log(np.array([-1j]), T_years, log_S))[0]
            integrand = np.real(
                np.exp(-1j * u * log_K) * phi_u_minus_i / (1j * u * phi_minus_i)
            )

        return float((w * integrand).sum())

    def price_call(self, S: float, K: float, T_years: float, **kw) -> float:
        """Prix d'un call européen.

        Parameters
        ----------
        S, K : float
        T_years : float
            Maturité en années.
        """
        if T_years <= 0:
            return max(S - K, 0.0)
        P1 = 0.5 + self._P_integral(S, K, T_years, which=1, **kw) / np.pi
        P2 = 0.5 + self._P_integral(S, K, T_years, which=2, **kw) / np.pi
        return S * P1 - K * np.exp(-self.r_f * T_years) * P2

    def price_put(self, S: float, K: float, T_years: float, **kw) -> float:
        """Prix d'un put européen (put-call parity)."""
        call = self.price_call(S, K, T_years, **kw)
        return call - S + K * np.exp(-self.r_f * T_years)

    def price(self, S: float, K: float, T_years: float, right: str = "call", **kw) -> float:
        right = right.lower()
        if right in ("c", "call"):
            return self.price_call(S, K, T_years, **kw)
        elif right in ("p", "put"):
            return self.price_put(S, K, T_years, **kw)
        raise ValueError(f"right invalide : {right}")

    # ----- Version batchée rapide ----------------------------------------

    def price_chain(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T_years: np.ndarray,
        rights: np.ndarray,
        n_quad: int = 64,
        upper: float = 200.0,
    ) -> np.ndarray:
        """Price un panel d'options en vectorisant l'intégrande.

        Optimisation : on regroupe les options par maturité (mêmes nœuds
        Gauss-Legendre + même call à char_fn_log).
        """
        K = np.asarray(K, dtype=float)
        T_years = np.asarray(T_years, dtype=float)
        rights = np.asarray([str(r).lower() for r in rights])
        N = len(K)
        if np.isscalar(S):
            S = np.full(N, float(S))
        else:
            S = np.asarray(S, dtype=float)

        u, w = _gauss_nodes(n_quad, 1e-8, upper)
        prices = np.empty(N)

        # Group by (log_S, T): char_fn_log is called once per group, then all
        # strikes in the group are priced in a single vectorized operation
        # (e_phase is a 2-D matrix [n_opts × n_quad] computed via outer product).
        unique_T = np.unique(T_years)
        log_S_all = np.log(S)
        for T in unique_T:
            mask_T = T_years == T
            for log_S_val in np.unique(log_S_all[mask_T]):
                mask = mask_T & (log_S_all == log_S_val)
                if not mask.any():
                    continue

                phi_u = np.exp(self.char_fn_log(u + 0j, T, log_S_val))
                phi_u_minus_i = np.exp(self.char_fn_log(u - 1j, T, log_S_val))
                phi_minus_i = np.exp(
                    self.char_fn_log(np.array([-1j]), T, log_S_val)
                )[0]

                S_val = np.exp(log_S_val)
                disc = np.exp(-self.r_f * T)

                indices = np.where(mask)[0]
                K_grp = K[indices]                        # (m,)
                log_K = np.log(K_grp)                     # (m,)

                # e_phase[i, j] = exp(-i * log_K[i] * u[j]): shape (m, n_quad)
                e_phase = np.exp(-1j * np.outer(log_K, u))

                g1 = np.real(e_phase * (phi_u_minus_i / (1j * u * phi_minus_i)))
                g2 = np.real(e_phase * (phi_u / (1j * u)))

                P1 = 0.5 + (w * g1).sum(axis=1) / np.pi  # (m,)
                P2 = 0.5 + (w * g2).sum(axis=1) / np.pi  # (m,)

                calls = S_val * P1 - K_grp * disc * P2    # (m,)
                is_put = ~np.isin(rights[indices], ("c", "call"))
                prices[indices] = np.where(is_put, calls - S_val + K_grp * disc, calls)

        return prices

    # ----- API de calibration (à compléter par chaque pricer) ------------

    @abstractmethod
    def get_calibration_vector(self) -> np.ndarray:
        """Vecteur des paramètres à calibrer (en espace interne, log-scale
        pour les positifs)."""
        ...

    @classmethod
    @abstractmethod
    def from_calibration_vector(
        cls, vec: np.ndarray, r_f: float = 0.0
    ) -> "BasePricer":
        """Reconstruction depuis le vecteur de calibration."""
        ...

    @classmethod
    @abstractmethod
    def calibration_bounds(cls) -> list[tuple[float, float]]:
        """Bornes pour chaque coord du vecteur de calibration."""
        ...

    @property
    @abstractmethod
    def is_stationary(self) -> bool:
        """Le modèle est-il stationnaire (à utiliser pour rejeter pendant
        l'optim)."""
        ...
