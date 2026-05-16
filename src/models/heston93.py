"""
Heston (1993) — Stochastic Volatility continu.

Référence : Heston, S. L. (1993). "A closed-form solution for options with
stochastic volatility with applications to bond and currency options."
Review of Financial Studies, 6(2), 327-343.

Modèle sous Q :
    dS/S       = r dt + sqrt(V) dW_1
    dV         = kappa (theta - V) dt + sigma sqrt(V) dW_2
    d<W_1,W_2> = rho dt

Paramètres :
    kappa : vitesse de mean-reversion de V       (par an)
    theta : niveau de long-terme de V            (variance annuelle)
    sigma : "vol of vol" — diffusion de V        (par sqrt(an))
    rho   : corrélation S/V                      (négatif sur actions = leverage)
    V_0   : variance instantanée à t=0

Feller condition : 2 kappa theta >= sigma^2 garantit V > 0 p.s.

Fonction caractéristique
------------------------
log phi(u, tau) = i u (log S_0 + r tau) + A(tau, u) + B(tau, u) V_0

avec, en posant
    b = kappa - rho sigma i u
    d = sqrt( b^2 + sigma^2 (u^2 + i u) )
    g = (b + d) / (b - d)

(on utilise la formulation alternative qui évite les sauts de log) :

    B(tau, u) = ((b + d) / sigma^2) * (1 - exp(d tau)) / (1 - g exp(d tau))
    A(tau, u) = (kappa theta / sigma^2)
                * [ (b + d) tau - 2 log( (1 - g exp(d tau)) / (1 - g) ) ]

Les Riccati ODE sous-jacentes sont :
    dA/dtau = kappa theta B
    dB/dtau = -1/2 (u^2 + i u) - b B + (sigma^2 / 2) B^2
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .base import BasePricer


@dataclass(frozen=True)
class HestonParameters:
    kappa: float
    theta: float
    sigma: float
    rho: float
    V_0: float

    @property
    def feller(self) -> float:
        """2 kappa theta - sigma^2. >0 garantit V > 0 p.s."""
        return 2 * self.kappa * self.theta - self.sigma ** 2

    @property
    def feller_satisfied(self) -> bool:
        return self.feller >= 0

    @property
    def long_run_vol(self) -> float:
        """Vol annualisée de long terme = sqrt(theta)."""
        return float(np.sqrt(self.theta))


# ============================================================================
# Briques math : A et B (Riccati closed-form)
# ============================================================================

def heston_AB(
    u: np.ndarray | complex,
    tau: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule (A(tau, u), B(tau, u)) pour Heston 1993.

    Vectorisé en u.

    Cas dégénérés gérés : pour u = 0 et u = -i, le coefficient u^2 + iu
    s'annule. La Riccati pour B devient dB/dtau = -b*B + (sigma^2/2) B^2
    avec B(0) = 0, dont la solution est B(tau) = 0 identiquement, donc
    A(tau) = 0. On force ces valeurs pour éviter les NaN numériques
    (b - d → 0 dans la formulation closed-form).
    """
    u = np.asarray(u, dtype=complex)
    scalar_input = u.ndim == 0
    if scalar_input:
        u = u.reshape(1)

    A = np.zeros_like(u)
    B = np.zeros_like(u)

    # Détection des cas dégénérés (u = 0 ou u = -i)
    degen = np.abs(u ** 2 + 1j * u) < 1e-14
    mask = ~degen

    if mask.any():
        u_ok = u[mask]
        b = kappa - rho * sigma * 1j * u_ok
        d = np.sqrt(b ** 2 + sigma ** 2 * (u_ok ** 2 + 1j * u_ok))
        g = (b + d) / (b - d)
        exp_dtau = np.exp(d * tau)

        B[mask] = ((b + d) / sigma ** 2) * (1 - exp_dtau) / (1 - g * exp_dtau)
        A[mask] = (kappa * theta / sigma ** 2) * (
            (b + d) * tau - 2 * np.log((1 - g * exp_dtau) / (1 - g))
        )

    if scalar_input:
        return A[0], B[0]
    return A, B


# ============================================================================
# Pricer
# ============================================================================

class HestonPricer(BasePricer):
    """Pricer Heston 1993.

    Parameters
    ----------
    params : HestonParameters
    r_f : float
        Taux sans-risque annualisé continu.
    """

    def __init__(self, params: HestonParameters, r_f: float = 0.0):
        self.params = params
        self.r_f = r_f

    # ----- CF -------------------------------------------------------------

    def char_fn_log(self, u, T_years: float, log_S: float):
        p = self.params
        A, B = heston_AB(u, T_years, p.kappa, p.theta, p.sigma, p.rho)
        return 1j * u * (log_S + self.r_f * T_years) + A + B * p.V_0

    # ----- API de calibration --------------------------------------------
    # Ordre interne : [log_kappa, log_theta, log_sigma, rho, log_V_0]

    def get_calibration_vector(self) -> np.ndarray:
        p = self.params
        return np.array([
            np.log(p.kappa),
            np.log(p.theta),
            np.log(p.sigma),
            p.rho,
            np.log(p.V_0),
        ])

    @classmethod
    def from_calibration_vector(cls, vec, r_f: float = 0.0) -> "HestonPricer":
        params = HestonParameters(
            kappa=float(np.exp(vec[0])),
            theta=float(np.exp(vec[1])),
            sigma=float(np.exp(vec[2])),
            rho=float(vec[3]),
            V_0=float(np.exp(vec[4])),
        )
        return cls(params, r_f=r_f)

    @classmethod
    def calibration_bounds(cls) -> list[tuple[float, float]]:
        return [
            (np.log(1e-3), np.log(50.0)),   # log_kappa : reversion 0.001 à 50 / an
            (np.log(1e-6), np.log(2.0)),    # log_theta : variance moyenne, vol 0.1% à 140%
            (np.log(1e-3), np.log(5.0)),    # log_sigma
            (-0.999, 0.999),                 # rho
            (np.log(1e-6), np.log(2.0)),    # log_V_0
        ]

    @property
    def is_stationary(self) -> bool:
        # SDE stationnaire si kappa > 0
        return self.params.kappa > 0
