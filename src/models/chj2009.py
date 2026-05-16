"""
Christoffersen-Heston-Jacobs (2009) — Multifactor Stochastic Volatility.

Référence : Christoffersen, P., Heston, S., & Jacobs, K. (2009). "The shape
and term structure of the index option smirk: Why multifactor stochastic
volatility models work so well." Management Science, 55(12), 1914-1932.

Idée centrale : la variance instantanée totale est la somme de deux
composantes Heston indépendantes :

    V_t = V_{1,t} + V_{2,t}
    dV_{i,t} = kappa_i (theta_i - V_{i,t}) dt + sigma_i sqrt(V_{i,t}) dW_{V,i,t}
    d<W_S, W_{V,i,t}> = rho_i dt
    d<W_{V,1}, W_{V,2}> = 0  (indépendance des composantes)

Typiquement on contraint kappa_1 >> kappa_2 (composante rapide vs lente) pour
capturer respectivement :
    - les chocs courts de volatilité (V_1, mean-reversion rapide)
    - la term-structure longue de volatilité (V_2, mean-reversion lente)

Fonction caractéristique
------------------------
Sous l'indépendance des deux facteurs de variance, la CF du log-prix
factorise. En log :

    log phi(u, tau) = i u (log S_0 + r tau)
                   + A_1(tau, u; kappa_1, theta_1, sigma_1, rho_1) + B_1 V_{1,0}
                   + A_2(tau, u; kappa_2, theta_2, sigma_2, rho_2) + B_2 V_{2,0}

où chaque (A_i, B_i) est exactement celui de Heston 1993 pour les paramètres
du facteur i. → On réutilise ``heston_AB`` du module heston93.

Pourquoi ça marche : V_t = V_1 + V_2 indépendants. Le log-prix dépend de
sqrt(V_t) dW_S. Pour faire le calcul propre on écrit dW_S = (sqrt(V_1) dW_S^1
+ sqrt(V_2) dW_S^2) / sqrt(V) où dW_S^i sont des Browniens "associés" à
chaque facteur. Le résultat : la CF s'écrit comme produit des CF de deux
sous-modèles Heston (chacun avec son rho_i bien défini).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .base import BasePricer
from .heston93 import heston_AB


@dataclass(frozen=True)
class CHJParameters:
    # Facteur 1 (typiquement rapide : kappa_1 grand)
    kappa_1: float
    theta_1: float
    sigma_1: float
    rho_1: float
    V_1_0: float
    # Facteur 2 (typiquement lent : kappa_2 petit)
    kappa_2: float
    theta_2: float
    sigma_2: float
    rho_2: float
    V_2_0: float

    @property
    def total_long_run_var(self) -> float:
        """theta_1 + theta_2."""
        return self.theta_1 + self.theta_2

    @property
    def total_initial_var(self) -> float:
        return self.V_1_0 + self.V_2_0

    @property
    def half_lives_years(self) -> tuple[float, float]:
        """Demi-vie de chaque facteur (log(2) / kappa)."""
        return (np.log(2) / self.kappa_1, np.log(2) / self.kappa_2)


# ============================================================================
# Pricer
# ============================================================================

class CHJPricer(BasePricer):
    """Pricer Christoffersen-Heston-Jacobs (2009) à deux facteurs."""

    def __init__(self, params: CHJParameters, r_f: float = 0.0):
        self.params = params
        self.r_f = r_f

    def char_fn_log(self, u, T_years: float, log_S: float):
        p = self.params
        A1, B1 = heston_AB(u, T_years, p.kappa_1, p.theta_1, p.sigma_1, p.rho_1)
        A2, B2 = heston_AB(u, T_years, p.kappa_2, p.theta_2, p.sigma_2, p.rho_2)
        return (
            1j * u * (log_S + self.r_f * T_years)
            + A1 + B1 * p.V_1_0
            + A2 + B2 * p.V_2_0
        )

    # ----- API de calibration --------------------------------------------
    # Ordre : [log_kappa_1, log_theta_1, log_sigma_1, rho_1, log_V_1_0,
    #          log_kappa_2, log_theta_2, log_sigma_2, rho_2, log_V_2_0]

    def get_calibration_vector(self) -> np.ndarray:
        p = self.params
        return np.array([
            np.log(p.kappa_1), np.log(p.theta_1), np.log(p.sigma_1), p.rho_1, np.log(p.V_1_0),
            np.log(p.kappa_2), np.log(p.theta_2), np.log(p.sigma_2), p.rho_2, np.log(p.V_2_0),
        ])

    @classmethod
    def from_calibration_vector(cls, vec, r_f: float = 0.0) -> "CHJPricer":
        params = CHJParameters(
            kappa_1=float(np.exp(vec[0])), theta_1=float(np.exp(vec[1])),
            sigma_1=float(np.exp(vec[2])), rho_1=float(vec[3]),
            V_1_0=float(np.exp(vec[4])),
            kappa_2=float(np.exp(vec[5])), theta_2=float(np.exp(vec[6])),
            sigma_2=float(np.exp(vec[7])), rho_2=float(vec[8]),
            V_2_0=float(np.exp(vec[9])),
        )
        return cls(params, r_f=r_f)

    @classmethod
    def calibration_bounds(cls) -> list[tuple[float, float]]:
        # Facteur 1 : "rapide" (kappa élevé), facteur 2 : "lent" (kappa bas)
        # On laisse l'optim choisir lequel est lequel sans contraindre l'ordre
        single_factor_bounds = [
            (np.log(1e-3), np.log(50.0)),   # log_kappa
            (np.log(1e-6), np.log(2.0)),    # log_theta
            (np.log(1e-3), np.log(5.0)),    # log_sigma
            (-0.999, 0.999),                 # rho
            (np.log(1e-6), np.log(2.0)),    # log_V_0
        ]
        return single_factor_bounds * 2

    @property
    def is_stationary(self) -> bool:
        return self.params.kappa_1 > 0 and self.params.kappa_2 > 0
