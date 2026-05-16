"""
Heston-Nandi (2000) — Closed-form GARCH option pricing.

Référence : Heston, S. L., & Nandi, S. (2000). "A closed-form GARCH option
valuation model." Review of Financial Studies, 13(3), 585-625.

Modèle (temps discret, indexé en JOURS) :

Sous P :
    log(S_t / S_{t-1}) = r + lambda h_t + sqrt(h_t) z_t,   z_t ~ N(0,1)
    h_t = omega + beta h_{t-1} + alpha (z_{t-1} - gamma sqrt(h_{t-1}))^2

Sous Q (équations 9-12 du papier) :
    lambda* = -1/2
    gamma*  = gamma + lambda + 1/2
(omega, alpha, beta inchangés.)

Fonction caractéristique
------------------------
Sous Q, le log-prix admet :

    log phi(u, n) = log(E^Q[ exp(i u log S_{t+n}) | F_t ])
                  = i u (log S_t + r * n) + A(n, u) + B(n, u) h_{t+1}

avec la récursion BACKWARD (n_steps pas):

    A_{n+1} = A_n + i u r + B_n omega - 0.5 log(1 - 2 alpha B_n)
    B_{n+1} = i u (lambda* + gamma*) - 0.5 (gamma*)^2 + beta B_n
              + 0.5 (i u - gamma*)^2 / (1 - 2 alpha B_n)

initialisation A_0 = B_0 = 0.

Notes sur les unités
--------------------
HN est intrinsèquement journalier. Pour s'aligner avec Heston/CHJ (annuel)
dans le pricer générique, on convertit :
    - n_steps = round(T_years * 252)
    - r_daily = r_annual / 252  (interne à la récursion)
    - h_next = variance journalière (l'utilisateur la passe en variance/jour ;
      la version annualisée serait h_next * 252).
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from .base import BasePricer


# ============================================================================
# Paramètres
# ============================================================================

@dataclass(frozen=True)
class HNParameters:
    omega: float
    alpha: float
    beta: float
    gamma: float
    lam: float

    @property
    def persistence(self) -> float:
        """beta + alpha * gamma^2 : doit être < 1 pour stationnarité."""
        return self.beta + self.alpha * self.gamma ** 2

    @property
    def unconditional_variance(self) -> float:
        """Variance journalière inconditionnelle (omega + alpha) / (1 - persistence)."""
        denom = 1.0 - self.persistence
        if denom <= 0:
            raise ValueError(
                f"Modèle non-stationnaire : persistence = {self.persistence:.4f}"
            )
        return (self.omega + self.alpha) / denom

    @property
    def long_run_vol_annual(self) -> float:
        """Vol annualisée de long terme."""
        return float(np.sqrt(self.unconditional_variance * 252))

    def to_risk_neutral(self) -> "HNParameters":
        return replace(
            self,
            gamma=self.gamma + self.lam + 0.5,
            lam=-0.5,
        )

    def as_array(self):
        return np.array([self.omega, self.alpha, self.beta, self.gamma, self.lam])

    @classmethod
    def from_array(cls, arr) -> "HNParameters":
        return cls(*[float(x) for x in arr])


# ============================================================================
# Filtre GARCH (historique des returns -> h_t)
# ============================================================================

def garch_filter(
    returns: np.ndarray,
    params: HNParameters,
    h0: float,
    r_daily: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtre HN-GARCH : reconstruit (h_t) depuis les returns observés.

    Retourne (h, z) où h a longueur T+1 (h[-1] = h_{T+1}, la variance "demain")
    et z a longueur T.
    """
    returns = np.asarray(returns, dtype=float)
    T = returns.size
    h = np.empty(T + 1)
    z = np.empty(T)
    h[0] = h0

    for t in range(T):
        sqrt_h = np.sqrt(h[t])
        z[t] = (returns[t] - r_daily - params.lam * h[t]) / sqrt_h
        h[t + 1] = (
            params.omega
            + params.beta * h[t]
            + params.alpha * (z[t] - params.gamma * sqrt_h) ** 2
        )

    return h, z


def neg_log_likelihood(
    returns: np.ndarray,
    params: HNParameters,
    h0: float,
    r_daily: float = 0.0,
) -> float:
    """Neg log-vraisemblance gaussienne — pour calibration P-MLE sur les
    returns historiques."""
    h, _ = garch_filter(returns, params, h0, r_daily)
    h_in = h[:-1]
    eps = returns - r_daily - params.lam * h_in
    ll = -0.5 * np.sum(np.log(2 * np.pi * h_in) + eps ** 2 / h_in)
    return -ll


# ============================================================================
# Récursion A/B
# ============================================================================

def AB_recursion(
    u: np.ndarray | complex,
    n_steps: int,
    params_q: HNParameters,
    r_daily: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Récursion A/B vectorisée en u, sous les paramètres Q."""
    u = np.asarray(u, dtype=complex)
    A = np.zeros_like(u)
    B = np.zeros_like(u)

    omega = params_q.omega
    alpha = params_q.alpha
    beta = params_q.beta
    gamma = params_q.gamma
    lam = params_q.lam

    for _ in range(n_steps):
        denom = 1.0 - 2.0 * alpha * B
        A_new = A + 1j * u * r_daily + B * omega - 0.5 * np.log(denom)
        B_new = (
            1j * u * (lam + gamma)
            - 0.5 * gamma ** 2
            + beta * B
            + 0.5 * (1j * u - gamma) ** 2 / denom
        )
        A, B = A_new, B_new

    return A, B


# ============================================================================
# Pricer
# ============================================================================

class HNPricer(BasePricer):
    """Pricer Heston-Nandi (2000).

    Parameters
    ----------
    params : HNParameters
        Paramètres SOUS P. Le pricer convertit en Q en interne.
    h_next : float
        Variance journalière conditionnelle h_{t+1} (issue du filtre GARCH).
    r_f : float
        Taux sans-risque ANNUALISÉ continu.
    """

    def __init__(
        self,
        params: HNParameters,
        h_next: float,
        r_f: float = 0.0,
    ):
        self.params_p = params
        self.params_q = params.to_risk_neutral()
        self.h_next = h_next
        self.r_f = r_f

    @property
    def r_daily(self) -> float:
        return self.r_f / 252

    def char_fn_log(self, u, T_years: float, log_S: float):
        n_steps = int(round(T_years * 252))
        A, B = AB_recursion(u, n_steps, self.params_q, self.r_daily)
        # Note: la récursion accumule déjà i*u*r_daily*n_steps dans A
        return 1j * u * log_S + A + B * self.h_next

    # ----- API de calibration --------------------------------------------
    # Ordre : [log_omega, log_alpha, beta, gamma, lam, log_h_next]
    # h_next est calibré comme un paramètre additionnel (peut être initialisé
    # depuis le filtre GARCH sur l'historique).

    def get_calibration_vector(self) -> np.ndarray:
        p = self.params_p
        return np.array([
            np.log(max(p.omega, 1e-20)),
            np.log(max(p.alpha, 1e-20)),
            p.beta, p.gamma, p.lam,
            np.log(max(self.h_next, 1e-20)),
        ])

    @classmethod
    def from_calibration_vector(cls, vec, r_f: float = 0.0) -> "HNPricer":
        params = HNParameters(
            omega=float(np.exp(vec[0])),
            alpha=float(np.exp(vec[1])),
            beta=float(vec[2]),
            gamma=float(vec[3]),
            lam=float(vec[4]),
        )
        h_next = float(np.exp(vec[5]))
        return cls(params, h_next=h_next, r_f=r_f)

    @classmethod
    def calibration_bounds(cls) -> list[tuple[float, float]]:
        return [
            (np.log(1e-15), np.log(1e-3)),  # log_omega
            (np.log(1e-12), np.log(1e-3)),  # log_alpha
            (1e-4, 0.999),                   # beta
            (-1000.0, 1000.0),               # gamma
            (-5.0, 5.0),                     # lam
            (np.log(1e-10), np.log(1e-2)),  # log_h_next
        ]

    @property
    def is_stationary(self) -> bool:
        return self.params_p.persistence < 0.9999
