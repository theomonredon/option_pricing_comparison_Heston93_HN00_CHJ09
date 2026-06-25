"""Polymorphic calibration. Works for any BasePricer subclass.

Supports multi-start: ``n_starts`` lance L-BFGS-B depuis ``n_starts`` points
initiaux (le premier = ``x0`` fourni, les autres = perturbations gaussiennes
de ``x0`` clipées aux bornes). On garde le résultat avec la plus faible loss.

Multi-start atténue le risque de minima locaux — surtout utile pour les
modèles à beaucoup de paramètres (CHJ avec 10 dimensions).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ..models import BasePricer
from .losses import loss_iv_vega_rmse, loss_price_rmse


LOSSES = {
    "iv_vega": loss_iv_vega_rmse,
    "price_rmse": loss_price_rmse,
}


def _generate_starts(
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    n_starts: int,
    seed: int | None,
    perturbation: float = 0.25,
) -> list[np.ndarray]:
    """Génère ``n_starts`` points initiaux.

    Le premier est ``x0`` lui-même. Les ``n_starts - 1`` suivants sont
    obtenus en perturbant chaque coordonnée par une uniforme dans
    ``[-perturbation, +perturbation] * (hi - lo)``, clipée aux bornes.
    """
    if n_starts <= 1:
        return [x0.copy()]
    rng = np.random.default_rng(seed)
    starts = [x0.copy()]
    bounds_arr = np.array(bounds, dtype=float)
    widths = bounds_arr[:, 1] - bounds_arr[:, 0]
    for _ in range(n_starts - 1):
        noise = rng.uniform(-perturbation, +perturbation, size=len(x0)) * widths
        x = np.clip(x0 + noise, bounds_arr[:, 0], bounds_arr[:, 1])
        starts.append(x)
    return starts


def calibrate(
    pricer_class: type[BasePricer],
    options: pd.DataFrame,
    x0: BasePricer,
    r_f: float = 0.0,
    loss: str = "iv_vega",
    method: str = "L-BFGS-B",
    maxiter: int = 200,
    ftol: float = 1e-10,
    n_starts: int = 1,
    seed: int | None = 42,
    verbose: bool = False,
) -> tuple[BasePricer, dict]:
    """Calibre un pricer sur un panel d'options.

    Parameters
    ----------
    n_starts : int
        Nombre de points initiaux pour le multi-start. 1 = simple start
        (comportement historique). Recommandé : 5 pour CHJ, 3 pour HN, 1-2
        pour Heston.
    seed : int | None
        Graine pour la génération des points perturbés (reproductibilité).
    """
    bounds = pricer_class.calibration_bounds()
    loss_fn = LOSSES[loss]
    x0_vec = x0.get_calibration_vector()
    starts = _generate_starts(x0_vec, bounds, n_starts, seed)

    history: list[tuple[np.ndarray, float]] = []

    def objective(vec):
        try:
            p = pricer_class.from_calibration_vector(vec, r_f=r_f)
            if not p.is_stationary:
                return 1e6
            v = loss_fn(p, options)
            if not np.isfinite(v):
                return 1e6
            history.append((vec.copy(), v))
            return v
        except Exception:
            return 1e6

    best_res = None
    best_loss = np.inf
    best_idx = -1
    all_losses = []
    all_converged = []

    for idx, start in enumerate(starts):
        res = minimize(
            objective,
            x0=start,
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": ftol},
        )
        all_losses.append(float(res.fun))
        all_converged.append(bool(res.success))
        if verbose:
            print(f"  start {idx + 1}/{n_starts}: loss={res.fun:.6f} converged={res.success}")
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_res = res
            best_idx = idx

    return pricer_class.from_calibration_vector(best_res.x, r_f=r_f), {
        "scipy_result": best_res,
        "history": history,
        "n_options": int(len(options)),
        "loss_name": loss,
        "final_loss": float(best_res.fun),
        "pricer_class": pricer_class.__name__,
        "converged": bool(best_res.success),
        "n_starts": int(n_starts),
        "best_start_idx": int(best_idx),
        "all_start_losses": all_losses,
        "all_start_converged": all_converged,
    }
