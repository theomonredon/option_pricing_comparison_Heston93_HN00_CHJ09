"""Polymorphic calibration. Works for any BasePricer subclass."""

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


def calibrate(
    pricer_class: type[BasePricer],
    options: pd.DataFrame,
    x0: BasePricer,
    r_f: float = 0.0,
    loss: str = "iv_vega",
    method: str = "L-BFGS-B",
    maxiter: int = 200,
    ftol: float = 1e-10,
    verbose: bool = False,
) -> tuple[BasePricer, dict]:
    """Calibrate a pricer on an option panel."""
    bounds = pricer_class.calibration_bounds()
    loss_fn = LOSSES[loss]
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
            if verbose and len(history) % 20 == 0:
                print(f"  iter {len(history):4d}: loss={v:.6f}")
            return v
        except Exception:
            return 1e6

    res = minimize(
        objective,
        x0=x0.get_calibration_vector(),
        method=method,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol},
    )
    return pricer_class.from_calibration_vector(res.x, r_f=r_f), {
        "scipy_result": res,
        "history": history,
        "n_options": int(len(options)),
        "loss_name": loss,
        "final_loss": float(res.fun),
        "pricer_class": pricer_class.__name__,
        "converged": bool(res.success),
    }
