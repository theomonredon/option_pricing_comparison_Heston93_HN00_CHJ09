"""Three option pricing models with a unified Fourier-inversion interface."""

from .base import BasePricer
from .heston93 import HestonParameters, HestonPricer, heston_AB
from .hn2000 import (
    HNParameters, HNPricer, garch_filter, neg_log_likelihood, AB_recursion,
)
from .chj2009 import CHJParameters, CHJPricer
from .black_scholes import bs_call, bs_put, bs_vega, implied_vol, implied_vol_vec, bs_vega_vec

MODEL_REGISTRY: dict[str, type[BasePricer]] = {
    "heston93": HestonPricer,
    "hn2000": HNPricer,
    "chj2009": CHJPricer,
}

__all__ = [
    "BasePricer", "MODEL_REGISTRY",
    "HestonParameters", "HestonPricer", "heston_AB",
    "HNParameters", "HNPricer", "garch_filter", "neg_log_likelihood", "AB_recursion",
    "CHJParameters", "CHJPricer",
    "bs_call", "bs_put", "bs_vega", "implied_vol", "implied_vol_vec", "bs_vega_vec",
]
