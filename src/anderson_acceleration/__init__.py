"""Anderson acceleration for fixed-point and implicit-layer experiments."""

from .ml import ImplicitLayerResult, solve_tanh_equilibrium
from .solver import AndersonResult, anderson_accelerate

__all__ = [
    "AndersonResult",
    "ImplicitLayerResult",
    "anderson_accelerate",
    "solve_tanh_equilibrium",
]
