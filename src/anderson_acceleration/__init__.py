"""Anderson acceleration for fixed-point and implicit-layer experiments."""

from .experiments import (
    EquilibriumFeatureResult,
    EquilibriumWeights,
    ReadoutResult,
    RidgeFixedPointResult,
    SolverSweepRow,
    equilibrium_features,
    fit_softmax_readout,
    fit_ridge_fixed_point,
    make_equilibrium_weights,
    make_two_moons,
    readout_accuracy,
    readout_predict,
    solver_memory_sweep,
)
from .ml import (
    EquilibriumDiagnostics,
    ImplicitLayerResult,
    solve_tanh_equilibrium,
    tanh_equilibrium_diagnostics,
)
from .solver import AndersonResult, anderson_accelerate, residual_diagnostics

__all__ = [
    "AndersonResult",
    "EquilibriumFeatureResult",
    "EquilibriumDiagnostics",
    "EquilibriumWeights",
    "ImplicitLayerResult",
    "ReadoutResult",
    "RidgeFixedPointResult",
    "SolverSweepRow",
    "anderson_accelerate",
    "equilibrium_features",
    "fit_ridge_fixed_point",
    "fit_softmax_readout",
    "make_equilibrium_weights",
    "make_two_moons",
    "readout_accuracy",
    "readout_predict",
    "residual_diagnostics",
    "solve_tanh_equilibrium",
    "solver_memory_sweep",
    "tanh_equilibrium_diagnostics",
]
