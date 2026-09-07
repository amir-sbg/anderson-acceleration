"""Anderson acceleration for fixed-point and implicit-layer experiments."""

from .experiments import (
    EquilibriumFeatureResult,
    EquilibriumWeights,
    ReadoutResult,
    equilibrium_features,
    fit_softmax_readout,
    make_equilibrium_weights,
    make_two_moons,
    readout_accuracy,
    readout_predict,
)
from .ml import (
    EquilibriumDiagnostics,
    ImplicitLayerResult,
    solve_tanh_equilibrium,
    tanh_equilibrium_diagnostics,
)
from .solver import AndersonResult, anderson_accelerate

__all__ = [
    "AndersonResult",
    "EquilibriumFeatureResult",
    "EquilibriumDiagnostics",
    "EquilibriumWeights",
    "ImplicitLayerResult",
    "ReadoutResult",
    "anderson_accelerate",
    "equilibrium_features",
    "fit_softmax_readout",
    "make_equilibrium_weights",
    "make_two_moons",
    "readout_accuracy",
    "readout_predict",
    "solve_tanh_equilibrium",
    "tanh_equilibrium_diagnostics",
]
