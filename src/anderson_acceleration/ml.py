from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import AndersonResult, anderson_accelerate


@dataclass(frozen=True)
class ImplicitLayerResult:
    hidden_state: np.ndarray
    logits: np.ndarray
    solver: AndersonResult


def solve_tanh_equilibrium(
    input_vector,
    recurrent_weight,
    input_weight,
    bias,
    *,
    readout_weight=None,
    readout_bias=None,
    initial_state=None,
    memory: int = 5,
    beta: float = 1.0,
    regularization: float = 1e-10,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> ImplicitLayerResult:
    """Solve a small implicit tanh layer.

    The fixed point has the same form used in lightweight implicit neural
    layers:

    ``h = tanh(W_h h + W_x x + b)``

    Anderson acceleration is used only for the forward equilibrium solve. The
    helper intentionally does not implement backpropagation; it is a compact
    experiment for studying the numerical behavior of equilibrium layers.
    """

    x = _as_vector(input_vector, "input_vector")
    recurrent = _as_matrix(recurrent_weight, "recurrent_weight")
    input_matrix = _as_matrix(input_weight, "input_weight")
    bias_vector = _as_vector(bias, "bias")

    hidden_dim = recurrent.shape[0]
    if recurrent.shape[1] != hidden_dim:
        raise ValueError("recurrent_weight must be square")
    if input_matrix.shape != (hidden_dim, x.shape[0]):
        raise ValueError("input_weight must have shape (hidden_dim, input_dim)")
    if bias_vector.shape[0] != hidden_dim:
        raise ValueError("bias must have length hidden_dim")

    h0 = (
        np.zeros(hidden_dim, dtype=float)
        if initial_state is None
        else _as_vector(initial_state, "initial_state")
    )
    if h0.shape[0] != hidden_dim:
        raise ValueError("initial_state must have length hidden_dim")

    drive = input_matrix @ x + bias_vector

    def fixed_point(hidden: np.ndarray) -> np.ndarray:
        return np.tanh(recurrent @ hidden + drive)

    solver = anderson_accelerate(
        fixed_point,
        h0,
        memory=memory,
        beta=beta,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
    )
    hidden_state = solver.solution
    logits = _readout(hidden_state, readout_weight, readout_bias)
    return ImplicitLayerResult(hidden_state=hidden_state, logits=logits, solver=solver)


def _readout(hidden_state, readout_weight, readout_bias) -> np.ndarray:
    if readout_weight is None:
        return hidden_state.copy()

    weight = _as_matrix(readout_weight, "readout_weight")
    if weight.shape[1] != hidden_state.shape[0]:
        raise ValueError("readout_weight must have shape (output_dim, hidden_dim)")
    if readout_bias is None:
        bias = np.zeros(weight.shape[0], dtype=float)
    else:
        bias = _as_vector(readout_bias, "readout_bias")
        if bias.shape[0] != weight.shape[0]:
            raise ValueError("readout_bias must have length output_dim")
    return weight @ hidden_state + bias


def _as_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
