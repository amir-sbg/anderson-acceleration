from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import AndersonResult, anderson_accelerate


@dataclass(frozen=True)
class ImplicitLayerResult:
    hidden_state: np.ndarray
    logits: np.ndarray
    solver: AndersonResult


@dataclass(frozen=True)
class EquilibriumDiagnostics:
    recurrent_spectral_norm: float
    local_jacobian_norm: float
    local_jacobian_spectral_radius: float
    contraction_margin: float


@dataclass(frozen=True)
class ImplicitAdjointResult:
    adjoint: np.ndarray
    linear_residual_norm: float
    system_condition_number: float


@dataclass(frozen=True)
class ImplicitInputGradientResult:
    input_gradient: np.ndarray
    adjoint: ImplicitAdjointResult


@dataclass(frozen=True)
class ImplicitParameterGradientResult:
    recurrent_gradient: np.ndarray
    input_weight_gradient: np.ndarray
    bias_gradient: np.ndarray
    adjoint: ImplicitAdjointResult


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
    rtol: float = 0.0,
    max_iter: int = 100,
    residual_guard: bool = False,
    guard_factor: float = 1.25,
    restart_on_rejection: bool = True,
    require_convergence: bool = False,
) -> ImplicitLayerResult:
    """Solve a small implicit tanh layer.

    The fixed point has the same form used in lightweight implicit neural
    layers:

    ``h = tanh(W_h h + W_x x + b)``

    Anderson acceleration is used only for the forward equilibrium solve. The
    Set ``residual_guard=True`` for maps that are only locally contractive.
    ``require_convergence`` is useful in inference paths where returning an
    unconverged hidden state would be harder to detect than a failed solve.
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
        rtol=rtol,
        max_iter=max_iter,
        residual_guard=residual_guard,
        guard_factor=guard_factor,
        restart_on_rejection=restart_on_rejection,
    )
    if require_convergence and not solver.converged:
        raise RuntimeError(
            "implicit tanh layer did not converge: "
            f"residual={solver.residual_norm:.3e} after {solver.iterations} iterations"
        )
    hidden_state = solver.solution
    logits = _readout(hidden_state, readout_weight, readout_bias)
    return ImplicitLayerResult(hidden_state=hidden_state, logits=logits, solver=solver)


def tanh_equilibrium_diagnostics(
    hidden_state,
    recurrent_weight,
) -> EquilibriumDiagnostics:
    """Estimate local stability of ``h = tanh(W_h h + b)`` around a solution."""
    hidden = _as_vector(hidden_state, "hidden_state")
    recurrent = _as_matrix(recurrent_weight, "recurrent_weight")
    if recurrent.shape != (hidden.shape[0], hidden.shape[0]):
        raise ValueError("recurrent_weight must have shape (hidden_dim, hidden_dim)")

    derivative = 1.0 - hidden**2
    local_jacobian = derivative[:, None] * recurrent
    recurrent_norm = float(np.linalg.svd(recurrent, compute_uv=False)[0])
    local_norm = float(np.linalg.svd(local_jacobian, compute_uv=False)[0])
    local_radius = float(np.max(np.abs(np.linalg.eigvals(local_jacobian))))
    return EquilibriumDiagnostics(
        recurrent_spectral_norm=recurrent_norm,
        local_jacobian_norm=local_norm,
        local_jacobian_spectral_radius=local_radius,
        contraction_margin=float(1.0 - local_norm),
    )


def solve_tanh_adjoint(
    hidden_state,
    recurrent_weight,
    hidden_gradient,
) -> ImplicitAdjointResult:
    """Solve the implicit backward system for a converged tanh layer.

    For ``h = f(h, x)``, implicit differentiation requires the adjoint
    ``v`` satisfying ``(I - J_f(h)^T) v = dL/dh``. This avoids unrolling the
    forward iterations and makes the conditioning of the backward solve
    directly observable.
    """
    hidden = _as_vector(hidden_state, "hidden_state")
    recurrent = _as_matrix(recurrent_weight, "recurrent_weight")
    gradient = _as_vector(hidden_gradient, "hidden_gradient")
    hidden_dim = hidden.shape[0]
    if recurrent.shape != (hidden_dim, hidden_dim):
        raise ValueError("recurrent_weight must have shape (hidden_dim, hidden_dim)")
    if gradient.shape != hidden.shape:
        raise ValueError("hidden_gradient must have length hidden_dim")

    derivative = 1.0 - hidden**2
    jacobian = derivative[:, None] * recurrent
    system = np.eye(hidden_dim) - jacobian.T
    try:
        adjoint = np.linalg.solve(system, gradient)
    except np.linalg.LinAlgError:
        adjoint = np.linalg.lstsq(system, gradient, rcond=None)[0]
    residual = system @ adjoint - gradient
    return ImplicitAdjointResult(
        adjoint=adjoint,
        linear_residual_norm=float(np.linalg.norm(residual)),
        system_condition_number=float(np.linalg.cond(system)),
    )


def tanh_implicit_input_gradient(
    hidden_state,
    recurrent_weight,
    input_weight,
    hidden_gradient,
) -> ImplicitInputGradientResult:
    """Differentiate a converged tanh equilibrium with respect to its input."""
    hidden = _as_vector(hidden_state, "hidden_state")
    input_matrix = _as_matrix(input_weight, "input_weight")
    if input_matrix.shape[0] != hidden.shape[0]:
        raise ValueError("input_weight must have shape (hidden_dim, input_dim)")

    adjoint = solve_tanh_adjoint(hidden, recurrent_weight, hidden_gradient)
    activation_derivative = 1.0 - hidden**2
    input_gradient = input_matrix.T @ (activation_derivative * adjoint.adjoint)
    return ImplicitInputGradientResult(input_gradient=input_gradient, adjoint=adjoint)


def tanh_implicit_parameter_gradients(
    hidden_state,
    input_vector,
    recurrent_weight,
    input_weight,
    hidden_gradient,
) -> ImplicitParameterGradientResult:
    """Differentiate a converged tanh equilibrium with respect to its parameters."""

    hidden = _as_vector(hidden_state, "hidden_state")
    inputs = _as_vector(input_vector, "input_vector")
    recurrent = _as_matrix(recurrent_weight, "recurrent_weight")
    input_matrix = _as_matrix(input_weight, "input_weight")
    hidden_dim = hidden.shape[0]
    if recurrent.shape != (hidden_dim, hidden_dim):
        raise ValueError("recurrent_weight must have shape (hidden_dim, hidden_dim)")
    if input_matrix.shape != (hidden_dim, inputs.shape[0]):
        raise ValueError("input_weight must have shape (hidden_dim, input_dim)")

    adjoint = solve_tanh_adjoint(hidden, recurrent, hidden_gradient)
    preactivation_gradient = (1.0 - hidden**2) * adjoint.adjoint
    return ImplicitParameterGradientResult(
        recurrent_gradient=np.outer(preactivation_gradient, hidden),
        input_weight_gradient=np.outer(preactivation_gradient, inputs),
        bias_gradient=preactivation_gradient,
        adjoint=adjoint,
    )


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
