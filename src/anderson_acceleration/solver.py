from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Sequence, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]
FixedPointMap = Callable[[np.ndarray], ArrayLike]


@dataclass(frozen=True)
class AndersonResult:
    """Result returned by :func:`anderson_accelerate`."""

    solution: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    residual_history: tuple[float, ...]


def anderson_accelerate(
    fixed_point: FixedPointMap,
    x0: ArrayLike,
    *,
    memory: int = 5,
    beta: float = 1.0,
    regularization: float = 1e-12,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> AndersonResult:
    """Solve ``x = fixed_point(x)`` with Anderson acceleration.

    Parameters
    ----------
    fixed_point:
        Function that evaluates one ordinary fixed-point update.
    x0:
        Initial iterate. Scalars, vectors, and small dense arrays are accepted.
    memory:
        Number of previous residuals used in the least-squares mixing problem.
        ``memory=0`` falls back to ordinary fixed-point iteration.
    beta:
        Damping applied to the accelerated update. ``1.0`` uses the full
        Anderson step, while smaller positive values blend toward the current
        iterates.
    regularization:
        Small diagonal stabilizer for the residual Gram matrix.
    tol:
        Stop when ``||fixed_point(x) - x||_2 <= tol``.
    max_iter:
        Maximum number of fixed-point evaluations.
    """

    _validate_options(memory, beta, regularization, tol, max_iter)

    x = _as_float_array(x0, "x0")
    original_shape = x.shape
    x_flat = x.reshape(-1)

    x_history: list[np.ndarray] = []
    g_history: list[np.ndarray] = []
    f_history: list[np.ndarray] = []
    residual_history: list[float] = []

    for iteration in range(1, max_iter + 1):
        g = _evaluate_fixed_point(fixed_point, x_flat.reshape(original_shape), original_shape)
        g_flat = g.reshape(-1)
        residual = g_flat - x_flat
        residual_norm = float(np.linalg.norm(residual))
        residual_history.append(residual_norm)

        if residual_norm <= tol:
            return AndersonResult(
                solution=g.copy(),
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                residual_history=tuple(residual_history),
            )

        x_history.append(x_flat.copy())
        g_history.append(g_flat.copy())
        f_history.append(residual.copy())

        window = min(memory + 1, len(f_history))
        if memory == 0 or window == 1:
            x_flat = g_flat
        else:
            alpha = _mixing_coefficients(f_history[-window:], regularization)
            recent_x = np.column_stack(x_history[-window:])
            recent_g = np.column_stack(g_history[-window:])
            x_flat = (1.0 - beta) * (recent_x @ alpha) + beta * (recent_g @ alpha)

            if not np.all(np.isfinite(x_flat)):
                x_flat = g_flat

    return AndersonResult(
        solution=x_flat.reshape(original_shape).copy(),
        converged=False,
        iterations=max_iter,
        residual_norm=residual_history[-1],
        residual_history=tuple(residual_history),
    )


def _mixing_coefficients(residuals: Sequence[np.ndarray], regularization: float) -> np.ndarray:
    residual_matrix = np.column_stack(residuals)
    count = residual_matrix.shape[1]

    gram = residual_matrix.T @ residual_matrix
    if regularization:
        gram = gram + regularization * np.eye(count)

    augmented = np.empty((count + 1, count + 1), dtype=float)
    augmented[:count, :count] = gram
    augmented[:count, count] = 1.0
    augmented[count, :count] = 1.0
    augmented[count, count] = 0.0

    rhs = np.zeros(count + 1, dtype=float)
    rhs[count] = 1.0

    try:
        return np.linalg.solve(augmented, rhs)[:count]
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(augmented, rhs, rcond=None)[0][:count]


def _evaluate_fixed_point(
    fixed_point: FixedPointMap,
    x: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    value = _as_float_array(fixed_point(x.copy()), "fixed_point(x)")

    if value.shape != expected_shape:
        raise ValueError(
            "fixed_point must return the same shape as x0: "
            f"expected {expected_shape}, got {value.shape}"
        )

    return value


def _as_float_array(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_options(
    memory: int,
    beta: float,
    regularization: float,
    tol: float,
    max_iter: int,
) -> None:
    if not isinstance(memory, Integral):
        raise ValueError("memory must be an integer")
    for name, value in (
        ("beta", beta),
        ("regularization", regularization),
        ("tol", tol),
    ):
        if not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if memory < 0:
        raise ValueError("memory must be non-negative")
    if not 0.0 < beta <= 1.0:
        raise ValueError("beta must be in the interval (0, 1]")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
