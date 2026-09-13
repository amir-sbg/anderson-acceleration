from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import numpy as np

from .ml import solve_tanh_equilibrium
from .solver import AndersonResult, anderson_accelerate, residual_diagnostics


@dataclass(frozen=True)
class EquilibriumWeights:
    recurrent_weight: np.ndarray
    input_weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class EquilibriumFeatureResult:
    hidden_states: np.ndarray
    iterations: tuple[int, ...]
    residuals: tuple[float, ...]
    convergence_rate: float


@dataclass(frozen=True)
class ReadoutResult:
    weights: np.ndarray
    bias: np.ndarray
    loss_history: tuple[float, ...]
    train_accuracy: float


@dataclass(frozen=True)
class SolverSweepRow:
    memory: int
    convergence_rate: float
    mean_iterations: float
    median_iterations: float
    max_residual: float
    hidden_state_std: float


@dataclass(frozen=True)
class RidgeFixedPointResult:
    weights: np.ndarray
    bias: float
    solver: AndersonResult
    residual_report: dict[str, float | int | bool]
    learning_rate: float


def make_two_moons(
    n_samples: int = 160,
    noise: float = 0.08,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 4:
        raise ValueError("n_samples must be at least 4")
    if noise < 0:
        raise ValueError("noise must not be negative")

    generator = np.random.default_rng(seed)
    outer_count = (n_samples + 1) // 2
    inner_count = n_samples // 2
    outer_angles = np.linspace(0.0, np.pi, outer_count)
    inner_angles = np.linspace(0.0, np.pi, inner_count)

    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([1.0 - np.cos(inner_angles), 0.5 - np.sin(inner_angles)])
    inputs = np.vstack([outer, inner])
    labels = np.concatenate([np.zeros(outer_count, dtype=int), np.ones(inner_count, dtype=int)])
    inputs = inputs + generator.normal(0.0, noise, size=inputs.shape)

    order = generator.permutation(n_samples)
    return inputs[order], labels[order]


def make_equilibrium_weights(
    input_dim: int,
    hidden_dim: int = 24,
    recurrent_scale: float = 0.65,
    input_scale: float = 0.9,
    seed: int = 0,
) -> EquilibriumWeights:
    if input_dim < 1 or hidden_dim < 1:
        raise ValueError("input_dim and hidden_dim must be positive")
    if recurrent_scale <= 0 or recurrent_scale >= 1:
        raise ValueError("recurrent_scale should stay in (0, 1) for a contraction-like map")
    if input_scale <= 0:
        raise ValueError("input_scale must be positive")

    generator = np.random.default_rng(seed)
    recurrent = generator.normal(0.0, 1.0, size=(hidden_dim, hidden_dim))
    spectral_norm = np.linalg.svd(recurrent, compute_uv=False)[0]
    recurrent = recurrent / max(spectral_norm, 1e-8) * recurrent_scale
    input_weight = generator.normal(0.0, input_scale / np.sqrt(input_dim), size=(hidden_dim, input_dim))
    bias = generator.normal(0.0, 0.05, size=hidden_dim)
    return EquilibriumWeights(
        recurrent_weight=recurrent,
        input_weight=input_weight,
        bias=bias,
    )


def equilibrium_features(
    inputs: np.ndarray,
    weights: EquilibriumWeights,
    *,
    memory: int = 5,
    beta: float = 1.0,
    tol: float = 1e-7,
    max_iter: int = 80,
) -> EquilibriumFeatureResult:
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 2 or 0 in inputs.shape:
        raise ValueError("inputs must be a non-empty two-dimensional matrix")
    _validate_equilibrium_weights(weights, input_dim=inputs.shape[1])

    hidden_states = []
    iterations = []
    residuals = []
    converged = 0
    for row in inputs:
        result = solve_tanh_equilibrium(
            row,
            weights.recurrent_weight,
            weights.input_weight,
            weights.bias,
            memory=memory,
            beta=beta,
            tol=tol,
            max_iter=max_iter,
        )
        hidden_states.append(result.hidden_state)
        iterations.append(result.solver.iterations)
        residuals.append(result.solver.residual_norm)
        converged += int(result.solver.converged)

    return EquilibriumFeatureResult(
        hidden_states=np.vstack(hidden_states),
        iterations=tuple(int(value) for value in iterations),
        residuals=tuple(float(value) for value in residuals),
        convergence_rate=float(converged / len(inputs)),
    )


def solver_memory_sweep(
    inputs: np.ndarray,
    weights: EquilibriumWeights,
    *,
    memories: tuple[int, ...] = (0, 1, 3, 5),
    beta: float = 1.0,
    tol: float = 1e-7,
    max_iter: int = 80,
) -> tuple[SolverSweepRow, ...]:
    if not memories:
        raise ValueError("memories must contain at least one value")
    if any(not isinstance(memory, Integral) or memory < 0 for memory in memories):
        raise ValueError("memories must be non-negative integers")

    rows = []
    for memory in memories:
        result = equilibrium_features(
            inputs,
            weights,
            memory=int(memory),
            beta=beta,
            tol=tol,
            max_iter=max_iter,
        )
        rows.append(
            SolverSweepRow(
                memory=int(memory),
                convergence_rate=result.convergence_rate,
                mean_iterations=float(np.mean(result.iterations)),
                median_iterations=float(np.median(result.iterations)),
                max_residual=float(max(result.residuals)),
                hidden_state_std=float(result.hidden_states.std()),
            )
        )
    return tuple(rows)


def fit_ridge_fixed_point(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    l2_penalty: float = 1e-3,
    learning_rate: float | None = None,
    memory: int = 5,
    beta: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> RidgeFixedPointResult:
    x, y = _regression_arrays(features, targets)
    if l2_penalty < 0:
        raise ValueError("l2_penalty must not be negative")

    design = np.column_stack([x, np.ones(len(x), dtype=float)])
    penalty = np.ones(design.shape[1], dtype=float)
    penalty[-1] = 0.0
    if learning_rate is None:
        lipschitz = np.linalg.svd(design, compute_uv=False)[0] ** 2 / len(x)
        learning_rate = 0.9 / max(float(lipschitz + l2_penalty), 1e-12)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    def fixed_point(theta: np.ndarray) -> np.ndarray:
        residual = design @ theta - y
        gradient = design.T @ residual / len(x) + l2_penalty * penalty * theta
        return theta - learning_rate * gradient

    solver = anderson_accelerate(
        fixed_point,
        np.zeros(design.shape[1], dtype=float),
        memory=memory,
        beta=beta,
        tol=tol,
        max_iter=max_iter,
        residual_guard=True,
    )
    return RidgeFixedPointResult(
        weights=solver.solution[:-1].copy(),
        bias=float(solver.solution[-1]),
        solver=solver,
        residual_report=residual_diagnostics(solver.residual_history),
        learning_rate=float(learning_rate),
    )


def fit_softmax_readout(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: Optional[int] = None,
    learning_rate: float = 0.25,
    epochs: int = 350,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> ReadoutResult:
    x, y, classes = _classification_arrays(features, labels, num_classes)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if weight_decay < 0:
        raise ValueError("weight_decay must not be negative")

    generator = np.random.default_rng(seed)
    weights = generator.normal(0.0, 0.02, size=(x.shape[1], classes))
    bias = np.zeros(classes, dtype=float)
    targets = np.eye(classes)[y]
    loss_history = []

    for _ in range(epochs):
        logits = x @ weights + bias
        probabilities = _softmax(logits)
        loss = -np.mean(np.log(probabilities[np.arange(len(y)), y] + 1e-12))
        loss += 0.5 * weight_decay * float(np.sum(weights**2))
        loss_history.append(float(loss))

        gradient = (probabilities - targets) / len(y)
        weights -= learning_rate * (x.T @ gradient + weight_decay * weights)
        bias -= learning_rate * gradient.sum(axis=0)

    accuracy = readout_accuracy(x, y, weights, bias)
    return ReadoutResult(
        weights=weights,
        bias=bias,
        loss_history=tuple(loss_history),
        train_accuracy=accuracy,
    )


def readout_predict(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if weights.ndim != 2 or bias.ndim != 1:
        raise ValueError("weights must be a matrix and bias must be a vector")
    if features.shape[1] != weights.shape[0] or weights.shape[1] != bias.shape[0]:
        raise ValueError("readout shapes are inconsistent")
    return np.argmax(features @ weights + bias, axis=1)


def readout_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> float:
    predictions = readout_predict(features, weights, bias)
    labels = np.asarray(labels, dtype=int)
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    return float(np.mean(predictions == labels))


def _validate_equilibrium_weights(weights: EquilibriumWeights, input_dim: int) -> None:
    recurrent = np.asarray(weights.recurrent_weight, dtype=float)
    input_weight = np.asarray(weights.input_weight, dtype=float)
    bias = np.asarray(weights.bias, dtype=float)
    if recurrent.ndim != 2 or recurrent.shape[0] != recurrent.shape[1]:
        raise ValueError("recurrent_weight must be a square matrix")
    if input_weight.shape != (recurrent.shape[0], input_dim):
        raise ValueError("input_weight shape does not match hidden and input dimensions")
    if bias.shape != (recurrent.shape[0],):
        raise ValueError("bias length must match hidden dimension")
    if not (
        np.all(np.isfinite(recurrent))
        and np.all(np.isfinite(input_weight))
        and np.all(np.isfinite(bias))
    ):
        raise ValueError("equilibrium weights must contain only finite values")


def _classification_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: Optional[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or 0 in x.shape:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("labels must be one-dimensional and match features")
    if y.size == 0 or np.any(y < 0):
        raise ValueError("labels must be non-negative")
    classes = int(y.max() + 1) if num_classes is None else int(num_classes)
    if classes < 2:
        raise ValueError("num_classes must be at least 2")
    if np.any(y >= classes):
        raise ValueError("labels must be smaller than num_classes")
    return x, y, classes


def _regression_arrays(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or 0 in x.shape:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("targets must be one-dimensional and match features")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("features and targets must contain only finite values")
    return x, y


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
