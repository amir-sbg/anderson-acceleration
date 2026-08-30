import numpy as np
import pytest

from anderson_acceleration import anderson_accelerate, solve_tanh_equilibrium
from anderson_acceleration.experiments import (
    equilibrium_features,
    fit_softmax_readout,
    make_equilibrium_weights,
    make_two_moons,
    readout_accuracy,
)


def test_cosine_fixed_point_needs_fewer_steps_than_picard() -> None:
    plain = anderson_accelerate(
        np.cos,
        np.array([1.0]),
        memory=0,
        tol=1e-10,
        max_iter=100,
    )
    result = anderson_accelerate(
        np.cos,
        np.array([1.0]),
        memory=4,
        tol=1e-10,
        max_iter=50,
    )

    assert plain.converged
    assert result.converged
    assert result.iterations < plain.iterations
    assert result.solution[0] == pytest.approx(0.7390851332, abs=1e-8)


def test_linear_fixed_point_matches_closed_form_solution() -> None:
    matrix = np.array([[0.55, 0.10], [0.05, 0.40]])
    offset = np.array([1.0, -0.5])
    expected = np.linalg.solve(np.eye(2) - matrix, offset)

    result = anderson_accelerate(
        lambda x: matrix @ x + offset,
        np.zeros(2),
        memory=3,
        tol=1e-11,
        max_iter=20,
    )

    assert result.converged
    np.testing.assert_allclose(result.solution, expected, atol=1e-8)


def test_memory_zero_runs_plain_fixed_point_iteration() -> None:
    result = anderson_accelerate(
        lambda x: 0.5 * x + 1.0,
        np.array([0.0]),
        memory=0,
        tol=1e-12,
        max_iter=5,
    )

    assert not result.converged
    assert result.iterations == 5
    assert result.solution[0] == pytest.approx(1.9375)
    assert len(result.residual_history) == 5


def test_preserves_input_shape() -> None:
    target = np.array([[1.0, -2.0], [0.5, 3.0]])

    result = anderson_accelerate(
        lambda x: 0.25 * x + 0.75 * target,
        np.zeros_like(target),
        memory=2,
        tol=1e-10,
        max_iter=20,
    )

    assert result.solution.shape == target.shape
    np.testing.assert_allclose(result.solution, target, atol=1e-8)


def test_tanh_equilibrium_layer_solves_fixed_point() -> None:
    x = np.array([0.5, -0.25])
    recurrent = np.array([[0.20, -0.05, 0.02], [0.03, 0.18, 0.00], [0.01, -0.04, 0.15]])
    input_weight = np.array([[0.4, -0.1], [0.2, 0.3], [-0.3, 0.2]])
    bias = np.array([0.01, -0.02, 0.03])
    readout = np.array([[0.6, -0.2, 0.1], [-0.1, 0.2, 0.4]])

    result = solve_tanh_equilibrium(
        x,
        recurrent,
        input_weight,
        bias,
        readout_weight=readout,
        memory=3,
        tol=1e-10,
        max_iter=40,
    )

    expected_hidden = np.tanh(recurrent @ result.hidden_state + input_weight @ x + bias)
    assert result.solver.converged
    np.testing.assert_allclose(result.hidden_state, expected_hidden, atol=1e-8)
    assert result.logits.shape == (2,)


def test_tanh_equilibrium_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="input_weight"):
        solve_tanh_equilibrium(
            np.array([1.0, 2.0]),
            np.eye(3) * 0.1,
            np.ones((2, 2)),
            np.zeros(3),
        )


def test_rejects_shape_changing_maps() -> None:
    with pytest.raises(ValueError, match="same shape"):
        anderson_accelerate(
            lambda x: np.array([1.0, 2.0]),
            np.array([0.0]),
            max_iter=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"memory": -1}, "memory"),
        ({"beta": 0.0}, "beta"),
        ({"regularization": -1.0}, "regularization"),
        ({"tol": 0.0}, "tol"),
        ({"max_iter": 0}, "max_iter"),
        ({"memory": 1.5}, "memory"),
        ({"beta": np.nan}, "beta"),
        ({"regularization": np.inf}, "regularization"),
        ({"tol": np.inf}, "tol"),
    ],
)
def test_rejects_invalid_solver_options(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        anderson_accelerate(lambda x: x, np.array([1.0]), **kwargs)


def test_two_moons_generator_is_deterministic() -> None:
    first_x, first_y = make_two_moons(n_samples=20, noise=0.03, seed=9)
    second_x, second_y = make_two_moons(n_samples=20, noise=0.03, seed=9)

    np.testing.assert_allclose(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)
    assert first_x.shape == (20, 2)
    assert set(first_y.tolist()) == {0, 1}


def test_equilibrium_features_return_solver_diagnostics() -> None:
    inputs, _ = make_two_moons(n_samples=8, noise=0.0, seed=2)
    weights = make_equilibrium_weights(input_dim=2, hidden_dim=6, seed=4)
    result = equilibrium_features(inputs, weights, memory=3, tol=1e-8, max_iter=60)

    assert result.hidden_states.shape == (8, 6)
    assert len(result.iterations) == 8
    assert result.convergence_rate == 1.0
    assert max(result.residuals) < 1e-6


def test_softmax_readout_learns_simple_boundary() -> None:
    features = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [1.0, -0.1],
            [2.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    result = fit_softmax_readout(
        features,
        labels,
        learning_rate=0.2,
        epochs=120,
        weight_decay=0.0,
        seed=1,
    )

    assert result.train_accuracy == 1.0
    assert result.loss_history[-1] < result.loss_history[0]
    assert readout_accuracy(features, labels, result.weights, result.bias) == 1.0
