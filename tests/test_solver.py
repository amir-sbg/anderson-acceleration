import numpy as np
import pytest

from anderson_acceleration import anderson_accelerate


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
    ],
)
def test_rejects_invalid_solver_options(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        anderson_accelerate(lambda x: x, np.array([1.0]), **kwargs)
