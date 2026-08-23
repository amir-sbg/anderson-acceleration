"""Run a tiny implicit neural layer with Anderson acceleration."""

import numpy as np

from anderson_acceleration import solve_tanh_equilibrium


def main() -> None:
    x = np.array([0.8, -0.4, 0.2])
    recurrent = np.array(
        [
            [0.28, -0.08, 0.02, 0.00],
            [0.04, 0.22, 0.05, -0.03],
            [-0.02, 0.07, 0.25, 0.04],
            [0.03, 0.00, -0.06, 0.20],
        ]
    )
    input_weight = np.array(
        [
            [0.6, -0.2, 0.1],
            [-0.1, 0.4, 0.3],
            [0.2, 0.1, -0.5],
            [0.3, -0.3, 0.2],
        ]
    )
    bias = np.array([0.05, -0.02, 0.01, 0.03])
    readout = np.array([[0.7, -0.3, 0.2, 0.1], [-0.2, 0.4, 0.1, 0.5]])

    result = solve_tanh_equilibrium(
        x,
        recurrent,
        input_weight,
        bias,
        readout_weight=readout,
        memory=4,
        tol=1e-9,
        max_iter=50,
    )

    print("Implicit tanh layer")
    print(f"converged: {result.solver.converged}")
    print(f"iterations: {result.solver.iterations}")
    print(f"residual: {result.solver.residual_norm:.3e}")
    print(f"hidden: {np.round(result.hidden_state, 4)}")
    print(f"logits: {np.round(result.logits, 4)}")


if __name__ == "__main__":
    main()
