"""Compare solver memory settings on equilibrium features."""

from anderson_acceleration import (
    make_equilibrium_weights,
    make_two_moons,
    solver_memory_sweep,
)


def main() -> None:
    inputs, _ = make_two_moons(n_samples=32, noise=0.04, seed=8)
    weights = make_equilibrium_weights(
        input_dim=2,
        hidden_dim=16,
        recurrent_scale=0.72,
        seed=5,
    )
    rows = solver_memory_sweep(inputs, weights, memories=(0, 1, 3, 5), max_iter=80)

    print("memory | convergence | mean iter | max residual")
    for row in rows:
        print(
            f"{row.memory:>6} | {row.convergence_rate:>11.3f} | "
            f"{row.mean_iterations:>9.2f} | {row.max_residual:.2e}"
        )


if __name__ == "__main__":
    main()
