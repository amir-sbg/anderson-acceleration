"""Compare ordinary fixed-point iteration with Anderson acceleration."""

import numpy as np

from anderson_acceleration import anderson_accelerate


def main() -> None:
    x0 = np.array([1.0])

    picard = anderson_accelerate(
        np.cos,
        x0,
        memory=0,
        tol=1e-10,
        max_iter=100,
    )
    accelerated = anderson_accelerate(
        np.cos,
        x0,
        memory=4,
        tol=1e-10,
        max_iter=100,
    )

    print("Solving x = cos(x)")
    print(f"Picard:   x={picard.solution[0]:.10f}, iterations={picard.iterations}")
    print(f"Anderson: x={accelerated.solution[0]:.10f}, iterations={accelerated.iterations}")


if __name__ == "__main__":
    main()

