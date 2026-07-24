# Anderson Acceleration

A small NumPy implementation of Anderson acceleration for fixed-point iterations.

The goal of this repository is to keep the method easy to read and easy to reuse. It is meant for experiments where an update already has the form:

```text
x_next = g(x)
```

Anderson acceleration keeps a short history of recent residuals, solves a small least-squares mixing problem, and proposes a better next iterate. In practice this can reduce the number of iterations for contractive nonlinear maps, self-consistency equations, implicit models, and other fixed-point style routines.

## What is included

- Damped Anderson acceleration with configurable memory.
- Plain fixed-point iteration fallback with `memory=0`.
- Dense NumPy implementation with no heavy solver framework.
- Shape checks and finite-value validation.
- A small result object with convergence status and residual history.
- Tests for scalar, vector, and matrix-shaped fixed-point problems.

## Installation

```bash
git clone https://github.com/amir-sbg/anderson-acceleration.git
cd anderson-acceleration

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Quick example

```python
import numpy as np

from anderson_acceleration import anderson_accelerate

result = anderson_accelerate(
    np.cos,
    np.array([1.0]),
    memory=4,
    tol=1e-10,
    max_iter=100,
)

print(result.solution)
print(result.iterations)
```

Run the included comparison example:

```bash
python examples/cosine_fixed_point.py
```

Example output:

```text
Solving x = cos(x)
Picard:   x=0.7390851332, iterations=58
Anderson: x=0.7390851332, iterations=32
```

The exact iteration count can vary slightly with numerical libraries and regularization settings, but the accelerated run should reach the same fixed point in fewer steps for this example.

## API

```python
anderson_accelerate(
    fixed_point,
    x0,
    memory=5,
    beta=1.0,
    regularization=1e-12,
    tol=1e-8,
    max_iter=100,
)
```

Parameters:

- `fixed_point`: function that evaluates one update `g(x)`.
- `x0`: initial scalar, vector, or dense array.
- `memory`: number of previous residuals used for mixing.
- `beta`: damping factor for the accelerated update.
- `regularization`: diagonal stabilizer for the least-squares system.
- `tol`: convergence threshold for `||g(x) - x||_2`.
- `max_iter`: maximum number of fixed-point evaluations.

The function returns `AndersonResult`:

```python
AndersonResult(
    solution,
    converged,
    iterations,
    residual_norm,
    residual_history,
)
```

## Project structure

```text
.
├── examples/
│   └── cosine_fixed_point.py
├── src/
│   └── anderson_acceleration/
│       ├── __init__.py
│       └── solver.py
├── tests/
│   └── test_solver.py
├── pyproject.toml
└── README.md
```

## Suggested repository metadata

Title: `Anderson Acceleration`

About: `NumPy implementation of damped Anderson acceleration for fixed-point iteration, with tests and a simple comparison example.`

