# Anderson Acceleration for Implicit ML Layers

A compact NumPy implementation of Anderson acceleration for fixed-point iteration, with a small implicit neural-layer example.

The goal of this repository is to keep the method easy to read and easy to reuse for experiments where an update already has the form:

```text
x_next = g(x)
```

Anderson acceleration keeps a short history of recent residuals, solves a small least-squares mixing problem, and proposes a better next iterate. In ML terms, this is useful for studying equilibrium-style layers, self-consistency updates, implicit models, and fixed-point reasoning blocks where the forward pass solves for a stable hidden state instead of stacking a fixed number of layers.

## What is included

- Damped Anderson acceleration with configurable memory.
- Plain fixed-point iteration fallback with `memory=0`.
- Dense NumPy implementation with no heavy solver framework.
- Shape checks and finite-value validation.
- A small result object with convergence status and residual history.
- A NumPy implicit tanh layer helper for `h = tanh(W_h h + W_x x + b)`.
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

## ML-style implicit layer example

The package also includes a tiny equilibrium-layer helper:

```python
import numpy as np

from anderson_acceleration import solve_tanh_equilibrium

result = solve_tanh_equilibrium(
    input_vector=np.array([0.8, -0.4, 0.2]),
    recurrent_weight=0.2 * np.eye(4),
    input_weight=np.ones((4, 3)) * 0.1,
    bias=np.zeros(4),
    memory=4,
)

print(result.hidden_state)
print(result.solver.residual_norm)
```

Run the full example:

```bash
python examples/implicit_tanh_layer.py
```

This is not a training framework. It is a small numerical experiment that mirrors the forward equilibrium solve used in implicit/deep-equilibrium style models.

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
│   ├── cosine_fixed_point.py
│   └── implicit_tanh_layer.py
├── src/
│   └── anderson_acceleration/
│       ├── __init__.py
│       ├── ml.py
│       └── solver.py
├── tests/
│   └── test_solver.py
├── pyproject.toml
└── README.md
```
