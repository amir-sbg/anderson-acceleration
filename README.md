# Anderson Acceleration for Implicit ML Layers

A compact NumPy implementation of Anderson acceleration for fixed-point maps, with experiments that connect the solver to implicit neural layers, equilibrium features, and fixed-point optimization.

Given an update

```text
x_next = g(x)
```

the solver stores recent residuals `g(x) - x`, solves a small constrained least-squares problem, and mixes recent iterates. `memory=0` gives ordinary fixed-point iteration. Damping, Tikhonov regularization, and an optional residual guard make the numerical trade-offs explicit.

## Included

- scalar, vector, and matrix-shaped fixed-point solves
- convergence and residual-history diagnostics
- an implicit tanh layer of the form `h = tanh(W_h h + W_x x + b)`
- local Jacobian norm, spectral radius, and contraction-margin checks
- a two-moons classifier using converged hidden states as features
- train-statistic feature standardization and a softmax readout
- a ridge-regression gradient fixed-point experiment
- memory sweeps for comparing convergence cost and stability

This is a numerical experiment library rather than a full training framework. The focus is the forward equilibrium solve and the behavior of acceleration around it.

## Setup

```bash
git clone https://github.com/amir-sbg/anderson-acceleration.git
cd anderson-acceleration
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Quick start

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

print(result.solution, result.iterations)
```

Run the examples:

```bash
python examples/cosine_fixed_point.py
python examples/implicit_tanh_layer.py
python examples/equilibrium_classifier.py
python examples/memory_sweep.py
```

The classifier compares raw two-moons inputs with equilibrium hidden features. The memory sweep keeps the data and weights fixed while changing only the solver history length.

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
    residual_guard=False,
    guard_factor=1.25,
)
```

The returned `AndersonResult` contains the solution, convergence flag, iteration count, final residual, and residual history. `residual_diagnostics` summarizes reduction, best iteration, monotonicity, and stagnation without rerunning the map.

The ML helpers expose `solve_tanh_equilibrium`, `tanh_equilibrium_diagnostics`, `equilibrium_features`, `standardize_features`, `fit_softmax_readout`, and `fit_ridge_fixed_point`.

## Project layout

```text
examples/
├── cosine_fixed_point.py
├── equilibrium_classifier.py
├── implicit_tanh_layer.py
└── memory_sweep.py
src/anderson_acceleration/
├── solver.py       Anderson iteration and diagnostics
├── ml.py           implicit-layer helpers and stability checks
└── experiments.py  datasets, readouts, and sweeps
tests/test_solver.py
```
