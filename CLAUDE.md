# Simplegrad — Project Guide

## Project Overview

Simplegrad is a deep learning framework built around a clean, Pythonic API. It provides numpy-backed tensors with automatic differentiation, a full set of neural network primitives, and tight numpy interoperability. The project also includes **SimpleBoard** — a web application for experiment tracking and visualization.

The framework has an educational objective: users should be able to read the source code and understand how deep learning works from the ground up.

## Repository Structure

```
simplegrad/               # Main package
    core/                 # Tensor class and autograd engine
    functions/            # Differentiable math, activations, losses, pooling, conv
    nn/                   # Neural network layers (Linear, Conv2d, Dropout, etc.)
    optimizers/           # SGD, Adam
    schedulers/           # Learning rate schedulers
    track/                # Experiment tracking (Tracker, SQLite backend)
    visual/               # Inline visualization (computation graphs, training plots)
    simpleboard/          # Web app for experiment tracking dashboard
    dtypes.py             # Supported data types

tests/                    # Test suite (pytest)
docs/                     # MkDocs documentation source
    api/                  # Auto-generated API reference pages
examples/                 # Usage examples
experiments/              # Default output directory for tracked runs
assets/                   # Static assets
build_web.py              # Builds the SimpleBoard frontend
pyproject.toml            # Project metadata, dependencies, tool config
mkdocs.yml                # Documentation site configuration
```

## Code Style

### Language

All code, comments, variable names, docstrings, and commit messages must be written in **English**. No other languages anywhere in the repository. This is a strict rule.

### Comments

- Do not use separator lines in comments (no `===`, `---`, `———`, `***`, or similar decorators).
- Keep comments concise and only where the logic is not self-evident.

### Docstrings

Every function and method that is part of the public API (callable by a user of the framework) must have a docstring in **Google style**.

Docstrings should be thorough. Since simplegrad has educational goals, explain not just what a function does but also the mathematical or conceptual meaning where relevant. Users reading the source should come away understanding the underlying principles.

Example of the expected style:

```python
def relu(x: Tensor) -> Tensor:
    """Applies the Rectified Linear Unit activation function element-wise.

    Computes max(0, x) for each element. ReLU is the most commonly used
    activation function in deep learning because it avoids the vanishing
    gradient problem present in sigmoid and tanh for large inputs, while
    remaining computationally cheap.

    Args:
        x: Input tensor of any shape.

    Returns:
        A tensor of the same shape as x, with all negative values set to zero.
        Gradients flow through only the positive elements during backpropagation.

    Example:
        >>> x = Tensor([-1.0, 0.0, 2.0])
        >>> relu(x)
        Tensor([0.0, 0.0, 2.0])
    """
```

Internal helper methods (prefixed with `_`) do not require docstrings unless their logic is non-obvious.

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

Tests live in `tests/`. Use `pytest tests/test_nn.py` or `pytest tests/test_functions.py` to run a specific file.

### Documentation

Preview docs locally:

```bash
mkdocs serve
```

Docs are deployed automatically to GitHub Pages on every push to `main` via the `docs.yml` workflow. To deploy manually:

```bash
mkdocs gh-deploy --force
```

### SimpleBoard Frontend

The SimpleBoard web app has a compiled frontend. After making changes to `simplegrad/simpleboard/app/`, rebuild it before committing:

```bash
python build_web.py
```

## Internals

### Numpy Backend

All tensor data is stored as a numpy `ndarray` in `tensor.values`. Every operation ultimately calls numpy under the hood. Do not introduce non-numpy code paths (e.g. plain Python lists or math) inside the core or functions packages — always operate on `.values` directly.

### Adding a New Differentiable Operation

Every differentiable function follows the same pattern. Use this as a template:

```python
def my_op(x: Tensor) -> Tensor:
    out = Tensor(np.some_numpy_op(x.values))
    out.prev = {x}
    out.oper = "MyOp"          # short label shown in computation graph
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _my_op_backward(x, out)
    return out


def _my_op_backward(x: Tensor, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += out.grad * <local_gradient_expression>
```

Key rules:
- Always use `_should_compute_grad(x)` to set `out.comp_grad` — never set it to a hardcoded boolean.
- Always check `x.comp_grad` inside the backward function before writing to `x.grad`.
- Always use `+=` when accumulating into `x.grad` (a tensor can receive gradients from multiple consumers).
- The `oper` string must not contain spaces (it is used as a graph node label).
- Import `_should_compute_grad` from `simplegrad.core.tensor`.
