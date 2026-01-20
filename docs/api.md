# API Reference

All solvers in `nano-optax` follow a **Scipy-like interface**. The pattern is:

1.  Instantiate a **Solver** object (e.g., `SGD`) with hyperparameters.
2.  Call the `.minimize()` method with your loss function and parameters.

```python
solver = SGD(step_size=0.01)
result = solver.minimize(loss_fun, params, data)
```

## Gradient Descent

::: nano_optax.gd.GD
    options:
      show_root_heading: false
      show_source: true

## Stochastic Gradient Descent

::: nano_optax.sgd.SGD
    options:
      show_root_heading: false
      show_source: true

