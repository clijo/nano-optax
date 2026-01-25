# Solvers

All solvers in `nano-optax` follow a **Scipy-like interface**. The pattern is:

1.  Instantiate a **Solver** object (e.g., `SGD`) with hyperparameters.
2.  Call the `.minimize()` method with your loss function and parameters.

```python
solver = SGD(step_size=0.01)
result = solver.minimize(loss_fun, params, data)
```

To use stateful schedules, pass `schedule_state` and a schedule function that returns `(lr, new_state)`.

## Gradient Descent

::: nano_optax.gd.GD

## Stochastic Gradient Descent

::: nano_optax.sgd.SGD

## Proximal Gradient Descent

::: nano_optax.prox_gd.ProxGD