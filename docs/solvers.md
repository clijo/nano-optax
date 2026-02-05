# Solvers

All solvers in `nano-optax` are **pure functions**. Each solver takes an
objective `f(params, *data)` and returns an `OptResult` with final parameters,
final objective value, and a per-epoch trace.

```python
from nano_optax import gd

result = gd(fun, init_params, data, lr=1e-2, max_epochs=100)
```

If you want to use a **stateful schedule**, pass a schedule function with
signature `(step, state) -> (lr, new_state)` and provide `schedule_state`.

## Gradient Descent

::: nano_optax.gd.gd

## Stochastic Gradient Descent

::: nano_optax.sgd.sgd

## Proximal Gradient Descent

::: nano_optax.prox_gd.prox_gd

### Proximal operators
`prox_gd` expects an uncurried prox operator $(x,\eta)\mapsto \operatorname{prox}_{\eta g}(x)$.
Two helpers are included:

```python
from nano_optax import prox_l1, prox_l2

prox_l1_op = prox_l1(reg=1.0)
prox_l2_op = prox_l2(reg=0.1)
```

::: nano_optax.prox_gd.prox_l1
::: nano_optax.prox_gd.prox_l2

## Accelerated Proximal Gradient Descent (FISTA)

::: nano_optax.apgd.apgd
