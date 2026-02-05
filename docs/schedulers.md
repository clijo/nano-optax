# Schedulers

Learning-rate schedules in `nano-optax` are **pure functions of step**. You
can pass any callable `schedule(step) -> lr` to a solver, or use the helpers
below.

## Quick Usage

```python
from nano_optax import sgd, step_lr

schedule = step_lr(base_lr=0.1, step_size=1000, gamma=0.5)
result = sgd(fun, init_params, data, lr=schedule, batch_size=16)
```

`step_lr` **counts minibatch steps**, not epochs. If you want to decay every
`N` epochs, set `step_size = N * batches_per_epoch`.

## Built-in Schedulers

- `constant_lr`: fixed learning rate.
- `lambda_lr`: user-defined schedule function.
- `step_lr`: multiplicative decay every `step_size` steps.

## Lambda Example

```python
import jax.numpy as jnp
from nano_optax import lambda_lr

schedule = lambda_lr(lambda step: jnp.exp(-0.001 * step))
```

## Stateful Schedule Example

If you need schedules that depend on previous values, pass a stateful schedule
function `(step, state) -> (lr, new_state)` and an initial `schedule_state`:

```python
import jax.numpy as jnp

def adaptive_schedule(step, state):
    lr = state["lr"]
    new_lr = lr * jnp.where(step % 100 == 0, 0.5, 1.0)
    return lr, {"lr": new_lr}
```

Use it by passing `schedule_state` to the solver.

## API

::: nano_optax.schedulers.constant_lr
::: nano_optax.schedulers.lambda_lr
::: nano_optax.schedulers.step_lr
::: nano_optax.schedulers.as_schedule
