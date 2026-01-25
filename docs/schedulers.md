# Schedulers

Learning-rate schedulers define how the step size changes over time.
In `nano-optax`, schedulers are pure functions of `step` and optional state:

```python
lr = scheduler(step)
```

## Quick Usage

```python
from nano_optax import StepLR, SGD

scheduler = StepLR(base_lr=0.1, step_size=1000, gamma=0.5)
solver = SGD(step_size=scheduler)
```

`StepLR` **counts minibatch steps**, not epochs. If you want to decay every
`N` epochs, set `step_size = N * batches_per_epoch`.

## Built-in Schedulers

- `ConstantLR`: fixed learning rate.
- `LambdaLR`: user-defined schedule function.
- `StepLR`: multiplicative decay every `step_size` steps.

## LambdaLR Example

```python
import jax.numpy as jnp
from nano_optax import LambdaLR

schedule = LambdaLR(
    base_lr=0.1,
    lr_lambda=lambda step: jnp.exp(-0.001 * step),
)
```

## Stateful Schedule Example

If you need schedules that depend on previous values, pass an explicit schedule state and return `(lr, new_state)`:

```python
import jax.numpy as jnp

def adaptive_schedule(step, state):
    lr = state["lr"]
    new_lr = lr * jnp.where(step % 100 == 0, 0.5, 1.0)
    return lr, {"lr": new_lr}
```

Use it by passing `schedule_state` to `minimize`.

## API

::: nano_optax.schedulers.LRScheduler
::: nano_optax.schedulers.ConstantLR
::: nano_optax.schedulers.LambdaLR
::: nano_optax.schedulers.StepLR
