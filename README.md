# Nano Optax

Implementation of classic optimization algorithms in JAX.

All solvers are **pure functions** that return an `OptResult`:

```python
result = sgd(fun, init_params, data, lr=0.01)
# result.params, result.final_value, result.trace
```


## Installation

```bash
uv add nano-optax
```

## Quick Start
```python
import jax
import jax.numpy as jnp
from nano_optax import sgd, step_lr

# 1. Setup Data
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (100, 1))
true_w = jnp.array([2.5])
noise = 0.01 * jax.random.normal(key, (100,))
y = X @ true_w + noise

data = (X, y)

# 2. Define Objective
def fun(params, x, y):
    pred = x @ params["w"]
    return jnp.mean((pred - y) ** 2)

# 3. Minimize
init_params = {"w": jnp.array([0.0])}
result = sgd(
    fun,
    init_params,
    data,
    lr=step_lr(base_lr=0.1, step_size=1000, gamma=0.5),
    batch_size=16,
    key=jax.random.PRNGKey(42),
)
print(result.params)
```
