# Nano Optax

Implementation of classic optimization algorithms in JAX.

## Installation

```bash
uv add nano-optax
```

## Quick Start
```python
import jax
import jax.numpy as jnp
from nano_optax import SGD, StepLR

# 1. Setup Data
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (100, 1))
true_w = jnp.array([2.5])
noise = 0.01 * jax.random.normal(key, (100,))
y = X @ true_w + noise

data = (X, y)

# 2. Define Loss
def loss_fun(params, x, y):
    pred = x @ params["w"]
    return jnp.mean((pred - y) ** 2)

# 3. Minimize
init_params = {"w": jnp.array([0.0])}
solver = SGD(step_size=StepLR(base_lr=0.1, step_size=1000, gamma=0.5))
result = solver.minimize(
    loss_fun, init_params, data,
    batch_size=16, key=jax.random.PRNGKey(42)
)
print(result.params)
```

## Schedulers

See [Schedulers](schedulers.md) for learning-rate schedule utilities and examples.
