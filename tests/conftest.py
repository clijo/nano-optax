import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def quadratic_problem():
    # Min 0.5 * (x - target)^2
    target = jnp.array([10.0, -5.0])

    def fun(params, data_x=None):
        # Data unused for simple scalar minimization
        return 0.5 * jnp.sum((params - target) ** 2)

    init_params = jnp.array([0.0, 0.0])
    data = (jnp.zeros((1, 1)),)  # Dummy data
    return fun, init_params, target, data


@pytest.fixture
def linear_regression_problem():
    # y = Xw + b
    key = jax.random.PRNGKey(42)
    N, D = 100, 3
    X = jax.random.normal(key, (N, D))
    true_w = jnp.array([1.5, -2.0, 3.0])
    y = X @ true_w

    def fun(params, x, y):
        pred = x @ params
        return jnp.mean((pred - y) ** 2)

    init_params = jnp.zeros((D,))
    data = (X, y)
    return fun, init_params, true_w, data
