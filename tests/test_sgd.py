import jax
import jax.numpy as jnp
from nano_optax import sgd


def test_sgd_linear_regression(linear_regression_problem):
    fun, init_params, true_w, data = linear_regression_problem

    # SGD with full batch (batch_size=N) should behave like GD
    result = sgd(
        fun,
        init_params,
        data,
        lr=0.1,
        max_epochs=200,
        batch_size=100,
        key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(result.params, true_w, atol=1e-2)


def test_sgd_minibatch(linear_regression_problem):
    fun, init_params, true_w, data = linear_regression_problem

    # Stochastic Minibatch
    result = sgd(
        fun,
        init_params,
        data,
        lr=0.01,
        max_epochs=300,
        batch_size=10,
        key=jax.random.PRNGKey(0),
    )

    # SGD noise makes it harder to hit exact tolerance, but should be close
    assert jnp.allclose(result.params, true_w, atol=2e-1)
