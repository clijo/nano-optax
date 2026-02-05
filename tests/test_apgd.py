import jax
import jax.numpy as jnp
from nano_optax import apgd, prox_l1


def test_apgd_acceleration():
    # Same objective as ProxGD
    target = 5.0
    reg = 1.0

    def smooth_fun(w, _):
        return 0.5 * (w - target) ** 2

    def nonsmooth_fun(w):
        return reg * jnp.abs(w)

    prox_op = prox_l1(reg)
    init_params = jnp.array(0.0)
    data = (jnp.zeros((10, 1)),)

    result = apgd(
        smooth_fun,
        nonsmooth_fun,
        prox_op,
        init_params,
        data,
        lr=0.1,
        max_epochs=100,
        tol=1e-4,
    )

    expected = 4.0
    assert jnp.allclose(result.params, expected, atol=1e-2)
    assert result.success
    assert isinstance(result.trace, jax.Array)
