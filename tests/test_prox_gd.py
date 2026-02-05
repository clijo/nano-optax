import jax
import jax.numpy as jnp
from nano_optax import prox_gd, prox_l1


def test_prox_gd_l1_regularization():
    # 1D problem: min 0.5(w - 5)^2 + 1 * |w|
    # Derivative of smooth part: w - 5
    # Subgradient of L1 at w: sign(w)
    # Optimality: w - 5 + sign(w) = 0
    # if w > 0: w - 5 + 1 = 0 => w = 4
    # Solution is 4.0

    target = 5.0
    reg = 1.0

    def smooth_fun(w, _):
        return 0.5 * (w - target) ** 2

    def nonsmooth_fun(w):
        return reg * jnp.abs(w)

    prox_op = prox_l1(reg)
    init_params = jnp.array(0.0)
    data = (jnp.zeros((10, 1)),)  # Dummy data

    result = prox_gd(
        smooth_fun,
        nonsmooth_fun,
        prox_op,
        init_params,
        data,
        lr=0.1,
        max_epochs=200,
        tol=1e-4,
    )

    expected = 4.0
    assert jnp.allclose(result.params, expected, atol=1e-2)
    assert result.success
    assert isinstance(result.trace, jax.Array)


def test_prox_l1_operator():
    # Test soft thresholding logic directly
    # Prox_L1(x, lambda) = sign(x) * max(|x| - lambda, 0)
    prox = prox_l1(reg=1.0)

    # x = 1.5, lr=1.0, reg=1.0 => thresh = 1*1 = 1
    # out = sign(1.5) * (1.5 - 1) = 0.5
    res = prox(jnp.array(1.5), 1.0)
    assert jnp.allclose(res, 0.5)

    # x = 0.5, thresh=1 => out = 0
    res = prox(jnp.array(0.5), 1.0)
    assert jnp.allclose(res, 0.0)

    # x = -1.5 => out = -1 * (1.5 - 1) = -0.5
    res = prox(jnp.array(-1.5), 1.0)
    assert jnp.allclose(res, -0.5)
