import jax
import jax.numpy as jnp
from nano_optax import gd


def test_jit_gd():
    target = jnp.array([1.0, 2.0])

    def loss(params, _):
        return jnp.sum((params - target) ** 2)

    @jax.jit
    def train_step(init_params):
        data = (jnp.zeros((1,)),)
        result = gd(loss, init_params, data, lr=0.1, max_epochs=50)
        return result.params, result.final_value

    init = jnp.array([0.0, 0.0])
    final_params, val = train_step(init)
    assert jnp.allclose(final_params, target, atol=1e-2)
    assert jnp.isfinite(val)


def test_vmap_over_lr():
    target = jnp.array([10.0])
    init = jnp.array([0.0])

    def loss(params, _):
        return jnp.sum((params - target) ** 2)

    def run_experiment(lr):
        data = (jnp.zeros((1,)),)
        return gd(loss, init, data, lr=lr, max_epochs=20).final_value

    lrs = jnp.array([0.01, 0.1, 0.5])
    final_values = jax.vmap(run_experiment)(lrs)

    assert final_values.shape == (3,)
    assert final_values[2] < final_values[0]


def test_grad_of_lr():
    target = jnp.array([1.0])
    init = jnp.array([0.0])

    def inner_loss(p, _):
        return 0.5 * jnp.sum((p - target) ** 2)

    def meta_loss(lr):
        res = gd(inner_loss, init, (jnp.zeros(1),), lr=lr, max_epochs=10)
        return res.final_value

    grad_lr = jax.grad(meta_loss)(0.01)
    assert jnp.isfinite(grad_lr)
    assert grad_lr < 0.0
