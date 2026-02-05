import jax
import jax.numpy as jnp
from nano_optax import gd, step_lr


def test_gd_quadratic_convergence(quadratic_problem):
    fun, init_params, target, data = quadratic_problem

    # GD with learning rate 0.1 on quadratic objective 0.5*x^2
    # should converge linearly.
    # Float32 precision limits make very small tolerances unattainable here.
    result = gd(fun, init_params, data, lr=0.1, max_epochs=200, tol=1e-5)

    assert jnp.allclose(result.params, target, atol=1e-2)
    assert result.success
    # Check return types
    assert isinstance(result.final_value, jax.Array)


def test_gd_stateful_schedule(quadratic_problem):
    fun, init_params, target, data = quadratic_problem

    # Decay LR
    scheduler = step_lr(base_lr=0.5, step_size=50, gamma=0.1)
    result = gd(fun, init_params, data, lr=scheduler, max_epochs=100)

    assert jnp.allclose(result.params, target, atol=1e-2)


def test_solver_early_convergence():
    # Problem initialized at solution
    target = jnp.array([1.0])
    init_params = jnp.array([1.0])

    def fun(params, _):
        return 0.5 * jnp.sum((params - target) ** 2)

    data = (jnp.zeros((1,)),)

    result = gd(fun, init_params, data, lr=0.1, max_epochs=100)

    # Should stay at 1.0
    assert jnp.allclose(result.params, target)
    # Should have detected convergence
    # Trace logic: first step (0) has value 0.0.
    # Since we start converged, we might do 1 check.
    # The scan returns the trace of values.
    # If using masking/cond, the value should be constant 0.0.
    assert result.trace[0] == 0.0
    assert result.trace[-1] == 0.0


def test_empty_data_ok():
    def fun(params):
        return jnp.sum(params**2)

    result = gd(fun, jnp.array(1.0), (), lr=0.1, max_epochs=1)
    assert jnp.isfinite(result.final_value)
