from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .schedulers import as_schedule
from .types import LearningRate, OptResult, PyTree, ScheduleState


class ProxGDState(NamedTuple):
    params: PyTree
    schedule_state: ScheduleState | None
    step: jax.Array
    value: jax.Array
    converged: jax.Array

def prox_gd(
    fun: Callable[..., jax.Array],
    g: Callable[[PyTree], jax.Array],
    prox: Callable[[jax.Array, jax.Array], jax.Array],
    init_params: PyTree,
    data: tuple = (),
    *,
    lr: LearningRate = 1e-3,
    max_epochs: int = 100,
    tol: float = 1e-6,
    schedule_state: ScheduleState | None = None,
    verbose: bool = False,
) -> OptResult:
    r"""Run proximal gradient descent for objectives of the form $f$ + $g$, where $f$ is $L$-smooth and convex, and $g$ is (possibly nonsmooth) proper, l.s.c., and convex. The proximal operator for $g$ must be passed via the `prox` argument as an uncurried map with signature $(x,\eta)\mapsto \operatorname{prox}_{\eta g}(x)$. At iteration $t$, the algorithm does a:
    1. (Gradient step): $y_{t} := x_{t-1} - \eta_{t}\nabla f(x_{t-1})$, and
    2. (Proximal step) $x_{t}:=\operatorname{prox}_{\eta_{t} g}(y_{t})$.

    Args:
        fun: Smooth function `f(params, *data) -> value`.
        g: Nonsmooth function `g(params) -> value`.
        prox: Proximal operator `prox(params, lr) -> params`.
        init_params: Initial parameters (PyTree).
        data: Tuple of data arrays.
        lr: Learning rate (constant, schedule, or stateful schedule).
        max_epochs: Number of epochs to run.
        tol: Convergence tolerance on gradient mapping norm.
        schedule_state: Optional initial state for a stateful schedule.
        verbose: Print progress during optimization.

    Returns:
        OptResult with final parameters, value, and trace.
    """
    scheduler, schedule_state = as_schedule(lr, schedule_state)
    tol_val = jnp.asarray(tol)

    init_state = ProxGDState(
        params=init_params,
        schedule_state=schedule_state,
        step=jnp.array(0, dtype=jnp.int32),
        value=jnp.array(jnp.inf),
        converged=jnp.array(False, dtype=jnp.bool_),
    )

    def step_fn(carry):
        params, sched_state, step_count = carry
        lr_val, new_sched_state = scheduler(step_count, sched_state)

        val, grads = jax.value_and_grad(fun)(params, *data)
        g_val = g(params)

        new_params = jax.tree.map(
            lambda p, gr: prox(p - lr_val * gr, lr_val), params, grads
        )

        grad_map_norm = jnp.sqrt(
            jax.tree_util.tree_reduce(
                jnp.add,
                jax.tree.map(
                    lambda p, new_p: jnp.sum(((p - new_p) / lr_val) ** 2),
                    params,
                    new_params,
                ),
            )
        )

        return (new_params, new_sched_state, step_count + 1), (
            val + g_val,
            grad_map_norm,
        )

    def epoch_fn(state: ProxGDState, _):
        def run_step(s: ProxGDState):
            (new_params, new_sched, new_step), (total_val, gm_norm) = step_fn(
                (s.params, s.schedule_state, s.step)
            )
            is_conv = gm_norm < tol_val
            return ProxGDState(
                params=new_params,
                schedule_state=new_sched,
                step=new_step,
                value=total_val,
                converged=is_conv,
            )

        def skip_step(s: ProxGDState):
            return s

        new_state = jax.lax.cond(state.converged, skip_step, run_step, state)

        if verbose:
            jax.debug.print("Epoch {}: value={}", new_state.step, new_state.value)

        return new_state, new_state.value

    def step_fn_no_tol(carry):
        params, sched_state, step_count = carry
        lr_val, new_sched_state = scheduler(step_count, sched_state)

        val, grads = jax.value_and_grad(fun)(params, *data)
        g_val = g(params)

        new_params = jax.tree.map(
            lambda p, gr: prox(p - lr_val * gr, lr_val), params, grads
        )

        return (new_params, new_sched_state, step_count + 1), (val + g_val)

    def epoch_fn_no_tol(state: ProxGDState, _):
        (new_params, new_sched, new_step), total_val = step_fn_no_tol(
            (state.params, state.schedule_state, state.step)
        )
        new_state = ProxGDState(
            params=new_params,
            schedule_state=new_sched,
            step=new_step,
            value=total_val,
            converged=jnp.array(False, dtype=jnp.bool_),
        )

        if verbose:
            jax.debug.print("Epoch {}: value={}", new_state.step, new_state.value)

        return new_state, new_state.value

    if tol <= 0.0:
        final_state, trace = jax.lax.scan(
            epoch_fn_no_tol, init_state, None, length=max_epochs
        )
    else:
        final_state, trace = jax.lax.scan(epoch_fn, init_state, None, length=max_epochs)
    final_value = fun(final_state.params, *data) + g(final_state.params)

    return OptResult(
        params=final_state.params,
        final_value=final_value,
        trace=trace,
        success=final_state.converged,
    )


def prox_l1(reg: float = 1.0) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""Return the L1 proximal operator (soft-thresholding) as an uncurried map
    $(x,\eta)\mapsto \operatorname{prox}_{\eta \| \cdot \|_1}(x)$. """
    if reg < 0:
        raise ValueError("Regularization coefficient must be nonnegative.")
    return lambda x, lr: jnp.sign(x) * jnp.maximum(0, jnp.abs(x) - reg * lr)


def prox_l2(reg: float = 1.0) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""Return the squared-L2 norm's proximal operator as an uncurried map
    $(x,\eta)\mapsto \operatorname{prox}_{\eta \| \cdot \|_{2}^{2}}(x)$."""
    if reg < 0:
        raise ValueError("Regularization coefficient must be nonnegative.")
    return lambda x, lr: x / (1 + (2 * reg * lr))
