from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .schedulers import as_schedule
from .types import LearningRate, OptResult, PyTree, ScheduleState


class GDState(NamedTuple):
    params: PyTree
    schedule_state: ScheduleState | None
    step: jax.Array
    value: jax.Array
    converged: jax.Array


def gd(
    fun: Callable[..., jax.Array],
    init_params: PyTree,
    data: tuple = (),
    *,
    lr: LearningRate = 1e-3,
    max_epochs: int = 100,
    tol: float = 1e-6,
    schedule_state: ScheduleState | None = None,
    verbose: bool = False,
) -> OptResult:
    """Run vanilla gradient descent.

    Args:
        fun: Objective function `f(params, *data) -> value`.
        init_params: Initial parameters (PyTree).
        data: Tuple of data arrays.
        lr: Learning rate (constant, schedule, or stateful schedule).
        max_epochs: Number of epochs to run.
        tol: Convergence tolerance on gradient norm.
        schedule_state: Optional initial state for a stateful schedule.
        verbose: Print progress during optimization.

    Returns:
        OptResult with final parameters, value, and trace.
    """
    scheduler, schedule_state = as_schedule(lr, schedule_state)
    tol_val = jnp.asarray(tol)

    init_val = fun(init_params, *data)

    init_state = GDState(
        params=init_params,
        schedule_state=schedule_state,
        step=jnp.array(0, dtype=jnp.int32),
        value=init_val,
        converged=jnp.array(False, dtype=jnp.bool_),
    )

    def scan_body(carry: GDState, _):
        params, sched_state, step, prev_val, converged = carry

        def perform_step(operand):
            p, s_state, s = operand
            lr_val, new_s_state = scheduler(s, s_state)
            val, grads = jax.value_and_grad(fun)(p, *data)

            sq_norm_grads = jax.tree_util.tree_reduce(
                jnp.add, jax.tree.map(lambda g: jnp.sum(g**2), grads)
            )
            grad_norm = jnp.sqrt(sq_norm_grads)

            new_p = jax.tree.map(lambda p_i, g_i: p_i - lr_val * g_i, p, grads)

            just_converged = grad_norm < tol_val
            final_p = jax.tree.map(
                lambda old, new: jnp.where(just_converged, old, new), p, new_p
            )

            return final_p, new_s_state, s + 1, val, just_converged

        def skip_step(operand):
            p, s_state, s = operand
            return p, s_state, s, prev_val, jnp.array(True, dtype=jnp.bool_)

        new_params, new_sched_state, new_step, new_val, now_converged = jax.lax.cond(
            converged,
            skip_step,
            perform_step,
            (params, sched_state, step),
        )

        new_state = GDState(
            params=new_params,
            schedule_state=new_sched_state,
            step=new_step,
            value=new_val,
            converged=now_converged,
        )

        if verbose:
            jax.debug.print("Epoch {}: value={}", new_step, new_val)

        return new_state, new_val

    final_state, trace = jax.lax.scan(scan_body, init_state, None, length=max_epochs)
    final_value = fun(final_state.params, *data)

    return OptResult(
        params=final_state.params,
        final_value=final_value,
        trace=trace,
        success=final_state.converged,
    )
