from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .schedulers import as_schedule
from .types import LearningRate, OptResult, PyTree, ScheduleState


class APGDState(NamedTuple):
    params: PyTree
    prev_params: PyTree
    mom_t: jax.Array
    schedule_state: ScheduleState | None
    step: jax.Array
    key: jax.Array | None
    value: jax.Array
    converged: jax.Array


def apgd(
    fun: Callable[..., jax.Array],
    g: Callable[[PyTree], jax.Array],
    prox: Callable[[jax.Array, jax.Array], jax.Array],
    init_params: PyTree,
    data: tuple[jax.Array, ...],
    *,
    lr: LearningRate = 1e-3,
    max_epochs: int = 100,
    batch_size: int | None = None,
    key: jax.Array | None = None,
    tol: float = 1e-6,
    schedule_state: ScheduleState | None = None,
    verbose: bool = False,
) -> OptResult:
    """Run accelerated proximal gradient descent (FISTA).

    Args:
        fun: Smooth function `f(params, *batch_data) -> value`.
        g: Nonsmooth function `g(params) -> value`.
        prox: Proximal operator `prox(params, lr) -> params`.
        init_params: Initial parameters (PyTree).
        data: Tuple of data arrays, sliced along axis 0.
        lr: Learning rate (constant, schedule, or stateful schedule).
        max_epochs: Number of epochs to run.
        batch_size: Minibatch size (None uses full batch).
        key: PRNGKey for shuffling (None disables shuffling).
        tol: Convergence tolerance on gradient mapping norm.
        schedule_state: Optional initial state for a stateful schedule.
        verbose: Print progress during optimization.

    Returns:
        OptResult with final parameters, value, and trace.
    """
    if not data:
        raise ValueError("data cannot be empty for APGD.")

    num_samples = len(data[0])
    batch_size = num_samples if batch_size is None else min(batch_size, num_samples)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    num_full_batches = num_samples // batch_size
    remainder = num_samples % batch_size

    scheduler, schedule_state = as_schedule(lr, schedule_state)
    tol_val = jnp.asarray(tol)

    init_state = APGDState(
        params=init_params,
        prev_params=init_params,
        mom_t=jnp.array(1.0),
        schedule_state=schedule_state,
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        value=jnp.array(jnp.inf),
        converged=jnp.array(False, dtype=jnp.bool_),
    )

    def step_fn(carry, indices):
        params, prev_params, mom_t, sched_state, step_count = carry

        next_t = (1.0 + jnp.sqrt(1.0 + 4.0 * mom_t**2)) / 2.0
        beta = (mom_t - 1.0) / next_t

        y_params = jax.tree.map(lambda p, pp: p + beta * (p - pp), params, prev_params)

        batch_data = jax.tree.map(lambda x: x[indices], data)
        lr_val, new_sched_state = scheduler(step_count, sched_state)

        batch_val, grads = jax.value_and_grad(fun)(y_params, *batch_data)
        new_params = jax.tree.map(
            lambda y, gr: prox(y - lr_val * gr, lr_val), y_params, grads
        )

        g_val = g(y_params)

        gm_sq = jax.tree_util.tree_reduce(
            jnp.add,
            jax.tree.map(
                lambda y, np: jnp.sum(((y - np) / lr_val) ** 2),
                y_params,
                new_params,
            ),
        )
        gm_norm = jnp.sqrt(gm_sq)

        return (new_params, params, next_t, new_sched_state, step_count + 1), (
            batch_val + g_val,
            gm_norm,
        )

    def epoch_scan(carry: APGDState, _):
        (
            params,
            prev_params,
            mom_t,
            sched_state,
            step,
            rng_key,
            prev_epoch_val,
            converged,
        ) = carry

        def run_epoch(operand):
            p, prev_p, m_t, s_state, s, k = operand

            if k is not None:
                new_k, subkey = jax.random.split(k)
                perm = jax.random.permutation(subkey, num_samples)
            else:
                new_k = k
                perm = jnp.arange(num_samples)

            weighted_sum = jnp.array(0.0)
            accum_gm_norm = jnp.array(0.0)
            count_batches = jnp.array(0.0)

            scan_carry = (p, prev_p, m_t, s_state, s)

            if num_full_batches > 0:
                full_indices = perm[: num_full_batches * batch_size].reshape(
                    (num_full_batches, batch_size)
                )
                scan_carry, (batch_vals, batch_gms) = jax.lax.scan(
                    step_fn, scan_carry, full_indices
                )
                weighted_sum += jnp.sum(batch_vals) * batch_size
                accum_gm_norm += jnp.sum(batch_gms)
                count_batches += num_full_batches

            if remainder > 0:
                rem_indices = perm[num_full_batches * batch_size :]
                scan_carry, (val, gm) = step_fn(scan_carry, rem_indices)
                weighted_sum += val * remainder
                accum_gm_norm += gm
                count_batches += 1

            new_p, new_prev_p, new_m_t, new_s_state, new_s = scan_carry

            epoch_val = weighted_sum / num_samples
            avg_gm_norm = accum_gm_norm / count_batches

            return (
                new_p,
                new_prev_p,
                new_m_t,
                new_s_state,
                new_s,
                new_k,
                epoch_val,
                avg_gm_norm,
            )

        def skip_epoch(operand):
            p, prev_p, m_t, s_state, s, k = operand
            return p, prev_p, m_t, s_state, s, k, prev_epoch_val, jnp.array(0.0)

        (
            new_params,
            new_prev_params,
            new_mom_t,
            new_sched_state,
            new_step,
            new_key,
            epoch_val,
            epoch_gm_norm,
        ) = jax.lax.cond(
            converged,
            skip_epoch,
            run_epoch,
            (params, prev_params, mom_t, sched_state, step, rng_key),
        )

        is_conv = epoch_gm_norm < tol_val
        now_converged = jnp.logical_or(converged, is_conv)

        new_state = APGDState(
            params=new_params,
            prev_params=new_prev_params,
            mom_t=new_mom_t,
            schedule_state=new_sched_state,
            step=new_step,
            key=new_key,
            value=epoch_val,
            converged=now_converged,
        )

        if verbose:
            jax.debug.print("Epoch {}: value={}", new_step, epoch_val)

        return new_state, epoch_val

    final_state, trace = jax.lax.scan(epoch_scan, init_state, None, length=max_epochs)
    final_value = fun(final_state.params, *data) + g(final_state.params)

    return OptResult(
        params=final_state.params,
        final_value=final_value,
        trace=trace,
        success=final_state.converged,
    )
