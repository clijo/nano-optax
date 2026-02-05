from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .schedulers import as_schedule
from .types import LearningRate, OptResult, PyTree, ScheduleState


class SGDState(NamedTuple):
    params: PyTree
    schedule_state: ScheduleState | None
    step: jax.Array
    key: jax.Array | None
    value: jax.Array


def sgd(
    fun: Callable[..., jax.Array],
    init_params: PyTree,
    data: tuple[jax.Array, ...],
    *,
    lr: LearningRate = 1e-3,
    max_epochs: int = 100,
    batch_size: int | None = 1,
    key: jax.Array | None = None,
    schedule_state: ScheduleState | None = None,
    verbose: bool = False,
) -> OptResult:
    """Run stochastic gradient descent.

    Args:
        fun: Objective function `f(params, *batch_data) -> value`.
        init_params: Initial parameters (PyTree).
        data: Tuple of data arrays, sliced along axis 0.
        lr: Learning rate (constant, schedule, or stateful schedule).
        max_epochs: Number of epochs to run.
        batch_size: Minibatch size (None uses full batch).
        key: PRNGKey for shuffling (None disables shuffling).
        schedule_state: Optional initial state for a stateful schedule.
        verbose: Print progress during optimization.

    Returns:
        OptResult with final parameters, value, and trace.
    """
    if not data:
        raise ValueError("data cannot be empty for SGD.")

    num_samples = len(data[0])
    batch_size = num_samples if batch_size is None else min(batch_size, num_samples)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    num_full_batches = num_samples // batch_size
    remainder = num_samples % batch_size

    scheduler, schedule_state = as_schedule(lr, schedule_state)

    init_state = SGDState(
        params=init_params,
        schedule_state=schedule_state,
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        value=jnp.array(jnp.inf),
    )

    def step_fn(carry, indices):
        params, sched_state, step_count = carry
        batch_data = jax.tree.map(lambda x: x[indices], data)
        lr_val, new_sched_state = scheduler(step_count, sched_state)

        val, grads = jax.value_and_grad(fun)(params, *batch_data)
        new_params = jax.tree.map(lambda p, g: p - lr_val * g, params, grads)

        return (new_params, new_sched_state, step_count + 1), val

    def epoch_scan(carry: SGDState, _):
        params, sched_state, step, rng_key, _ = carry

        if rng_key is not None:
            new_key, subkey = jax.random.split(rng_key)
            perm = jax.random.permutation(subkey, num_samples)
        else:
            new_key = rng_key
            perm = jnp.arange(num_samples)

        total_val = jnp.array(0.0)
        scan_carry = (params, sched_state, step)

        if num_full_batches > 0:
            full_indices = perm[: num_full_batches * batch_size].reshape(
                (num_full_batches, batch_size)
            )
            scan_carry, batch_vals = jax.lax.scan(step_fn, scan_carry, full_indices)
            total_val += jnp.sum(batch_vals) * batch_size

        if remainder > 0:
            rem_indices = perm[num_full_batches * batch_size :]
            scan_carry, val = step_fn(scan_carry, rem_indices)
            total_val += val * remainder

        new_params, new_sched_state, new_step = scan_carry
        epoch_val = total_val / num_samples

        new_state = SGDState(
            params=new_params,
            schedule_state=new_sched_state,
            step=new_step,
            key=new_key,
            value=epoch_val,
        )

        if verbose:
            jax.debug.print("Epoch {}: value={}", new_step, epoch_val)

        return new_state, epoch_val

    final_state, trace = jax.lax.scan(epoch_scan, init_state, None, length=max_epochs)
    final_value = fun(final_state.params, *data)

    return OptResult(
        params=final_state.params,
        final_value=final_value,
        trace=trace,
        success=True,
    )
