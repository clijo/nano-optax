from __future__ import annotations

from typing import Callable, cast

import jax
import jax.numpy as jnp

from .types import LearningRate, ScheduleFn, ScheduleState


def constant_lr(lr: float | jax.Array) -> Callable[[jax.Array], jax.Array]:
    """Return a constant learning-rate schedule."""
    lr_val = jnp.asarray(lr)

    def schedule(step: jax.Array) -> jax.Array:  # noqa: ARG001
        return lr_val

    return schedule


def lambda_lr(
    lr_lambda: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
    """Schedule defined by a user-provided callable."""

    def schedule(step: jax.Array) -> jax.Array:
        return lr_lambda(step)

    return schedule


def step_lr(
    base_lr: float | jax.Array,
    step_size: int,
    gamma: float = 0.1,
) -> Callable[[jax.Array], jax.Array]:
    """Decay learning rate by gamma every `step_size` steps."""
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    base_lr = jnp.asarray(base_lr)
    gamma_val = jnp.asarray(gamma)

    def schedule(step: jax.Array) -> jax.Array:
        exponent = jnp.floor_divide(step, step_size)
        return base_lr * jnp.power(gamma_val, exponent)

    return schedule


def as_schedule(
    lr: LearningRate,
    schedule_state: ScheduleState | None = None,
) -> tuple[ScheduleFn, ScheduleState | None]:
    """Normalize to a pure schedule function with explicit state.

    Returns a function `(step, state) -> (lr, new_state)` and the initial state.
    If the schedule is stateless, the state is passed through unchanged. The
    schedule state must be a JAX PyTree to be compatible with JIT/scan.
    """
    if callable(lr):
        if schedule_state is None:
            stateless = cast(Callable[[jax.Array], jax.Array], lr)

            def scheduler(
                step: jax.Array, state: ScheduleState | None
            ) -> tuple[jax.Array, ScheduleState | None]:
                return stateless(step), state

            return scheduler, schedule_state

        stateful = cast(
            Callable[[jax.Array, ScheduleState], tuple[jax.Array, ScheduleState]], lr
        )

        def scheduler(
            step: jax.Array, state: ScheduleState
        ) -> tuple[jax.Array, ScheduleState]:
            lr_val, new_state = stateful(step, state)
            return lr_val, new_state

        return scheduler, schedule_state

    lr_val = jnp.asarray(lr)

    def scheduler(
        step: jax.Array, state: ScheduleState | None
    ) -> tuple[jax.Array, ScheduleState | None]:
        return lr_val, state

    return scheduler, schedule_state
