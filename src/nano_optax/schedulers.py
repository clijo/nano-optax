from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, cast
import inspect

import jax
import jax.numpy as jnp


class LRScheduler(ABC):
    """Base class for learning rate schedulers (pure, functional API)."""

    def __init__(self, base_lr: float | jax.Array) -> None:
        self.base_lr = jnp.array(base_lr)

    @abstractmethod
    def get_lr(self, step: jax.Array) -> jax.Array:
        """Return the learning rate for a given step (pure function)."""

    def __call__(self, step: jax.Array) -> jax.Array:
        return self.get_lr(jnp.array(step))


class ConstantLR(LRScheduler):
    """Constant learning rate scheduler."""

    def get_lr(self, step: jax.Array) -> jax.Array:  # noqa: ARG002
        return self.base_lr


class LambdaLR(LRScheduler):
    """Schedule defined by a lambda (multiplier by default).

    Note: To wrap an absolute LR schedule function `schedule(step) -> lr`,
    use `LambdaLR(base_lr=1.0, lr_lambda=schedule)`.
    """

    def __init__(
        self,
        base_lr: float | jax.Array,
        lr_lambda: Callable[[jax.Array], jax.Array],
    ) -> None:
        super().__init__(base_lr)
        self.lr_lambda = lr_lambda

    def get_lr(self, step: jax.Array) -> jax.Array:
        return self.base_lr * self.lr_lambda(step)


class StepLR(LRScheduler):
    """Decay LR by gamma every `step_size` steps."""

    def __init__(
        self,
        base_lr: float | jax.Array,
        step_size: int,
        gamma: float = 0.1,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        super().__init__(base_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step: jax.Array) -> jax.Array:
        exponent = jnp.floor_divide(step, self.step_size)
        return self.base_lr * jnp.power(self.gamma, exponent)


def _expects_state(schedule: Callable[..., object]) -> bool:
    """Return True if the callable appears to require a state argument."""
    try:
        sig = inspect.signature(schedule)
    except (TypeError, ValueError):
        return False

    required = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]
    return len(required) >= 2


def as_scheduler(
    step_size: float | jax.Array | Callable[[jax.Array], jax.Array] | LRScheduler,
) -> LRScheduler:
    """Coerce a learning-rate input into an LRScheduler."""
    if isinstance(step_size, LRScheduler):
        return step_size
    if callable(step_size):
        schedule = cast(
            Callable[[jax.Array], jax.Array], step_size
        )  # cast for type hinting for ty
        return LambdaLR(base_lr=1.0, lr_lambda=schedule)
    return ConstantLR(step_size)


def as_schedule(
    step_size: float
    | jax.Array
    | Callable[[jax.Array], jax.Array]
    | Callable[[jax.Array, object], tuple[jax.Array, object]]
    | LRScheduler,
    schedule_state: object | None = None,
) -> tuple[Callable[[jax.Array, object | None], tuple[jax.Array, object | None]], object | None]:
    """Normalize to a pure schedule function with explicit state.

    Returns a function `(step, state) -> (lr, new_state)` and the initial state.
    If the schedule is stateless, the state is passed through unchanged.
    """
    if callable(step_size) and not isinstance(step_size, LRScheduler):
        if schedule_state is None and _expects_state(step_size):
            raise ValueError(
                "schedule_state must be provided for a stateful schedule function."
            )
        if schedule_state is not None and _expects_state(step_size):
            schedule_fn = step_size
        else:
            def schedule_fn(step, state):
                return step_size(step), state
        return schedule_fn, schedule_state

    scheduler = as_scheduler(step_size)

    def schedule_fn(step, state):
        return scheduler(step), state

    return schedule_fn, schedule_state
