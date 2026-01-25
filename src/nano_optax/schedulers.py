from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Callable, cast

import jax
import jax.numpy as jnp


class LRScheduler(ABC):
    """Base class for learning rate schedulers."""

    def __init__(self, base_lr: float | jax.Array) -> None:
        self.base_lr = jnp.array(base_lr)

    @abstractmethod
    def get_lr(self, step: jax.Array) -> jax.Array:
        """Return the learning rate for a given step."""

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
    lr: float | jax.Array | Callable[[jax.Array], jax.Array] | LRScheduler,
) -> LRScheduler:
    """Coerce a learning-rate input into an LRScheduler."""
    if isinstance(lr, LRScheduler):
        return lr
    if callable(lr):
        schedule = cast(
            Callable[[jax.Array], jax.Array], lr
        )  # cast for type hinting for ty
        return LambdaLR(base_lr=1.0, lr_lambda=schedule)
    return ConstantLR(lr)


def as_schedule(
    lr: float
    | jax.Array
    | Callable[[jax.Array], jax.Array]
    | Callable[[jax.Array, object], tuple[jax.Array, object]]
    | LRScheduler,
    schedule_state: object | None = None,
) -> tuple[
    Callable[[jax.Array, object | None], tuple[jax.Array, object | None]], object | None
]:
    """Normalize to a pure schedule function with explicit state.

    Returns a function `(step, state) -> (lr, new_state)` and the initial state.
    If the schedule is stateless, the state is passed through unchanged.
    """
    if callable(lr) and not isinstance(lr, LRScheduler):
        needs_state = _expects_state(lr)
        if schedule_state is None and needs_state:
            raise ValueError(
                "schedule_state must be provided for a stateful schedule function."
            )
        if schedule_state is not None and needs_state:
            stateful = cast(Callable[[jax.Array, object], tuple[jax.Array, object]], lr)

            def scheduler(step, state):
                lr_val, new_state = stateful(step, state)
                return lr_val, new_state

        else:
            stateless = cast(Callable[[jax.Array], jax.Array], lr)

            def scheduler(step, state):
                return stateless(step), state

        return scheduler, schedule_state

    lr_scheduler = as_scheduler(lr)

    def scheduler(step, state):
        return lr_scheduler(step), state

    return scheduler, schedule_state
