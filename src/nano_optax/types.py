from typing import Any, Callable, NamedTuple, Union, TYPE_CHECKING

import jax

PyTree = Any
"""Type alias for JAX PyTrees (nested structures of arrays)."""

if TYPE_CHECKING:
    from .schedulers import LRScheduler

ScheduleState = PyTree
"""State carried by a schedule function."""

ScheduleFn = Callable[[jax.Array, ScheduleState], tuple[jax.Array, ScheduleState]]
"""Schedule function that returns (lr, new_state)."""

LearningRate = Union[
    float,
    jax.Array,
    Callable[[jax.Array], jax.Array],
    "LRScheduler",
    ScheduleFn,
]
"""Learning rate: constant, schedule function, or scheduler."""


class OptResult(NamedTuple):
    """Optimization result container.

    Attributes:
        params: Final optimized parameters.
        final_value: Final objective value.
        trace: History of objective values per epoch.
        success: Whether optimization terminated without errors.
    """

    params: PyTree
    final_value: float
    trace: list[float]
    success: bool = True
