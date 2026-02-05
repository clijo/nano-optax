from typing import Any, Callable, NamedTuple

import jax

PyTree = Any
"""Type alias for JAX PyTrees (nested structures of arrays)."""

ScheduleState = PyTree
"""State carried by a schedule function (must be a JAX PyTree)."""

ScheduleFn = Callable[[jax.Array, ScheduleState], tuple[jax.Array, ScheduleState]]
"""Schedule function that returns (lr, new_state)."""

LearningRate = float | jax.Array | Callable[[jax.Array], jax.Array] | ScheduleFn
"""Learning rate: constant, stateless schedule, or stateful schedule."""


class OptResult(NamedTuple):
    """Optimization result container.

    Attributes:
        params: Final optimized parameters.
        final_value: Objective value at the returned parameters.
        trace: Objective values per epoch at the update evaluation point.
        success: Whether the convergence criterion was met.
    """

    params: PyTree
    final_value: jax.Array
    trace: jax.Array
    success: bool | jax.Array = True
