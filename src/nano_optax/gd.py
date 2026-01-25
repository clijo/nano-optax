import jax
import jax.numpy as jnp
from typing import Callable

from .base import Solver
from .schedulers import as_schedule
from .types import OptResult, PyTree, LearningRate, ScheduleState


class GD(Solver):
    r"""Vanilla Gradient Descent (GD) solver.

    Updates parameters using the full dataset at each step.

    The update rule for parameter $\theta$ at step $t$ is:

    $$
    \theta_{t+1} := \theta_t - \eta_t \nabla \mathcal{L}(\theta_t; \mathcal{D})
    $$

    where $\mathcal{D}$ is the entire dataset and $\eta_t$ is the learning rate
    at step $t$ (which may be constant or follow a schedule).
    """

    def __init__(
        self,
        lr: LearningRate = 1e-3,
        max_epochs: int = 100,
        **kwargs,
    ) -> None:
        """
        Args:
            lr: The learning rate. Can be a float for constant LR,
                a callable `schedule(step) -> float`, or an LRScheduler.
            max_epochs: Maximum number of epochs to train.
            **kwargs: Additional arguments passed to the base Solver
                (e.g., `tol`, `verbose`).
        """
        super().__init__(lr, **kwargs)
        self.max_epochs = max_epochs

    def minimize(
        self,
        fun: Callable,
        init_params: PyTree,
        data: tuple,
        max_epochs: int | None = None,
        schedule_state: ScheduleState | None = None,
    ) -> OptResult:
        """Minimize the function using vanilla Gradient Descent.

        Args:
            fun: A function f(params, *data) -> loss.
            init_params: Initial parameters (PyTree).
            data: Tuple of data arrays (e.g., (X, y)).
            max_epochs: Override the instance's max_epochs.
            schedule_state: Optional initial state for a stateful schedule.

        Returns:
            The optimization result.
        """
        epochs = max_epochs if max_epochs is not None else self.max_epochs

        scheduler, schedule_state = as_schedule(self.lr, schedule_state)

        @jax.jit
        def step(params, lr, *args):
            val, grads = jax.value_and_grad(fun)(params, *args)
            new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
            return new_params, val

        params = init_params
        trace = []

        for epoch in range(epochs):
            lr, schedule_state = scheduler(jnp.array(epoch), schedule_state)
            params, val = step(params, lr, *data)
            trace.append(float(val))

            if self.verbose:
                print(f"Epoch {epoch}: val={trace[-1]:.6e}, lr={float(lr):.6e}")

            if epoch > 0 and abs(trace[-2] - trace[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch}")
                break

        return OptResult(params=params, final_value=trace[-1], trace=trace)
