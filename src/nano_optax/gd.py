import jax
from typing import Callable

from .base import Solver
from .types import OptResult, PyTree, LearningRate


class GD(Solver):
    r"""Vanilla Gradient Descent (GD) solver.

    Updates parameters using the full dataset at each step.

    The update rule for parameter $\theta$ at step $t$ is:

    $$
    \theta_{t+1} = \theta_t - \eta_t \nabla \mathcal{L}(\theta_t; \mathcal{D})
    $$

    where $\mathcal{D}$ is the entire dataset and $\eta_t$ is the learning rate
    at step $t$ (which may be constant or follow a schedule).
    """

    def __init__(
        self,
        step_size: LearningRate = 1e-3,
        max_epochs: int = 100,
        **kwargs,
    ) -> None:
        """
        Args:
            step_size: The learning rate. Can be a float for constant LR,
                or a callable `schedule(step) -> float` for dynamic LR.
            max_epochs: Maximum number of epochs to train.
            **kwargs: Additional arguments passed to the base Solver
                (e.g., `tol`, `verbose`).
        """
        super().__init__(step_size, **kwargs)
        self.max_epochs = max_epochs

    def minimize(
        self,
        fun: Callable,
        init_params: PyTree,
        data: tuple,
        max_epochs: int | None = None,
    ) -> OptResult:
        """Minimize the function using vanilla Gradient Descent.

        Args:
            fun: A function f(params, *data) -> loss.
            init_params: Initial parameters (PyTree).
            data: Tuple of data arrays (e.g., (X, y)).
            max_epochs: Override the instance's max_epochs.

        Returns:
            The optimization result.
        """
        epochs = max_epochs if max_epochs is not None else self.max_epochs

        # Normalize step_size to a schedule function
        schedule_fn = (
            self.step_size if callable(self.step_size) else lambda _: self.step_size
        )

        @jax.jit
        def update(params, lr, *args):
            val, grads = jax.value_and_grad(fun)(params, *args)
            new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
            return new_params, val

        params = init_params
        val_trace = []

        for epoch in range(epochs):
            lr = schedule_fn(epoch)
            params, val = update(params, lr, *data)
            val_trace.append(val)

            if self.verbose:
                print(f"Epoch {epoch}: val={float(val):.6f}, lr={lr:.6f}")

            if epoch > 0 and abs(val_trace[-2] - val_trace[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch}")
                break

        return OptResult(params=params, final_loss=val, trace=val_trace)
