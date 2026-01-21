from abc import ABC, abstractmethod
from typing import Callable, Tuple, Any

from .types import OptResult, PyTree, LearningRate


class Solver(ABC):
    """Abstract base class for all solvers in nano-optax."""

    def __init__(
        self,
        step_size: LearningRate,
        tol: float = 1e-5,
        verbose: bool = False,
    ) -> None:
        """Initialize the solver.

        Args:
            step_size: The learning rate. Can be a float for constant LR,
                or a callable `schedule(step) -> float` for dynamic LR.
            tol: Tolerance for convergence based on loss change.
            verbose: Whether to print progress during optimization.
        """
        self.step_size = step_size
        self.tol = tol
        self.verbose = verbose

    def _get_lr(self, step: int) -> float:
        """Get the learning rate for a given step.

        Args:
            step: The current optimization step (0-indexed).

        Returns:
            The learning rate for this step.
        """
        if callable(self.step_size):
            return self.step_size(step)
        return self.step_size

    @abstractmethod
    def minimize(
        self,
        fun: Callable,
        init_params: PyTree,
        data: Tuple[Any, ...],
    ) -> OptResult:
        """Minimize the objective function.

        Args:
            fun: A function f(params, *data) -> loss.
            init_params: Initial parameters (PyTree).
            data: Tuple of data arrays (e.g., (X, y)).

        Returns:
            The optimization result.
        """
        pass

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({attrs})"
