from abc import ABC, abstractmethod
from typing import Callable

from .types import OptResult, PyTree, LearningRate


class Solver(ABC):
    """Abstract base class for all solvers in nano-optax."""

    def __init__(
        self,
        lr: LearningRate,
        tol: float = 1e-5,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            lr: The learning rate. Can be a float for constant LR,
                a callable `schedule(step) -> float`, or an LRScheduler.
            tol: Tolerance for convergence based on loss change.
            verbose: Whether to print progress during optimization.
        """
        self.lr = lr
        self.tol = tol
        self.verbose = verbose

    @abstractmethod
    def minimize(
        self,
        fun: Callable,
        init_params: PyTree,
        data: tuple,
    ) -> OptResult:
        """Minimize the objective function.

        Args:
            fun: A function f(params, *data) -> value.
            init_params: Initial parameters (PyTree).
            data: Tuple of data arrays (e.g., (X, y)).

        Returns:
            The optimization result.
        """
        pass

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({attrs})"
