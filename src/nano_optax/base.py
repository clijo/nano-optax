from abc import ABC, abstractmethod
from typing import Callable, Tuple, Any
from .types import OptResult, PyTree


class Solver(ABC):
    def __init__(self, step_size: float, tol: float = 1e-4, verbose: bool = False):
        self.step_size = step_size
        self.tol = tol
        self.verbose = verbose

    @abstractmethod
    def minimize(
        self, fun: Callable, init_params: PyTree, data: Tuple[Any, ...]
    ) -> OptResult:
        """
        Args:
            fun: A function f(params, *data) -> float
            init_params: Initial guess for parameters (PyTree or Array)
            data: Tuple of (X, y) or other data arguments passed to fun
        """
        pass
