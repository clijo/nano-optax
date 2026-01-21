from typing import Any, List, NamedTuple, Union, Callable

# Type alias for JAX PyTrees (nested structures of arrays)
# type for PyTree is complex to define in Python, so we make do with Any
PyTree = Any

# Type alias for learning rate: either a constant or a schedule function
LearningRate = Union[float, Callable[[int], float]]


class OptResult(NamedTuple):
    params: PyTree
    final_loss: float
    trace: List[float]
    success: bool = True
