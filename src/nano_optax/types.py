from typing import Any, List, Optional
from dataclasses import dataclass, field

# Type alias for JAX PyTrees (nested structures of arrays)
# type for PyTree is complex to define in Python, so we make do with Any
PyTree = Any


@dataclass
class OptState:
    params: PyTree
    iter_num: int
    error: float


@dataclass
class OptResult:
    params: PyTree
    final_loss: float
    trace: List[float] = field(default_factory=list)
    success: bool = True
