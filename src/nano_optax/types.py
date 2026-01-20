from typing import Any, List, Optional, NamedTuple

# Type alias for JAX PyTrees (nested structures of arrays)
# type for PyTree is complex to define in Python, so we make do with Any
PyTree = Any

class OptResult(NamedTuple):
    params: PyTree
    final_loss: float
    trace: List[float]
    success: bool = True
