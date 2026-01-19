import jax
from .base import Solver
from .types import OptResult, PyTree


class GD(Solver):
    """Vanilla Gradient Descent solver.

    Updates parameters using the full dataset at each step.
    """

    def __init__(self, step_size=1e-3, max_epochs=100, **kwargs):
        super().__init__(step_size, **kwargs)
        self.max_epochs = max_epochs

    def minimize(self, fun, init_params: PyTree, data: tuple, max_epochs: int = None):
        params = init_params
        loss_trace = []

        # use provided max_epochs or fall back to instance default
        epochs = max_epochs if max_epochs is not None else self.max_epochs

        @jax.jit
        def update(params, *args):
            val, grads = jax.value_and_grad(fun)(params, *args)
            new_params = jax.tree_util.tree_map(
                lambda p, g: p - self.step_size * g, params, grads
            )
            return new_params, val

        # Optimization loop
        for epoch in range(epochs):
            params, loss = update(params, *data)
            loss_trace.append(loss)

            if self.verbose:
                print(f"Epoch {epoch}: Loss = {loss:.5f}")

            if epoch > 0 and abs(loss_trace[-2] - loss_trace[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch} with loss {loss:.5f}")
                break

        return OptResult(params=params, final_loss=loss, trace=loss_trace)
