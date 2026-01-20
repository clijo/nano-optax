import jax
from typing import Callable
from .base import Solver
from .types import OptResult, PyTree


class SGD(Solver):
    r"""Stochastic Gradient Descent (SGD) solver.

    Updates parameters using minibatches at each step. By default, it runs
    pure SGD with a batch size of 1. Adjust `batch_size` in the `minimize` method.

    An epoch still corresponds to an entire pass over the data, but now consists
    of $\lceil n / \text{batch_size} \rceil$ gradient steps, instead of 1 as in vanilla GD.

    Attributes:
        step_size (float): The learning rate $\eta$.
        max_epochs (int): Maximum number of epochs to train.
        tol (float): Tolerance for convergence based on loss change.
        verbose (bool): Whether to print progress.
    """

    def __init__(self, step_size=1e-3, max_epochs=100, **kwargs):
        super().__init__(step_size, **kwargs)
        self.max_epochs = max_epochs

    def minimize(
        self,
        fun: Callable,
        init_params: PyTree,
        data: tuple,
        max_epochs: int = None,
        batch_size: int = 1,
        key: jax.Array = None,
    ) -> OptResult:
        """Minimize the function using SGD.

        Args:
            fun: A function f(params, *data) -> (loss, aux) or loss.
            init_params: Initial parameters (PyTree).
            data: Tuple of data arrays (e.g. (X, y)).
            max_epochs: Override the instance's max_epochs.
            batch_size: Size of minibatches. Default 1.
            key: PRNGKey for shuffling data. Required for non-deterministic shuffling.

        Returns:
            OptResult: The optimization result.
        """
        params = init_params
        loss_trace = []
        
        epochs = max_epochs if max_epochs is not None else self.max_epochs

        # Prepare batching helper
        num_samples = len(data[0])
        num_batches = num_samples // batch_size
        remainder_size = num_samples % batch_size
        truncated_size = num_batches * batch_size

        if self.verbose:
            msg = f"Dataset size: {num_samples}. Batch size: {batch_size}. Using {num_batches} full batches"
            if remainder_size > 0:
                msg += f" and a remainder of {remainder_size}."
            print(msg)

        def prepare_batches(data, key=None):
            # shuffle if key is provided
            if key is not None:
                perm = jax.random.permutation(key, num_samples)
                # apply permutation to all arrays in data tuple
                data = jax.tree_util.tree_map(lambda x: x[perm], data)

            # Split into main part and remainder
            main_data = jax.tree_util.tree_map(lambda x: x[:truncated_size], data)
            remainder_data = (
                jax.tree_util.tree_map(lambda x: x[truncated_size:], data)
                if remainder_size > 0
                else None
            )

            # Reshape the main part for scanning
            def reshape_batch(arr):
                return arr.reshape((num_batches, batch_size) + arr.shape[1:])

            batched_main = jax.tree_util.tree_map(reshape_batch, main_data)
            return batched_main, remainder_data

        # pre-compile gradient step
        @jax.jit
        def train_step(params, batch):
            val, grads = jax.value_and_grad(fun)(params, *batch)
            new_params = jax.tree_util.tree_map(
                lambda p, g: p - self.step_size * g, params, grads
            )
            return new_params, val

        @jax.jit
        def epoch_scan(params, batched_data):
            # scan takes (carry, x) -> (carry, y)
            final_params, losses = jax.lax.scan(
                f=train_step, init=params, xs=batched_data
            )
            return final_params, losses

        # Optimization loop
        current_key = key
        for epoch in range(epochs):
            step_key = None
            if current_key is not None:
                current_key, step_key = jax.random.split(current_key)

            # Prepare batches for this epoch
            batched_data, remainder_data = prepare_batches(data, step_key)

            params, batch_losses = epoch_scan(params, batched_data)

            # Average the loss over the entire dataset
            total_loss_sum = jax.numpy.sum(batch_losses) * batch_size

            if remainder_data is not None:
                params, rem_loss = train_step(params, remainder_data)
                total_loss_sum += rem_loss * remainder_size

            epoch_loss = total_loss_sum / num_samples
            loss_trace.append(epoch_loss)

            if self.verbose:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.5f}")

            if epoch > 0 and abs(loss_trace[-2] - loss_trace[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch} with loss {epoch_loss:.5f}")
                break

        return OptResult(params=params, final_loss=epoch_loss, trace=loss_trace)
