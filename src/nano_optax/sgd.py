import jax
import jax.numpy as jnp
from typing import Callable

from .base import Solver
from .types import OptResult, PyTree, LearningRate


class SGD(Solver):
    r"""Stochastic Gradient Descent (SGD) solver.

    Updates parameters using minibatches at each step. By default, it runs
    pure SGD with a batch size of 1. Adjust `batch_size` in the `minimize` method.

    An epoch corresponds to an entire pass over the data, consisting of
    $\lceil \#(\mathcal{D}) / \text{batch_size} \rceil$ gradient steps.

    The learning rate can be constant or follow a schedule based on the global
    step count (each gradient step in each epoch).
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
        batch_size: int = 1,
        key: jax.Array | None = None,
    ) -> OptResult:
        """Minimize the function using Stochastic Gradient Descent.

        Args:
            fun: A function f(params, *data) -> loss.
            init_params: Initial parameters (PyTree).
            data: Tuple of data arrays (e.g., (X, y)).
            max_epochs: Override the instance's max_epochs.
            batch_size: Size of minibatches.
            key: PRNGKey for shuffling data. Required for non-deterministic shuffling.

        Returns:
            The optimization result.
        """
        epochs = max_epochs if max_epochs is not None else self.max_epochs

        # Batching setup
        num_samples = len(data[0])
        num_batches = num_samples // batch_size
        remainder_size = num_samples % batch_size
        truncated_size = num_batches * batch_size

        # Normalize step_size to a schedule function
        schedule_fn = (
            self.step_size if callable(self.step_size) else lambda _: self.step_size
        )

        if self.verbose:
            msg = f"Dataset: {num_samples} samples, batch_size={batch_size}, {num_batches} batches"
            if remainder_size > 0:
                msg += f" (+{remainder_size} remainder)"
            print(msg)

        @jax.jit
        def train_step(carry, batch):
            """Single gradient step on one batch."""
            params, step = carry
            lr = schedule_fn(step)
            val, grads = jax.value_and_grad(fun)(params, *batch)
            new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
            return (new_params, step + 1), val

        @jax.jit
        def run_epoch(params, step, batched_data):
            """Run all batches in an epoch via lax.scan."""
            (params, step), vals = jax.lax.scan(
                train_step, (params, step), batched_data
            )
            return params, step, vals

        def prepare_batches(data, shuffle_key=None):
            """Shuffle and reshape data into batches."""
            if shuffle_key is not None:
                perm = jax.random.permutation(shuffle_key, num_samples)
                data = jax.tree_util.tree_map(lambda x: x[perm], data)

            main = jax.tree_util.tree_map(lambda x: x[:truncated_size], data)
            remainder = (
                jax.tree_util.tree_map(lambda x: x[truncated_size:], data)
                if remainder_size > 0
                else None
            )
            batched = jax.tree_util.tree_map(
                lambda x: x.reshape((num_batches, batch_size) + x.shape[1:]), main
            )
            return batched, remainder

        # Opt loop

        params = init_params
        step = jnp.array(0)
        val_trace = []
        current_key = key

        for epoch in range(epochs):
            # Shuffle data each epoch
            shuffle_key = None
            if current_key is not None:
                current_key, shuffle_key = jax.random.split(current_key)

            batched_data, remainder = prepare_batches(data, shuffle_key)

            # Main batches
            params, step, batch_vals = run_epoch(params, step, batched_data)
            total_val = jnp.sum(batch_vals) * batch_size

            # Remainder batch
            if remainder is not None:
                (params, step), rem_val = train_step((params, step), remainder)
                total_val += rem_val * remainder_size

            epoch_val = float(total_val / num_samples)
            val_trace.append(epoch_val)

            if self.verbose:
                lr = float(schedule_fn(int(step) - 1))
                print(f"Epoch {epoch}: val={epoch_val:.6f}, lr={lr:.6f}")

            # Convergence check
            if epoch > 0 and abs(val_trace[-2] - val_trace[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch}")
                break

        return OptResult(params=params, final_loss=epoch_val, trace=val_trace)
