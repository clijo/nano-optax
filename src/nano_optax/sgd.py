import functools
import warnings
from typing import Callable

import jax
import jax.numpy as jnp

from .base import Solver
from .schedulers import as_schedule
from .types import OptResult, PyTree, LearningRate, ScheduleState


@functools.partial(jax.jit, static_argnames=("scheduler", "fun"))
def _opt_step(
    carry: tuple[PyTree, jax.Array, ScheduleState | None],
    indices: jax.Array,
    data: tuple[jax.Array, ...],
    scheduler: Callable[
        [jax.Array, ScheduleState | None], tuple[jax.Array, ScheduleState | None]
    ],
    fun: Callable[..., jax.Array],
) -> tuple[tuple[PyTree, jax.Array, ScheduleState | None], tuple[jax.Array, jax.Array]]:
    """Single gradient step on one minibatch.

    Args:
        carry: Tuple of (params, step_counter).
        indices: Indices selecting the minibatch from data.
        data: Full dataset as a tuple of arrays.
        scheduler: Learning rate schedule, maps step -> lr.
        fun: Objective function with signature f(params, *batch_data) -> scalar.

    Returns:
        Updated (params, step_counter) and the batch value.
    """
    params, step, schedule_state = carry
    batch_data = jax.tree.map(lambda x: x[indices], data)
    lr, schedule_state = scheduler(step, schedule_state)
    batch_val, grads = jax.value_and_grad(fun)(params, *batch_data)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return (new_params, step + 1, schedule_state), (batch_val, lr)


@functools.partial(jax.jit, static_argnames=("scheduler", "fun"))
def _run_epoch(
    carry: tuple[PyTree, jax.Array, ScheduleState | None],
    batched_indices: jax.Array,
    data: tuple[jax.Array, ...],
    scheduler: Callable[
        [jax.Array, ScheduleState | None], tuple[jax.Array, ScheduleState | None]
    ],
    fun: Callable[..., jax.Array],
) -> tuple[PyTree, jax.Array, ScheduleState | None, jax.Array, jax.Array]:
    """Run all batches in an epoch via lax.scan.

    Args:
        carry: Tuple of (params, step_counter).
        batched_indices: Array of shape (num_batches, batch_size) with indices.
        data: Full dataset as a tuple of arrays.
        scheduler: Learning rate schedule.
        fun: Objective function.

    Returns:
        Updated params, step counter, and sum of batch values.
    """

    def scan_body(
        c: tuple[PyTree, jax.Array, ScheduleState | None], idx: jax.Array
    ) -> tuple[
        tuple[PyTree, jax.Array, ScheduleState | None], tuple[jax.Array, jax.Array]
    ]:
        return _opt_step(c, idx, data, scheduler, fun)

    (params, step, schedule_state), (batch_vals, lr_vals) = jax.lax.scan(
        scan_body, carry, batched_indices
    )
    batch_sum = jnp.sum(batch_vals)
    last_lr = lr_vals[-1]
    return params, step, schedule_state, batch_sum, last_lr


class SGD(Solver):
    r"""Stochastic Gradient Descent (SGD) solver.

    Minimizes an objective function using minibatch gradient descent:

    $$\theta_{t+1} := \theta_t - \eta_t \nabla_\theta f(\theta_t; \mathcal{B}_t)$$

    where $\mathcal{B}_t$ is a randomly sampled minibatch at step $t$.

    An **epoch** is one complete pass over the dataset, so $\lceil \#(\mathcal{D}) / \text{batch_size} \rceil$ gradient steps.
    """

    def __init__(
        self,
        lr: LearningRate = 1e-3,
        max_epochs: int = 100,
        **kwargs,
    ) -> None:
        """
        Args:
            lr: Learning rate input. Can be a float for constant LR, a callable
                `schedule(step) -> lr`, a callable
                `schedule(step, state) -> (lr, new_state)`, or an LRScheduler.
            max_epochs: Maximum number of epochs to train.
            **kwargs: Base solver arguments (tol, verbose).
        """
        super().__init__(lr, **kwargs)
        self.max_epochs = max_epochs

    def minimize(
        self,
        fun: Callable[..., jax.Array],
        init_params: PyTree,
        data: tuple[jax.Array, ...],
        max_epochs: int | None = None,
        batch_size: int = 1,
        log_every: int = 1,
        check_every: int = 1,
        key: jax.Array | None = None,
        schedule_state: ScheduleState | None = None,
    ) -> OptResult:
        r"""Minimize the objective function using Stochastic Gradient Descent.

        Args:
            fun: Objective function with signature `f(params, *data) -> scalar`.
            init_params: Initial parameters (PyTree).
            data: Tuple of data arrays (e.g., `(X, y)`). All arrays must have
                the same length along axis 0 (number of samples).
            max_epochs: Override the instance's max_epochs. If None, uses
                `self.max_epochs`.
            batch_size: Size of minibatches. If `batch_size > num_samples`,
                the entire dataset is used as one batch (full-batch GD).
            log_every: Print progress every N epochs when `verbose=True`.
            check_every: Check convergence every N epochs (>=1).
            key: JAX PRNGKey for shuffling data each epoch.
            schedule_state: Optional initial state for a stateful schedule.

        Returns:
            OptResult containing final parameters, final value, and training trace.

        Raises:
            ValueError: If data is empty or arrays have mismatched lengths.

        Note:
            Each sample is seen exactly once per epoch. If `num_samples` is not
            divisible by `batch_size`, the last batch is smaller.
            Epoch values in the trace are sample-weighted mean batch losses.

        Warning:
            If `key=None`, data is processed in sequential order each epoch.
            This is deterministic cyclic gradient descent, not true SGD.
            For stochastic behavior, always provide a PRNGKey.
        """
        epochs = max_epochs if max_epochs is not None else self.max_epochs

        if not data:
            raise ValueError("data cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if log_every < 1:
            raise ValueError("log_every must be >= 1")
        if check_every < 1:
            raise ValueError("check_every must be >= 1")
        num_samples = len(data[0])
        if num_samples == 0:
            raise ValueError("data arrays cannot be empty")
        if not all(len(d) == num_samples for d in data):
            raise ValueError(
                f"All data arrays must have the same number of samples. "
                f"Got lengths: {[len(d) for d in data]}"
            )

        batch_size = min(batch_size, num_samples)
        num_full_batches = num_samples // batch_size
        remainder = num_samples % batch_size

        scheduler, schedule_state = as_schedule(self.lr, schedule_state)

        if key is None:
            warnings.warn(
                "No PRNGKey provided; data will not be shuffled. "
                "This is deterministic cyclic gradient descent, not SGD. "
                "For stochastic behavior, pass a PRNGKey via the `key` argument.",
                UserWarning,
                stacklevel=2,
            )

        if self.verbose:
            total_batches = num_full_batches + (1 if remainder else 0)
            print(
                f"SGD: n={num_samples}, batch={batch_size}, batches/epoch={total_batches}"
            )

        params = init_params
        step = jnp.array(0)
        val_trace = []
        prev_epoch_val = None
        current_key = key
        last_lr = None

        for epoch in range(epochs):
            if current_key is None:
                perm = jnp.arange(num_samples)
            else:
                current_key, subkey = jax.random.split(current_key)
                perm = jax.random.permutation(subkey, num_samples)

            total_sum = jnp.array(0.0)

            # Full batches via scan
            if num_full_batches > 0:
                full_indices = perm[: num_full_batches * batch_size].reshape(
                    (num_full_batches, batch_size)
                )
                params, step, schedule_state, batch_sum, last_lr = _run_epoch(
                    (params, step, schedule_state),
                    full_indices,
                    data,
                    scheduler,
                    fun,
                )
                total_sum += batch_sum * batch_size

            # Remainder batch
            if remainder:
                rem_indices = perm[num_full_batches * batch_size :]
                (params, step, schedule_state), (val, last_lr) = _opt_step(
                    (params, step, schedule_state), rem_indices, data, scheduler, fun
                )
                total_sum += val * remainder

            # Weighted mean over samples
            epoch_val = total_sum / num_samples
            val_trace.append(epoch_val)

            if self.verbose and epoch % log_every == 0:
                epoch_val_host = jax.device_get(epoch_val)
                if last_lr is None:
                    print(f"Epoch {epoch:4d}: val={float(epoch_val_host):.6e}")
                else:
                    current_lr = jax.device_get(last_lr)
                    print(
                        f"Epoch {epoch:4d}: "
                        f"val={float(epoch_val_host):.6e}, "
                        f"lr={float(current_lr):.6e}"
                    )

            if epoch > 0 and epoch % check_every == 0:
                delta = jnp.abs(prev_epoch_val - epoch_val)
                converged = bool(jax.device_get(delta < self.tol))
                if converged:
                    if self.verbose:
                        print(f"Converged at epoch {epoch} (value change < {self.tol})")
                    break

            prev_epoch_val = epoch_val

        val_trace_host = jax.device_get(jnp.stack(val_trace))
        val_trace_list = val_trace_host.tolist()
        return OptResult(
            params=params,
            final_value=val_trace_list[-1],
            trace=val_trace_list,
        )
