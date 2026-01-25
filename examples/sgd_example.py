import jax
import jax.numpy as jnp
from nano_optax import SGD, StepLR


def run_test():
    # Synthetic regression: nonlinear target with a small MLP.
    key = jax.random.PRNGKey(0)
    key, x_key, noise_key = jax.random.split(key, 3)

    num_samples = 512
    input_dim = 5
    hidden_dim = 16

    X = jax.random.normal(x_key, (num_samples, input_dim))
    # Nonlinear ground-truth function with interactions.
    y_true = (
        1.2 * jnp.sin(X[:, 0])
        + 0.5 * X[:, 1] * X[:, 2]
        - 0.7 * jnp.cos(2.0 * X[:, 3])
        + 0.3 * X[:, 4] ** 3
    )
    noise = 0.01 * jax.random.normal(noise_key, (num_samples,))
    y = y_true + noise

    data = (X, y)

    def loss_fun(params, x, y):
        # 2-layer MLP with tanh nonlinearity.
        h = jnp.tanh(x @ params["W1"] + params["b1"])
        pred = h @ params["W2"] + params["b2"]
        return jnp.mean((pred.squeeze() - y) ** 2)

    # Random init; small weights for stability.
    key, w1_key, w2_key = jax.random.split(key, 3)
    init_params = {
        "W1": 0.1 * jax.random.normal(w1_key, (input_dim, hidden_dim)),
        "b1": jnp.zeros((hidden_dim,)),
        "W2": 0.1 * jax.random.normal(w2_key, (hidden_dim, 1)),
        "b2": jnp.zeros((1,)),
    }

    solver = SGD(
        # StepLR counts minibatch steps (not epochs), so use a larger step_size.
        step_size=StepLR(base_lr=0.1, step_size=1000, gamma=0.5),
        max_epochs=500,
        verbose=True,
        tol=1e-6,
    )

    print("Starting minimization (nonlinear regression with minibatches)...")
    try:
        result = solver.minimize(
            loss_fun, init_params, data, batch_size=32, key=jax.random.PRNGKey(42)
        )
        print("Final value:", result.final_value)

    except Exception as e:
        print(f"Optimization failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_test()
