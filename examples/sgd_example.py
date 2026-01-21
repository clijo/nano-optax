import jax
import jax.numpy as jnp
from nano_optax import SGD


def run_test():
    # synthetic data
    # 100 samples, linear relationship
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (100, 1))
    true_w = jnp.array([2.5])
    noise = 0.01 * jax.random.normal(key, (100,))
    y = X @ true_w + noise

    data = (X, y)

    # linear model: y = w*x
    def loss_fun(params, x, y):
        # params is a dict (PyTree)
        pred = x @ params["w"]
        # Allow for batch dimension
        return jnp.mean((pred - y) ** 2)

    init_params = {"w": jnp.array([0.0])}
    solver = SGD(
        step_size=lambda step: 0.1 * jnp.exp(-0.01 * step),
        max_epochs=50,
        verbose=True,
        tol=1e-5,
    )

    print("Starting minimization (SGD with minibatches)...")
    try:
        # Pass batch_size and key here
        result = solver.minimize(
            loss_fun, init_params, data, batch_size=16, key=jax.random.PRNGKey(42)
        )
        print("Final params:", result.params)
        print("Final loss:", result.final_loss)

        # Check against true value
        w_err = jnp.abs(result.params["w"] - true_w)
        print(f"Error in w: {w_err}")
        if w_err < 0.01:
            print("SUCCESS.")
        else:
            print("FAILURE.")

    except Exception as e:
        print(f"Optimization failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_test()
