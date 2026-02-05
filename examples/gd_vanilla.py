import jax.numpy as jnp
from nano_optax import gd


def run_test():
    # synthetic data
    X = jnp.array([[1.0], [2.0], [3.0]])
    y = jnp.array([2.0, 4.0, 6.0])
    data = (X, y)

    # linear model: y = w*x
    def fun(params, x, y):
        # params is a dict (PyTree)
        pred = x @ params["w"]
        return jnp.mean((pred - y) ** 2)

    init_params = {"w": jnp.array([0.0])}

    print("Starting minimization (Vanilla GD)...")
    try:
        result = gd(fun, init_params, data, lr=0.01, max_epochs=20, verbose=True)
        print("Final params:", result.params)
        print("Final value:", result.final_value)
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_test()
