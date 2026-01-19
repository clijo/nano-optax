import jax.numpy as jnp
from nano_optax.gd import GD

def run_test():
    # synthetic data
    X = jnp.array([[1.0], [2.0], [3.0]])
    y = jnp.array([2.0, 4.0, 6.0])
    weights = jnp.array([1.0, 0.5, 2.0]) # weights for each sample
    data = (X, y, weights)

    # linear model: y = w*x
    def loss_fun(params, x, y, w_sample):
        # params is a dict (PyTree)
        pred = x @ params["w"]
        # weighted MSE
        return jnp.mean(w_sample * (pred - y) ** 2)

    init_params = {"w": jnp.array([0.0])}

    solver = GD(step_size=0.01, max_epochs=20, verbose=True)

    print("Starting minimization (Flexible Args Test)...")
    try:
        result = solver.minimize(loss_fun, init_params, data)
        print("Final params:", result.params)
        print("Final loss:", result.final_loss)
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
