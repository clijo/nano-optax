import jax
import jax.numpy as jnp

from nano_optax import prox_gd, prox_l1


def run_test():
    # Lasso regression: minimize 0.5 * ||X w - y||^2 + reg * ||w||_1
    key = jax.random.PRNGKey(0)
    key, x_key, noise_key, w_key = jax.random.split(key, 4)

    num_samples = 256
    num_features = 20
    reg = 0.05

    X = jax.random.normal(x_key, (num_samples, num_features))
    true_w = jax.random.normal(w_key, (num_features,))
    noise = 0.05 * jax.random.normal(noise_key, (num_samples,))
    y = X @ true_w + noise

    data = (X, y)

    def smooth_fun(w, x, y):
        residual = x @ w - y
        return 0.5 * jnp.mean(residual**2)

    def nonsmooth_fun(w):
        return reg * jnp.sum(jnp.abs(w))

    init_params = jnp.zeros((num_features,))

    print("Starting minimization (L1-regularized least squares)...")
    try:
        result = prox_gd(
            smooth_fun,
            nonsmooth_fun,
            prox_l1(reg),
            init_params,
            data,
            lr=0.1,
            max_epochs=200,
            tol=1e-6,
            verbose=True,
        )
        print("Final value (full objective):", result.final_value)
        print("True w (first 5):", jax.device_get(true_w[:5]))
        print("Learned w (first 5):", jax.device_get(result.params[:5]))
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_test()
