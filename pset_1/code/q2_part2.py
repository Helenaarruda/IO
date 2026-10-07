# IO Problem Set 1 by Helena, Igor, and Qixuan. Question 2 Part 2 Code

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

root = Path(__file__).resolve().parent.parent
temp_dir, output_dir = root / "temp", root / "output"
output_dir.mkdir(parents=True, exist_ok=True)

products = ["a1", "a2", "a3", "b1", "b2"]
owners = jnp.array([0, 0, 0, 1, 1])
ownership = (owners[:, None] == owners[None, :]).astype(float)

nodes, weights = hermgauss(40)
nodes = jnp.asarray(np.sqrt(2) * nodes)
weights = jnp.asarray(weights / np.sqrt(np.pi))


def market_shares(theta, p, w, g):
    mu, beta_w, beta_g = theta
    delta = jnp.exp(beta_w * w + beta_g * g)
    alpha = jnp.exp(mu + nodes)
    probs = jax.nn.softmax(delta[None, :] - alpha[:, None] * p[None, :], axis=1)
    return jnp.sum(weights[:, None] * probs, axis=0)


def single_t_loglikelihood(theta, data_t):
    q, p, w, g = data_t
    s = market_shares(theta, p, w, g)
    return jnp.sum(q * jnp.log(jnp.clip(s, 1e-300, 1.0)))


def loglikelihood(theta, data):
    return jnp.sum(
        jax.vmap(single_t_loglikelihood, in_axes=(None, 0))(theta, data)
    )


def estimate_demand(data):
    value_grad = jax.jit(
        jax.value_and_grad(lambda x: -loglikelihood(x, data))
    )

    def objective(x):
        value, grad = value_grad(jnp.asarray(x))
        return float(value), np.asarray(grad, dtype=float)

    result = min(
        (
            minimize(
                objective,
                np.array([m, 0.0, 0.0]),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-3, 3), (-2, 2), (-2, 2)],
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000},
            )
            for m in (0.0, -1.0, 1.0)
        ),
        key=lambda r: r.fun,
    )

    if not result.success:
        raise RuntimeError("Demand estimation failed: " + result.message)

    return np.asarray(result.x)


share_jacobian = jax.jacrev(market_shares, argnums=1)


def single_t_implied_mc(theta, data_t):
    q, p, w, g, _ = data_t
    D = share_jacobian(theta, p, w, g)
    return p - jnp.linalg.solve(-(ownership * D).T, q)


def estimate_cost(theta, data):
    _, _, w, g, ct = data

    mc = jax.vmap(
        single_t_implied_mc,
        in_axes=(None, 0),
    )(jnp.asarray(theta), data)

    X = np.column_stack(
        [
            np.asarray((1 - w) * ct).ravel(),
            np.asarray(g).ravel(),
        ]
    )

    y = np.asarray(mc).ravel()
    gamma = np.linalg.lstsq(X, y, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((y - X @ gamma) ** 2)))

    return gamma, rmse


def load_panel(path):
    df = pd.read_csv(path)

    return tuple(
        jnp.asarray(
            df.pivot(
                index="t",
                columns="j",
                values=c,
            )[products].to_numpy(dtype=float)
        )
        for c in ["qjt", "pjt", "wj", "gj", "ct"]
    )


def estimate_dataset(path):
    data = load_panel(path)

    demand = estimate_demand(data[:4])
    gamma, rmse = estimate_cost(demand, data)

    names = [
        "mu_hat",
        "beta_w_hat",
        "beta_g_hat",
        "gamma_w_hat",
        "gamma_g_hat",
        "cost_rmse",
    ]

    return dict(
        zip(
            names,
            [*demand, *gamma, rmse],
        )
    )


datasets = (
    [f"bikes_{n}.csv" for n in range(1, 4)]
    + [f"bikes_collusion_{n}.csv" for n in range(1, 4)]
)

results = []

for name in datasets:
    path = temp_dir / name

    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    print(f"\nEstimating {name}...")

    results.append(
        {
            "dataset": name,
            **estimate_dataset(path),
        }
    )


results = pd.DataFrame(results)

print(
    "\n"
    + results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

results_path = output_dir / "q2_part2_estimation_results.csv"

results.to_csv(
    results_path,
    index=False,
)

print(
    f"\nEstimation results saved to:\n{results_path}"
)
