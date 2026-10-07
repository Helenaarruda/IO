# IO Problem Set 1 by Helena, Igor, and Qixuan. Question 2 Part 1 Code

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import differential_evolution, minimize

products = np.array(["a1", "a2", "a3", "b1", "b2"])
w = np.array([0.3, 0.1, 1.0, 0.3, 0.5])
g = np.array([4.8, 1.9, 4.0, 2.0, 3.25])
A, B = np.array([0, 1, 2]), np.array([3, 4])
beta_w, beta_g, gamma_w, gamma_g = -0.1, 0.2, 0.25, 0.3
delta = np.exp(beta_w * w + beta_g * g)

root = Path(__file__).resolve().parent.parent
output_dir, temp_dir = root / "output", root / "temp"
for d in (output_dir, temp_dir):
    d.mkdir(parents=True, exist_ok=True)


def carbon_cost(t): return 0.5 + t / 10

def marginal_cost(t): return gamma_w * (1 - w) * carbon_cost(t) + gamma_g * g


# Part 1(iii)
def plot_positioning(t):
    mc = marginal_cost(t)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(delta, mc, s=60)

    counts = {}
    for j, x, y in zip(products, delta, mc):
        key = (round(float(x), 10), round(float(y), 10))
        n = counts.get(key, 0)
        ax.annotate(j, (x, y), xytext=(6, 6 + 14 * n),
                    textcoords="offset points")
        counts[key] = n + 1

    ax.set(
        xlabel=r"Quality $\delta_j$",
        ylabel="Marginal cost",
        title=f"Product Positioning and Marginal Cost, t = {t}"
    )

    fig.tight_layout()
    path = output_dir / f"q2_part1_iii_t{t}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# Demand: iid Type-I EV shock and alpha ~ LogNormal(mu, 1)
nodes, weights = hermgauss(40)
nodes, weights = np.sqrt(2) * nodes, weights / np.sqrt(np.pi)


def market_shares(prices, mu):
    alpha = np.exp(mu + nodes)
    u = delta - alpha[:, None] * np.asarray(prices, dtype=float)
    u -= u.max(axis=1, keepdims=True)

    e = np.exp(u)
    return (
        weights[:, None]
        * e / e.sum(axis=1, keepdims=True)
    ).sum(axis=0)


def firm_profit(prices, t, mu, idx):
    s, mc = market_shares(prices, mu), marginal_cost(t)
    return float(np.sum((prices[idx] - mc[idx]) * s[idx]))


def payoff(prices, t, mu, idx):
    return (
        firm_profit(prices, t, mu, A)
        + firm_profit(prices, t, mu, B)
        if idx is None
        else firm_profit(prices, t, mu, idx)
    )


def upper_bound(t, mu):
    mc = marginal_cost(t)
    return max(10.0, float(mc.max() + 4 * delta.max() / np.exp(mu)))


def best_response(prices, t, mu, choice, payoff_idx,
                  global_search=False, seed=0):

    prices = np.asarray(prices, dtype=float)
    mc, upper = marginal_cost(t), upper_bound(t, mu)
    bounds = [(1e-6, upper)] * len(choice)

    def objective(x):
        p = prices.copy()
        p[choice] = x
        return -payoff(p, t, mu, payoff_idx)

    if global_search:
        r = differential_evolution(
            objective,
            bounds,
            seed=seed,
            popsize=8,
            maxiter=120,
            tol=1e-7,
            polish=True
        )

    else:
        starts = [
            prices[choice],
            mc[choice] + 0.5,
            mc[choice] + 1.5,
            mc[choice] + 3
        ]

        rs = [
            minimize(
                objective,
                np.clip(x, 1e-6, upper),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                    "maxiter": 1000
                }
            )
            for x in starts
        ]

        r = min(rs, key=lambda x: x.fun)

    return r.x, -r.fun


COMPETITIVE = [
    (A, A),
    (B, B)
]

COLLUSIVE = [
    (np.array([0, 2]), A),
    (np.array([4]), B),
    (np.array([1, 3]), None)
]


def solve_game(t, mu, players, initial=None, seed_offset=0):

    prices = (
        marginal_cost(t) + 1
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )

    for _ in range(300):
        old = prices.copy()

        for choice, payoff_idx in players:
            br, _ = best_response(
                prices,
                t,
                mu,
                choice,
                payoff_idx
            )

            prices[choice] = (
                0.3 * prices[choice]
                + 0.7 * br
            )

        if np.max(np.abs(prices - old)) < 1e-8:
            break

    else:
        raise RuntimeError(
            f"Best-response iteration did not converge "
            f"for mu={mu}, t={t}."
        )

    gain = 0.0

    for i, (choice, payoff_idx) in enumerate(players):

        current = payoff(
            prices,
            t,
            mu,
            payoff_idx
        )

        seed = (
            seed_offset
            + 1000
            + 100 * t
            + 10 * i
            + int(round(10 * (mu + 1)))
        )

        _, best = best_response(
            prices,
            t,
            mu,
            choice,
            payoff_idx,
            True,
            seed
        )

        gain = max(
            gain,
            best - current
        )

    if gain > 1e-5:
        raise RuntimeError(
            f"Nash verification failed for mu={mu}, t={t}. "
            f"Maximum profitable deviation = {gain:.6g}"
        )

    return (
        prices,
        market_shares(prices, mu),
        gain
    )


def generate_panel(mu, players, label, seed_offset=0):

    rows, previous = [], None

    for t in range(1, 11):

        prices, shares, gain = solve_game(
            t,
            mu,
            players,
            previous,
            seed_offset
        )

        previous, c = prices.copy(), carbon_cost(t)

        print(
            f"{label}: mu = {mu: .1f}, "
            f"t = {t:2d}, "
            f"max profitable deviation = {gain:.2e}"
        )

        rows += [
            {
                "j": products[j],
                "t": t,
                "qjt": shares[j],
                "pjt": prices[j],
                "wj": w[j],
                "gj": g[j],
                "ct": c
            }
            for j in range(5)
        ]

    return pd.DataFrame(rows)


def save_panels(prefix, players, label, seed_offset=0):

    paths = []

    for n, mu in enumerate(
        (-0.5, 0.0, 0.5),
        1
    ):

        path = temp_dir / f"{prefix}_{n}.csv"

        generate_panel(
            mu,
            players,
            label,
            seed_offset
        ).to_csv(
            path,
            index=False,
            columns=[
                "j",
                "t",
                "qjt",
                "pjt",
                "wj",
                "gj",
                "ct"
            ]
        )

        print(f"\nSaved {path}\n")
        paths.append(path)

    return paths


def main():

    figures = [
        plot_positioning(t)
        for t in (1, 10)
    ]

    competitive = save_panels(
        "bikes",
        COMPETITIVE,
        "Competitive"
    )

    collusive = save_panels(
        "bikes_collusion",
        COLLUSIVE,
        "Collusion",
        5000
    )

    print(
        "\nProduct   Quality      MC(t=1)     MC(t=10)"
    )

    for j, q, mc1, mc10 in zip(
        products,
        delta,
        marginal_cost(1),
        marginal_cost(10)
    ):

        print(
            f"{j:4s}      "
            f"{q:.6f}     "
            f"{mc1:.4f}       "
            f"{mc10:.4f}"
        )

    print(
        "\nFigures saved to:",
        *figures,
        sep="\n"
    )

    print(
        "\nCompetitive datasets saved to:",
        *competitive,
        sep="\n"
    )

    print(
        "\nCollusive datasets saved to:",
        *collusive,
        sep="\n"
    )


if __name__ == "__main__":
    main()
