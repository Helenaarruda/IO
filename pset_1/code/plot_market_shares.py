"""Generate clean market-share paths for the assignment.

The default figures use mu=0 and have one plot per market regime, avoiding a
visually crowded combined chart. The same script can generate either regime
or both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def load_shares(mu: float, case: str) -> pd.DataFrame:
    df = pd.read_csv(OUTPUT / "market_shares.csv")
    return df[(df["mu"] == mu) & (df["case"] == case)].copy()


def plot_one(mu: float, case: str) -> Path:
    df = load_shares(mu, case)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    for product, group in df.groupby("j", sort=False):
        group = group.sort_values("t")
        ax.plot(
            group["t"],
            group["share"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=product,
        )

    regime = "a2-b1 coordination" if case == "collusive" else "Bertrand-Nash"
    ax.set_title(rf"Market-share paths ($\mu={mu:g}$, {regime})")
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Market share")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    suffix = "collusion" if case == "collusive" else "bertrand"
    mu_suffix = { -0.5: "minus05", 0.0: "0", 0.5: "05"}[mu]
    out = OUTPUT / f"market_shares_mu{mu_suffix}_{suffix}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("bertrand", "collusion", "both"),
        default="bertrand",
    )
    parser.add_argument(
        "--mu",
        type=float,
        choices=(-0.5, 0.0, 0.5),
        default=0.0,
    )
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if args.case in ("bertrand", "both"):
        print(f"saved {plot_one(args.mu, 'separate')}")
    if args.case in ("collusion", "both"):
        print(f"saved {plot_one(args.mu, 'collusive')}")


if __name__ == "__main__":
    main()
