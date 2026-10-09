"""Generate the Bresnahan-style product-positioning figure for part (iii)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bikes_model import G, PRODUCTS, Q, marginal_costs

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for t in (1, 10):
        pass

    # One figure with two separate panels, as requested in the assignment.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    offsets = [(6, 7), (-2, 8), (6, 8), (-2, -13), (6, 8)]

    for ax, t in zip(axes, (1, 10)):
        mc = marginal_costs(t)
        ax.scatter(Q, mc)
        for j, label in enumerate(PRODUCTS):
            ax.annotate(
                label,
                (Q[j], mc[j]),
                xytext=offsets[j],
                textcoords="offset points",
            )
        ax.set_title(f"t = {t}")
        ax.set_xlabel("Quality $q_j$")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Marginal cost $mc_{jt}$")
    fig.suptitle("Product positioning and marginal cost")
    fig.tight_layout()

    path = OUTPUT / "product_positioning.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
