"""Solve all assignment equilibria and write the required CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from bikes_model import (
    G,
    PRODUCTS,
    Q,
    W,
    carbon_cost,
    marginal_costs,
    prepare_products,
    solve_equilibrium,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

MUS = (-0.5, 0.0, 0.5)
PERIODS = tuple(range(1, 11))
CASE_SUFFIX = {
    -0.5: "1",
    0.0: "2",
    0.5: "3",
}

FIELDS = ["j", "t", "qjt", "pjt", "wj", "gj", "ct"]
SHARE_FIELDS = ["case", "mu", "j", "t", "share"]


def assignment_rows(eq, t: int) -> list[dict[str, object]]:
    """Create the exact seven-column assignment panel for all original products."""
    p = eq.product_prices()
    ct = carbon_cost(t)
    rows = []
    for j in range(len(PRODUCTS)):
        rows.append({
            "j": PRODUCTS[j],
            "t": t,
            "qjt": f"{Q[j]:.15g}",
            "pjt": f"{p[j]:.15g}",
            "wj": f"{W[j]:.15g}",
            "gj": f"{G[j]:.15g}",
            "ct": f"{ct:.15g}",
        })
    return rows


def diagnostic_row(eq, t: int, mu: float, case: str) -> dict[str, object]:
    active_products = []
    for b in eq.active_blocks:
        active_products.extend(eq.blocks[b].members)

    keep, dropped = prepare_products(
        t,
        collusive=(case == "collusive"),
    )

    return {
        "case": case,
        "mu": mu,
        "t": t,
        "foc_norm": f"{eq.foc_norm:.12e}",
        "max_deviation_gain": f"{eq.max_deviation_gain:.12e}",
        "valid": eq.valid,
        "deviation_controller": eq.deviation_controller,
        "outside_share": f"{eq.outside_share:.15g}",
        "active_products": ";".join(PRODUCTS[sorted(set(active_products))]),
        "retained_products": ";".join(PRODUCTS[keep]),
        "dropped_products": ";".join(PRODUCTS[dropped]),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, object]] = []
    shares_panel: list[dict[str, object]] = []

    for collusive in (False, True):
        case = "collusive" if collusive else "separate"

        for mu in MUS:
            suffix = CASE_SUFFIX[mu]
            filename = (
                f"bikes_collusion_{suffix}.csv"
                if collusive
                else f"bikes_{suffix}.csv"
            )

            rows: list[dict[str, object]] = []
            print(f"\n=== {case.upper()} | mu={mu:+.1f} ===")

            for t in PERIODS:
                eq = solve_equilibrium(
                    t,
                    mu,
                    collusive=collusive,
                    verify=True,
                )
                rows.extend(assignment_rows(eq, t))

                product_shares = eq.product_shares()
                for j in range(len(PRODUCTS)):
                    shares_panel.append({
                        "case": case,
                        "mu": mu,
                        "j": PRODUCTS[j],
                        "t": t,
                        "share": product_shares[j],
                    })

                diagnostics.append(
                    diagnostic_row(eq, t, mu, case)
                )

                print(
                    f"t={t:2d} | FOC={eq.foc_norm:.3e} | "
                    f"max-dev={eq.max_deviation_gain:.3e} | "
                    f"valid={eq.valid} | "
                    f"active={','.join(PRODUCTS[sorted(set(j for b in eq.active_blocks for j in eq.blocks[b].members))])}"
                )

            path = OUTPUT / filename
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"saved {path}")

    diagnostic_fields = list(diagnostics[0])
    diag_path = OUTPUT / "equilibrium_diagnostics.csv"
    with diag_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=diagnostic_fields)
        writer.writeheader()
        writer.writerows(diagnostics)
    print(f"saved {diag_path}")

    share_path = OUTPUT / "market_shares.csv"
    with share_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHARE_FIELDS)
        writer.writeheader()
        writer.writerows(shares_panel)
    print(f"saved {share_path}")


if __name__ == "__main__":
    main()
