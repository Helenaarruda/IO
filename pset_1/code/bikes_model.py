"""Core model for the bicycle vertical-differentiation exercise.

The model is

    u_ij = q_j - alpha_i p_j,
    log(alpha_i) ~ N(mu, 1),
    u_i0 = 0.

Exact-quality tie assumptions
-----------------------------
1. If equal quality products have different marginal costs in the
   Bertrand-Nash game, only the lowest-MC product survives, and the
   efficient firm is constrained to charge the higher firm's MC. This is
   the reduced form of the assumed entry threat.
2. If equal quality AND equal MC, all efficient tied products remain and
   are imposed to undercut to MC, with the resulting quality-block demand
   split equally.
3. In the collusion exercise, a2 and b1 are kept as a common-price block
   so that their coordinated price can be solved even when their MCs differ.

At each trial price vector, the actual demand hull is recomputed. The solver
optimizes the prices of the full reduced set in one FOC system: fixed-price
tie blocks are held fixed, and the other blocks are free variables. A block
outside the current hull receives zero share and zero local share derivatives.
This is deliberately a simple FOC-based approximation, not a global Nash check.

Partial-equilibrium treatment:
- products removed by the maintained tie/entry assumption do not enter demand;
- constrained-price tie blocks remain consumer options at their fixed prices,
  but their prices are not optimization variables;
- all remaining unrestricted quality blocks enter one FOC solve;
- the hull is recalculated at every trial price vector;
- no active-set enumeration or global deviation optimization is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

PRODUCTS = np.array(["a1", "a2", "a3", "b1", "b2"])
W = np.array([0.3, 0.1, 1.0, 0.3, 0.5])
G = np.array([4.8, 1.9, 4.0, 2.0, 3.25])
FIRMS = np.array(["A", "A", "A", "B", "B"])

BETA_W = -0.1
BETA_G = 0.2
GAMMA_W = 0.25
GAMMA_G = 0.3

Q = np.exp(BETA_W * W + BETA_G * G)

A2_INDEX = int(np.where(PRODUCTS == "a2")[0][0])
B1_INDEX = int(np.where(PRODUCTS == "b1")[0][0])

# Prices used only for products with zero demand / products removed by the
# entry rule. Their exact values are not economically identified; the user
# asked to keep the high numerical values in the exported panel instead of
# replacing them with blanks.
INACTIVE_PRICE_BASE = 100.0


@dataclass(frozen=True)
class QualityBlock:
    members: tuple[int, ...]
    quality: float
    mc: float
    full_tie: bool
    collusion_pair: bool

    # In the separate-firm Bertrand game, a quality tie is imposed as:
    #
    #   unequal MC: efficient firm serves the block at the HIGHER MC
    #   equal MC:   tied firms price at MC and split demand equally
    #
    # None means the block has an unrestricted price.
    bertrand_fixed_price: float | None


@dataclass
class Equilibrium:
    t: int
    mu: float
    case: str
    p_block: np.ndarray
    share_block: np.ndarray
    outside_share: float
    active_blocks: np.ndarray
    blocks: list[QualityBlock]
    foc_norm: float
    max_deviation_gain: float = 0.0
    deviation_controller: str = ""
    deviation_profit: float = np.nan
    current_profit: float = np.nan
    valid: bool = False

    def product_prices(self) -> np.ndarray:
        """Return numeric product-level prices for CSV export.

        Every retained block reports its solved price or its imposed tie price.
        Products removed by the unequal-MC entry rule receive a high numeric
        placeholder for export only.
        """
        out = INACTIVE_PRICE_BASE + 10.0 * Q.copy()
        for b, block in enumerate(self.blocks):
            if np.isfinite(self.p_block[b]):
                for j in block.members:
                    out[j] = self.p_block[b]
            else:
                for j in block.members:
                    out[j] = INACTIVE_PRICE_BASE + 10.0 * Q[j]
        return out

    def product_shares(self) -> np.ndarray:
        """Split each quality-block share equally among block members."""
        out = np.zeros(len(PRODUCTS), dtype=float)
        for b, block in enumerate(self.blocks):
            for j in block.members:
                out[j] = self.share_block[b] / len(block.members)
        return out


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def carbon_cost(t: int) -> float:
    return 0.5 + t / 10.0


def marginal_costs(t: int) -> np.ndarray:
    return GAMMA_W * (1.0 - W) * carbon_cost(t) + GAMMA_G * G


# ---------------------------------------------------------------------------
# Quality-tie preprocessing
# ---------------------------------------------------------------------------


def _quality_groups(indices: Iterable[int], tol: float = 1e-10) -> list[list[int]]:
    order = sorted(indices, key=lambda j: Q[j])
    groups: list[list[int]] = []
    for j in order:
        if not groups or abs(Q[j] - Q[groups[-1][0]]) > tol:
            groups.append([j])
        else:
            groups[-1].append(j)
    return groups


def prepare_products(
    t: int,
    *,
    collusive: bool,
    q_tol: float = 1e-10,
    mc_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the model's entry/tie rules and return retained/dropped products."""
    mc = marginal_costs(t)
    keep: list[int] = []
    dropped: list[int] = []

    for group in _quality_groups(range(len(PRODUCTS)), q_tol):
        group_set = set(group)
        pair_present = A2_INDEX in group_set and B1_INDEX in group_set

        # Part (vi) explicitly keeps a2 and b1 as the coordinated block.
        if collusive and pair_present:
            keep.extend(group)
            continue

        min_mc = np.min(mc[group])
        efficient = [j for j in group if abs(mc[j] - min_mc) <= mc_tol]
        keep.extend(efficient)
        dropped.extend(j for j in group if j not in efficient)

    keep = np.array(sorted(keep, key=lambda j: Q[j]), dtype=int)
    return keep, np.array(dropped, dtype=int)


def build_blocks(
    keep: np.ndarray,
    t: int,
    *,
    collusive: bool,
    q_tol: float = 1e-10,
) -> list[QualityBlock]:
    """Construct quality blocks and impose the maintained tie assumptions.

    For the separate-firm game:

    * Unequal-MC quality tie:
        - retain only the lower-MC product;
        - FIX its price at the higher MC;
        - the higher-MC firm is treated as out of the pricing game.

    * Full quality-and-MC tie:
        - retain all tied efficient products;
        - FIX their common price at MC;
        - split the block's demand equally.

    These are maintained assumptions, not conditions that the numerical
    solver is allowed to undo.

    For the collusion exercise:
        - a2 and b1 remain together as a coordinated common-price block.
    """

    mc = marginal_costs(t)

    # Map each retained quality to its original quality group.
    original_groups = _quality_groups(
        range(len(PRODUCTS)),
        q_tol,
    )

    blocks: list[QualityBlock] = []

    for group in _quality_groups(
        keep.tolist(),
        q_tol,
    ):
        group = list(group)
        group_set = set(group)

        # Find the corresponding original quality group.
        original_group = None
        for og in original_groups:
            if any(j in og for j in group):
                original_group = og
                break

        if original_group is None:
            raise RuntimeError("Could not recover original quality group.")

        original_mc = mc[
            np.asarray(original_group, dtype=int)
        ]

        pair = group_set == {A2_INDEX, B1_INDEX}

        unique_firms = np.unique(
            FIRMS[np.asarray(group, dtype=int)]
        )

        same_mc = np.allclose(
            original_mc,
            original_mc[0],
            atol=1e-10,
            rtol=0.0,
        )

        # Equal-q/equal-MC tie across multiple firms.
        full_tie = (
            len(unique_firms) > 1
            and same_mc
        )

        # Default: unrestricted price.
        bertrand_fixed_price = None

        if not collusive:

            if len(original_group) > 1:

                min_mc = float(
                    np.min(original_mc)
                )
                max_mc = float(
                    np.max(original_mc)
                )

                if max_mc - min_mc > 1e-10:
                    # Unequal-MC tie:
                    # surviving efficient product(s) charge the HIGHER MC.
                    bertrand_fixed_price = max_mc

                else:
                    # Full tie:
                    # common price = MC.
                    bertrand_fixed_price = min_mc

        blocks.append(
            QualityBlock(
                members=tuple(group),
                quality=float(Q[group[0]]),
                mc=float(
                    np.mean(
                        mc[
                            np.asarray(group, dtype=int)
                        ]
                    )
                ),
                full_tie=bool(full_tie),
                collusion_pair=bool(
                    pair and collusive
                ),
                bertrand_fixed_price=bertrand_fixed_price,
            )
        )

    return blocks


# ---------------------------------------------------------------------------
# Demand and Jacobian on a fixed hull
# ---------------------------------------------------------------------------


def lognormal_cdf(alpha: np.ndarray, mu: float) -> np.ndarray:
    alpha = np.maximum(alpha, 1e-300)
    return ndtr(np.log(alpha) - mu)


def lognormal_pdf(alpha: np.ndarray, mu: float) -> np.ndarray:
    alpha = np.maximum(alpha, 1e-300)
    z = np.log(alpha) - mu
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * alpha)


def block_demand(
    q_active: np.ndarray,
    p_active: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, float]:
    """Demand shares for a fixed quality hull."""
    q0 = np.r_[0.0, q_active]
    p0 = np.r_[0.0, p_active]
    dp = np.diff(p0)
    if np.any(dp <= 0):
        raise ValueError("Prices must be strictly increasing along the hull.")

    alpha = np.diff(q0) / dp
    if np.any(alpha <= 0):
        raise ValueError("Thresholds must be positive.")

    F = lognormal_cdf(alpha, mu)
    if len(alpha) == 1:
        shares = np.array([F[-1]])
    else:
        shares = np.r_[F[:-1] - F[1:], F[-1]]

    outside = float(1.0 - F[0])
    return shares, outside


def block_demand_jacobian(
    q_active: np.ndarray,
    p_active: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Analytical ds/dp for the fixed-hull demand system."""
    q0 = np.r_[0.0, q_active]
    p0 = np.r_[0.0, p_active]
    dp = np.diff(p0)
    if np.any(dp <= 0):
        raise ValueError("Prices must be strictly increasing along the hull.")

    alpha = np.diff(q0) / dp
    if np.any(alpha <= 0):
        raise ValueError("Thresholds must be positive.")

    H = alpha * lognormal_pdf(alpha, mu) / dp
    K = len(p_active)
    J = np.zeros((K, K), dtype=float)

    # Interior block r has s_r = F_r - F_{r+1}.
    for r in range(K - 1):
        hr = H[r]
        J[r, r] -= hr
        if r > 0:
            J[r, r - 1] += hr

        hn = H[r + 1]
        J[r, r] -= hn
        J[r, r + 1] += hn

    # Highest-quality block has s_K = F_K.
    J[-1, :] = 0.0
    J[-1, -1] = -H[-1]
    if K > 1:
        J[-1, -2] = H[-1]

    return J


def hull_indices(
    q_blocks: np.ndarray,
    p_blocks: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return blocks on the upper envelope of q - alpha p."""
    ids = list(range(len(q_blocks)))
    if not ids:
        return np.array([], dtype=int)

    # Pareto pruning.
    changed = True
    while changed:
        changed = False
        for pos, j in enumerate(ids):
            dominated = any(
                k != j
                and q_blocks[k] >= q_blocks[j] - tol
                and p_blocks[k] <= p_blocks[j] + tol
                and (
                    q_blocks[k] > q_blocks[j] + tol
                    or p_blocks[k] < p_blocks[j] - tol
                )
                for k in ids
            )
            if dominated:
                ids.pop(pos)
                changed = True
                break

    while True:
        qh = np.r_[0.0, q_blocks[ids]]
        ph = np.r_[0.0, p_blocks[ids]]
        dp = np.diff(ph)
        if np.any(dp <= 1e-12):
            return np.array([], dtype=int)

        slopes = np.diff(qh) / dp
        bad = np.where(slopes[:-1] <= slopes[1:] + tol)[0]
        if len(bad) == 0:
            return np.array(ids, dtype=int)

        # Remove the middle quality block.
        ids.pop(int(bad[0]))
        if not ids:
            return np.array([], dtype=int)


# ---------------------------------------------------------------------------
# Product/block transformations
# ---------------------------------------------------------------------------


def block_arrays(blocks: list[QualityBlock]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([b.quality for b in blocks], dtype=float),
        np.array([b.mc for b in blocks], dtype=float),
    )


def product_prices_from_blocks(
    blocks: list[QualityBlock],
    p_block: np.ndarray,
) -> np.ndarray:
    """Map block prices to product-level prices."""
    out = INACTIVE_PRICE_BASE + 10.0 * Q.copy()
    for b, block in enumerate(blocks):
        if np.isfinite(p_block[b]):
            for j in block.members:
                out[j] = p_block[b]
    return out


def product_shares_from_blocks(
    blocks: list[QualityBlock],
    share_block: np.ndarray,
) -> np.ndarray:
    """Split each quality-block share equally among its tied products.

    Blocks outside the current demand hull have zero block share, so their
    constituent products also receive zero share.
    """
    out = np.zeros(len(PRODUCTS), dtype=float)
    for b, block in enumerate(blocks):
        for j in block.members:
            out[j] = share_block[b] / len(block.members)
    return out


def firm_profits(
    blocks: list[QualityBlock],
    p_block: np.ndarray,
    share_block: np.ndarray,
    t: int,
) -> dict[str, float]:
    p = product_prices_from_blocks(blocks, p_block)
    s = product_shares_from_blocks(blocks, share_block)
    mc = marginal_costs(t)
    retained = np.array(
        [j for block in blocks for j in block.members],
        dtype=int,
    )
    return {
        f: float(np.sum((p[idx] - mc[idx]) * s[idx]))
        for f in np.unique(FIRMS[retained])
        for idx in [retained[FIRMS[retained] == f]]
    }




# ============================================================================
# FOC SYSTEM FOR THE FULL REDUCED SET
# ============================================================================

def _actual_demand_and_jacobian(
    q_blocks: np.ndarray,
    p_blocks: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute shares/Jacobian for the actual hull implied by the prices.

    Returns:
        shares_all: share for every block, zero if block is off the hull
        outside_share: outside-option share
        active: indices of blocks on the actual hull
        J_all: full square Jacobian, padded with zeros for off-hull blocks

    This means we always solve over the full reduced product set, but products
    that are not optimal at the current prices simply receive share zero.
    The resulting FOC system is piecewise smooth at hull changes.
    """
    n = len(q_blocks)
    shares_all = np.zeros(n, dtype=float)
    J_all = np.zeros((n, n), dtype=float)

    active = hull_indices(
        q_blocks,
        p_blocks,
    )

    if len(active) == 0:
        # Defensive fallback for an invalid price configuration.
        # The optimizer will receive a large residual from the FOC wrapper.
        return shares_all, 1.0, active, J_all

    q_active = q_blocks[active]
    p_active = p_blocks[active]

    shares_active, outside = block_demand(
        q_active,
        p_active,
        mu,
    )

    J_active = block_demand_jacobian(
        q_active,
        p_active,
        mu,
    )

    shares_all[active] = shares_active
    J_all[np.ix_(active, active)] = J_active

    return shares_all, outside, active, J_all


def _full_reduced_decisions(
    blocks: list[QualityBlock],
    retained_products: np.ndarray,
    *,
    collusive: bool,
) -> list[dict]:
    """One price decision for every unrestricted block.

    Separate-firm case:
        - maintained tie prices are fixed and not decision variables;
        - each ordinary retained product/block has a free price.

    Collusion case:
        - a2 and b1 form one common-price decision;
        - other unrestricted blocks retain their own price decisions.
    """
    retained = set(int(j) for j in retained_products)
    specs: list[dict] = []

    for b, block in enumerate(blocks):
        if not collusive and block.bertrand_fixed_price is not None:
            continue

        if collusive and block.collusion_pair:
            controller_products = [
                j for j in (A2_INDEX, B1_INDEX)
                if j in retained
            ]
            specs.append({
                "block": b,
                "products": controller_products,
                "kind": "pair",
                "label": "A+B(a2,b1)",
            })
            continue

        # Ordinary single-product block. For the product's firm-level FOC,
        # include all of that firm's retained products' profit components.
        j = block.members[0]
        firm = FIRMS[j]
        firm_products = [
            k for k in retained
            if FIRMS[k] == firm
        ]

        specs.append({
            "block": b,
            "products": firm_products,
            "kind": "firm_product_price",
            "label": str(firm),
        })

    return specs


def _full_reduced_foc(
    x: np.ndarray,
    *,
    t: int,
    mu: float,
    blocks: list[QualityBlock],
    fixed_prices: np.ndarray,
    decision_blocks: np.ndarray,
    specs: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate the FOC vector for the full reduced set at a trial x."""
    q_blocks, _ = block_arrays(blocks)

    p_blocks = fixed_prices.copy()
    p_blocks[decision_blocks] = x

    # Calculate the actual hull, shares and padded Jacobian dynamically.
    shares_block, outside, active, J = _actual_demand_and_jacobian(
        q_blocks,
        p_blocks,
        mu,
    )

    # If no valid hull could be constructed, return a penalty to the solver.
    if len(active) == 0:
        return (
            np.full(len(specs), 1e6),
            p_blocks,
            shares_block,
            outside,
        )

    product_prices = product_prices_from_blocks(
        blocks,
        p_blocks,
    )
    product_shares = product_shares_from_blocks(
        blocks,
        shares_block,
    )
    mc = marginal_costs(t)

    residuals: list[float] = []

    for spec in specs:
        b = int(spec["block"])
        products = np.asarray(spec["products"], dtype=int)

        # d s_j / d p_b, allocating a tied block's derivative equally.
        ds_products = np.zeros(len(PRODUCTS), dtype=float)
        for block_index, block in enumerate(blocks):
            for j in block.members:
                ds_products[j] = (
                    J[block_index, b]
                    / len(block.members)
                )

        # Only the product(s) in the decision block change their prices.
        dp_products = np.zeros(len(PRODUCTS), dtype=float)
        for j in blocks[b].members:
            if j in products:
                dp_products[j] = 1.0

        dprofit = np.sum(
            (product_prices[products] - mc[products])
            * ds_products[products]
        ) + np.sum(
            dp_products[products]
            * product_shares[products]
        )

        residuals.append(float(dprofit))

    return (
        np.asarray(residuals, dtype=float),
        p_blocks,
        shares_block,
        outside,
    )


# ============================================================================
# PUBLIC SOLVER
# ============================================================================

def solve_equilibrium(
    t: int,
    mu: float,
    *,
    collusive: bool,
    verify: bool = True,
) -> Equilibrium:
    """Solve one FOC system over the entire reduced product set.

    Steps:
      1. Apply maintained quality-tie / entry assumptions.
      2. Build the retained quality blocks.
      3. Fix the prices imposed by the tie assumptions.
      4. Put every other block price in one vector of free variables.
      5. Solve all free-price FOCs together with least_squares.
      6. Recompute the actual hull at the resulting prices and assign zero
         share to blocks outside it.

    No subset enumeration and no global deviation check are performed.
    The `verify` argument is retained for compatibility with solve_equilibria.py.
    `valid=True` means only that the FOC residual is sufficiently small.

    Limitation:
      Since demand is recomputed through the upper hull, the FOC function is
      piecewise smooth. Inactive blocks have zero shares and zero local FOCs;
      this simple implementation accepts that outcome as requested.
    """
    del verify  # kept only for compatibility with the existing runner

    retained, dropped = prepare_products(
        t,
        collusive=collusive,
    )
    blocks = build_blocks(
        retained,
        t,
        collusive=collusive,
    )

    q_blocks, mc_blocks = block_arrays(
        blocks
    )

    # Fixed prices from maintained Bertrand tie assumptions.
    fixed_prices = np.full(
        len(blocks),
        np.nan,
        dtype=float,
    )
    if not collusive:
        for b, block in enumerate(blocks):
            if block.bertrand_fixed_price is not None:
                fixed_prices[b] = block.bertrand_fixed_price

    # All unrestricted blocks are solved simultaneously.
    specs = _full_reduced_decisions(
        blocks,
        retained,
        collusive=collusive,
    )
    decision_blocks = np.asarray(
        [spec["block"] for spec in specs],
        dtype=int,
    )

    # Set an initial price for every block, including fixed-price blocks.
    # Initial free prices are quality-squared, nudged above their own MC.
    p0 = np.maximum(
        q_blocks**2,
        mc_blocks + 0.1,
    )

    for b in range(len(blocks)):
        if np.isfinite(fixed_prices[b]):
            p0[b] = fixed_prices[b]

    x0 = p0[decision_blocks]

    if len(decision_blocks) == 0:
        x_star = np.array([], dtype=float)
        p_star = fixed_prices.copy()
        shares, outside, active, _ = _actual_demand_and_jacobian(
            q_blocks,
            p_star,
            mu,
        )
        residual = 0.0
    else:
        lower = np.full(
            len(decision_blocks),
            1e-6,
        )
        upper = np.maximum(
            25.0,
            8.0 * q_blocks[decision_blocks] + 10.0,
        )

        def foc(x: np.ndarray) -> np.ndarray:
            residuals, _, _, _ = _full_reduced_foc(
                x,
                t=t,
                mu=mu,
                blocks=blocks,
                fixed_prices=fixed_prices,
                decision_blocks=decision_blocks,
                specs=specs,
            )
            return residuals

        # Try several starting values for the SAME full reduced-set FOC system.
        # No subsets are enumerated. We keep the solution with the smallest
        # residual, since the demand hull can change as prices move.
        q_free = q_blocks[decision_blocks]
        mc_free = mc_blocks[decision_blocks]
        starts = [
            x0,
            np.maximum(mc_free + 0.5, 0.5),
            np.maximum(2.0 * q_free + 0.2, mc_free + 0.1),
            np.maximum(3.0 * q_free, mc_free + 0.1),
            np.maximum(5.0 * q_free, mc_free + 0.1),
            np.maximum(1.5 * q_free**2, mc_free + 0.1),
        ]

        best = None
        errors = []

        for start in starts:
            start = np.clip(start, lower + 1e-8, upper - 1e-8)
            try:
                result = least_squares(
                    foc,
                    start,
                    bounds=(lower, upper),
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10,
                    max_nfev=2000,
                )
                rr = float(np.linalg.norm(foc(result.x)))
                if best is None or rr < best[0]:
                    best = (rr, result.x)
            except Exception as exc:
                errors.append(str(exc))

        if best is None:
            raise RuntimeError(
                f"FOC solver failed for t={t}, mu={mu}, "
                f"collusive={collusive}. Details: {'; '.join(errors[:3])}"
            )

        residual, x_star = best

        residuals, p_star, shares, outside = _full_reduced_foc(
            x_star,
            t=t,
            mu=mu,
            blocks=blocks,
            fixed_prices=fixed_prices,
            decision_blocks=decision_blocks,
            specs=specs,
        )

    # Recompute final hull using the solved price for every retained block.
    active = hull_indices(
        q_blocks,
        p_star,
    )
    shares, outside, active, _ = _actual_demand_and_jacobian(
        q_blocks,
        p_star,
        mu,
    )

    eq = Equilibrium(
        t=t,
        mu=mu,
        case="collusive" if collusive else "separate",
        p_block=p_star,
        share_block=shares,
        outside_share=float(outside),
        active_blocks=active,
        blocks=blocks,
        foc_norm=residual,
        max_deviation_gain=np.nan,
        deviation_controller="not checked",
        deviation_profit=np.nan,
        current_profit=np.nan,
        valid=(residual <= 1e-5),
    )

    if not eq.valid:
        # Return the numerical candidate with diagnostics rather than silently
        # switching to a smaller active set. The caller can inspect foc_norm.
        print(
            f"WARNING: FOC norm {residual:.3e} exceeds tolerance for "
            f"t={t}, mu={mu}, collusive={collusive}."
        )

    return eq
