"""Two-step estimation: demand NLLS, then marginal-cost inversion.

Run from the project root:
    ./.venv/bin/python code/estimate_two_step.py

Observed inside shares are stored in qjt. The script estimates demand first,
then computes ds/dp with JAX and inverts the Bertrand and collusive ownership
matrices wherever the product-level inversion is well-defined. No bootstrap or
parameter restrictions are used. Exact ties are reported as not identified by
the smooth product-level inversion rather than assigned arbitrary costs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr as scipy_ndtr

HERE = Path(__file__).resolve().parent
if (HERE / "output").is_dir() and (HERE / "bikes_model.py").exists():
    ROOT = HERE
elif (HERE.parent / "output").is_dir():
    ROOT = HERE.parent
else:
    ROOT = HERE
MODEL_DIR = ROOT / "code" if (ROOT / "code" / "bikes_model.py").exists() else ROOT
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import bikes_model as bm  # noqa: E402

PRODUCTS = [str(x) for x in bm.PRODUCTS]
PRODUCT_INDEX = {name: i for i, name in enumerate(PRODUCTS)}
MUS = (-0.5, 0.0, 0.5)
MU_INDEX = {mu: i for i, mu in enumerate(MUS)}
DATASETS = [
    ("Bertrand data", -0.5, "bikes_1.csv"),
    ("Bertrand data", 0.0, "bikes_2.csv"),
    ("Bertrand data", 0.5, "bikes_3.csv"),
    ("Collusion data", -0.5, "bikes_collusion_1.csv"),
    ("Collusion data", 0.0, "bikes_collusion_2.csv"),
    ("Collusion data", 0.5, "bikes_collusion_3.csv"),
]
# Starting values only: beta_w, beta_g, mu(-.5), mu(0), mu(.5).
# Every component is freely optimized; no equality restrictions are imposed.
INITIAL_GUESS = np.array([-0.07, 0.14, 0.0, 0.0, 0.0], dtype=float)
TOL = 1e-8
CONDITION_LIMIT = 1e12


def lognormal_cdf(a: float, mu: float) -> float:
    if a <= 0:
        return 0.0
    if np.isinf(a):
        return 1.0
    return float(scipy_ndtr(np.log(a) - mu))


def demand_shares(theta, w, g, p) -> np.ndarray:
    """Model-implied inside shares at observed prices; outside share is omitted."""
    beta_w, beta_g, mu = np.asarray(theta, dtype=float)
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.exp(beta_w * w + beta_g * g)
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise FloatingPointError("Invalid trial parameters or prices")

    # Combine identical (quality, price) options and split their common share.
    blocks: list[list[int]] = []
    for j in range(len(p)):
        match = None
        for b, members in enumerate(blocks):
            k = members[0]
            qtol = TOL * max(1.0, abs(q[j]), abs(q[k]))
            ptol = TOL * max(1.0, abs(p[j]), abs(p[k]))
            if abs(q[j] - q[k]) <= qtol and abs(p[j] - p[k]) <= ptol:
                match = b
                break
        if match is None:
            blocks.append([j])
        else:
            blocks[match].append(j)

    qb = np.array([np.mean(q[m]) for m in blocks])
    pb = np.array([np.mean(p[m]) for m in blocks])
    sb = np.zeros(len(blocks), dtype=float)
    for j, (qj, pj) in enumerate(zip(qb, pb)):
        lower, upper = 0.0, qj / pj
        feasible = True
        for k, (qk, pk) in enumerate(zip(qb, pb)):
            if j == k:
                continue
            dp, dq = pj - pk, qj - qk
            if abs(dp) <= TOL * max(1.0, abs(pj), abs(pk)):
                if dq < -TOL * max(1.0, abs(qj), abs(qk)):
                    feasible = False
                    break
            elif dp > 0:
                upper = min(upper, dq / dp)
            else:
                lower = max(lower, dq / dp)
        lower = max(0.0, lower)
        if feasible and upper > lower:
            sb[j] = max(0.0, lognormal_cdf(upper, mu) - lognormal_cdf(lower, mu))

    out = np.zeros(len(p), dtype=float)
    for b, members in enumerate(blocks):
        out[members] = sb[b] / len(members)
    return out


def load_markets(path: Path) -> list[dict]:
    """Load market shares from qjt. The column sjt is never read."""
    frame = pd.read_csv(path)
    required = {"j", "t", "qjt", "pjt", "wj", "gj", "ct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    shares_all = pd.to_numeric(frame["qjt"], errors="coerce").to_numpy(float)
    if np.any(~np.isfinite(shares_all)) or np.any(shares_all < -1e-8) or np.any(shares_all > 1 + 1e-8):
        raise ValueError(
            f"{path.name}: qjt must be market shares in [0,1]. "
            "Rerun the corrected solve_equilibria.py if this file still stores quality in qjt."
        )

    markets = []
    for t, group in frame.groupby("t", sort=True):
        part = group.set_index("j").reindex(PRODUCTS)
        p = pd.to_numeric(part["pjt"], errors="coerce").to_numpy(float)
        share = pd.to_numeric(part["qjt"], errors="coerce").to_numpy(float)
        w = pd.to_numeric(part["wj"], errors="coerce").to_numpy(float)
        g = pd.to_numeric(part["gj"], errors="coerce").to_numpy(float)
        ct = float(pd.to_numeric(part["ct"], errors="coerce").iloc[0])
        missing_price = ~np.isfinite(p)
        if np.any(np.abs(share[missing_price]) > 1e-7):
            raise ValueError(f"{path.name}, t={t}: a product has positive qjt but no pjt.")
        if np.sum(share) > 1 + 1e-7:
            raise ValueError(f"{path.name}, t={t}: inside shares sum to more than one.")
        markets.append({"t": int(t), "p": p, "share": share, "w": w, "g": g, "ct": ct})
    return markets


def predict_full_shares(theta, market: dict) -> np.ndarray:
    predicted = np.zeros(len(PRODUCTS), dtype=float)
    priced = np.isfinite(market["p"])
    if np.any(priced):
        predicted[priced] = demand_shares(
            theta,
            market["w"][priced],
            market["g"][priced],
            market["p"][priced],
        )
    return predicted


def fit_demand_regime(markets_by_mu: dict[float, list[dict]]) -> dict:
    """Pool the three mu scenarios; beta parameters common, each mu is free."""
    observations = [(mu, market) for mu in MUS for market in markets_by_mu[mu]]

    def unpack(x):
        return {mu: np.array([x[0], x[1], x[2 + MU_INDEX[mu]]], dtype=float) for mu in MUS}

    def residuals(x):
        params = unpack(x)
        out = []
        try:
            for mu, market in observations:
                out.extend(predict_full_shares(params[mu], market) - market["share"])
        except (FloatingPointError, ValueError, OverflowError):
            return np.full(len(observations) * len(PRODUCTS), 1e4 + np.linalg.norm(x))
        return np.asarray(out, dtype=float)

    start_residuals = residuals(INITIAL_GUESS)
    fit = least_squares(
        residuals,
        x0=INITIAL_GUESS,
        max_nfev=1000,
        x_scale="jac",
        diff_step=1e-6,
    )
    final_residuals = residuals(fit.x)
    return {
        "x": np.asarray(fit.x, dtype=float),
        "params_by_mu": unpack(fit.x),
        "start_SSE": float(start_residuals @ start_residuals),
        "SSE": float(final_residuals @ final_residuals),
        "RMSE": float(np.sqrt(np.mean(final_residuals ** 2))),
        "n_obs": len(final_residuals),
        "success": bool(fit.success),
        "message": str(fit.message),
        "nfev": int(fit.nfev),
    }


def shares_and_reduced_jacobian_jax(
    q: np.ndarray,
    p: np.ndarray,
    mu: float,
    tied_mask: np.ndarray,
):
    """Compute shares and ds/dp for non-tied-price products.

    Equal-price products remain in the choice menu as fixed-price options,
    but their rows and columns are excluded from the supply inversion. Exact
    duplicate (quality, price) options are collapsed into one demand block.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    tied_mask = np.asarray(tied_mask, dtype=bool)
    free_ids = np.flatnonzero(~tied_mask)

    # Collapse exact (quality, price) duplicates only for demand evaluation.
    menu_blocks: list[dict] = []
    for i in range(len(q)):
        found = None
        for b, block in enumerate(menu_blocks):
            k = block["members"][0]
            qtol = TOL * max(1.0, abs(q[i]), abs(q[k]))
            ptol = TOL * max(1.0, abs(p[i]), abs(p[k]))
            if abs(q[i] - q[k]) <= qtol and abs(p[i] - p[k]) <= ptol:
                found = b
                break
        if found is None:
            menu_blocks.append({"members": [i], "q": q[i], "p": p[i]})
        else:
            menu_blocks[found]["members"].append(i)

    # The hull routine expects options ordered by quality.
    menu_blocks.sort(key=lambda b: b["q"])
    qb = np.asarray([b["q"] for b in menu_blocks], dtype=float)
    pb = np.asarray([b["p"] for b in menu_blocks], dtype=float)
    hull_local = bm.hull_indices(qb, pb)
    if len(hull_local) == 0:
        raise ValueError("Cannot construct a valid demand hull at these prices.")
    hull_blocks = [menu_blocks[int(i)] for i in hull_local]
    qh = np.asarray([b["q"] for b in hull_blocks], dtype=float)
    ph = np.asarray([b["p"] for b in hull_blocks], dtype=float)

    # Map each non-tied product to a hull block if it lies on the hull.
    free_position = {int(product_id): k for k, product_id in enumerate(free_ids)}
    block_for_free = np.full(len(free_ids), -1, dtype=int)
    free_at_hull = np.full(len(hull_blocks), -1, dtype=int)
    for h, block in enumerate(hull_blocks):
        # A block with an equal-price duplicate is tied and therefore fixed.
        members = block["members"]
        if len(members) == 1:
            i = int(members[0])
            if i in free_position:
                pos = free_position[i]
                free_at_hull[h] = pos
                block_for_free[pos] = h

    qj = jnp.asarray(qh, dtype=jnp.float64)
    phj = jnp.asarray(ph, dtype=jnp.float64)
    free_at_hull_j = jnp.asarray(free_at_hull, dtype=jnp.int32)
    block_for_free_j = jnp.asarray(block_for_free, dtype=jnp.int32)

    def share_fn(free_prices):
        # Keep equal-price blocks fixed, varying only products included in the
        # reduced marginal-cost inversion.
        prices_h = phj
        for h in range(len(hull_blocks)):
            pos = int(free_at_hull[h])
            if pos >= 0:
                prices_h = prices_h.at[h].set(free_prices[pos])

        q0 = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), qj))
        p0 = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), prices_h))
        alpha = jnp.diff(q0) / jnp.diff(p0)
        cdf = ndtr(jnp.log(alpha) - mu)
        block_shares = cdf[-1:] if qj.shape[0] == 1 else jnp.concatenate((cdf[:-1] - cdf[1:], cdf[-1:]))

        result = jnp.zeros((len(free_ids),), dtype=jnp.float64)
        for k in range(len(free_ids)):
            h = int(block_for_free[k])
            if h >= 0:
                result = result.at[k].set(block_shares[h])
        return result

    free_p = p[free_ids]
    free_shares = np.asarray(share_fn(jnp.asarray(free_p, dtype=jnp.float64)), dtype=float)
    delta = np.asarray(jax.jacrev(share_fn)(jnp.asarray(free_p, dtype=jnp.float64)), dtype=float)
    return free_ids, free_shares, delta


def ownership_matrix(names: list[str], collusive: bool) -> np.ndarray:
    idx = [PRODUCT_INDEX[name] for name in names]
    ownership = np.array([[float(bm.FIRMS[i] == bm.FIRMS[j]) for j in idx] for i in idx])
    if collusive and "a2" in names and "b1" in names:
        i, j = names.index("a2"), names.index("b1")
        ownership[i, j] = ownership[j, i] = 1.0
    return ownership


def invert_market(theta: np.ndarray, market: dict, source_regime: str, mu_label: float):
    """Set mc=p for equal-price products and invert only the other products."""
    active_ids = np.flatnonzero(np.isfinite(market["p"]))
    names_all = [PRODUCTS[i] for i in active_ids]
    p_all = market["p"][active_ids]
    s_all = market["share"][active_ids]
    w_all, g_all = market["w"][active_ids], market["g"][active_ids]
    beta_w, beta_g, mu_hat = theta
    q_all = np.exp(beta_w * w_all + beta_g * g_all)

    # Any products sharing a price are assigned mc=p and excluded from the
    # ownership-adjusted Jacobian inversion, as requested.
    tied_mask = np.zeros(len(p_all), dtype=bool)
    for i in range(len(p_all)):
        for j in range(i + 1, len(p_all)):
            ptol = TOL * max(1.0, abs(p_all[i]), abs(p_all[j]))
            if abs(p_all[i] - p_all[j]) <= ptol:
                tied_mask[i] = True
                tied_mask[j] = True

    free_ids = np.flatnonzero(~tied_mask)
    costs, diagnostics = [], []

    # Compute the reduced demand Jacobian once. The tied-price menu options
    # remain in demand at their observed prices, but are not free price choices.
    reduced_share = reduced_delta = None
    jacobian_error = None
    if len(free_ids) > 0:
        try:
            returned_ids, reduced_share, reduced_delta = shares_and_reduced_jacobian_jax(
                q_all, p_all, float(mu_hat), tied_mask
            )
            if not np.array_equal(returned_ids, free_ids):
                raise RuntimeError("Internal product ordering mismatch in reduced inversion.")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
            jacobian_error = str(exc)

    for collusive in (False, True):
        assumption = "Collusive ownership" if collusive else "Bertrand ownership"
        mc_all = np.full(len(p_all), np.nan)
        mc_all[tied_mask] = p_all[tied_mask]
        rank = 0
        condition = np.nan
        inv_status = "no untied-price products" if len(free_ids) == 0 else "ok"

        if len(free_ids) > 0:
            if jacobian_error is not None:
                inv_status = "reduced inversion failed"
            else:
                try:
                    free_names = [names_all[i] for i in free_ids]
                    p_free = p_all[free_ids]
                    s_free = s_all[free_ids]
                    O = ownership_matrix(free_names, collusive=collusive)
                    A = O * reduced_delta.T
                    rank = int(np.linalg.matrix_rank(A))
                    condition = float(np.linalg.cond(A))
                    if rank == len(free_ids) and np.isfinite(condition) and condition < CONDITION_LIMIT:
                        mc_all[free_ids] = p_free + np.linalg.solve(A, s_free)
                        inv_status = "ok"
                    else:
                        mc_all[free_ids] = p_free + np.linalg.pinv(A, rcond=1e-10) @ s_free
                        inv_status = "pseudoinverse used"
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    inv_status = "reduced inversion failed"

        for i, name in enumerate(names_all):
            costs.append({
                "source_regime": source_regime,
                "mu_scenario": mu_label,
                "t": market["t"],
                "product": name,
                "ownership_assumption": assumption,
                "price": float(p_all[i]),
                "observed_share_qjt": float(s_all[i]),
                "mc_hat": float(mc_all[i]) if np.isfinite(mc_all[i]) else np.nan,
            })
        diagnostics.append({
            "source_regime": source_regime,
            "mu_scenario": mu_label,
            "t": market["t"],
            "ownership_assumption": assumption,
            "n_priced_products": len(names_all),
            "n_equal_price_products_set_mc_eq_p": int(np.sum(tied_mask)),
            "reduced_matrix_rank": rank,
            "reduced_matrix_condition_number": condition,
            "inversion_status": inv_status,
        })
    return costs, diagnostics


def markdown_table(headers, rows):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.10g}" if np.isfinite(v) else "NA"
        return str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def main():
    # Keep all estimation results in memory. The only files written are the
    # two Markdown tables intended for inclusion in the QMD.
    outdir = ROOT / "output"
    all_data = {"Bertrand data": {}, "Collusion data": {}}
    for regime, mu, filename in DATASETS:
        all_data[regime][mu] = load_markets(outdir / filename)

    demand_rows = []
    cost_rows = []
    for regime, markets_by_mu in all_data.items():
        fit = fit_demand_regime(markets_by_mu)
        for mu in MUS:
            theta = fit["params_by_mu"][mu]
            sse = 0.0
            n = 0
            for market in markets_by_mu[mu]:
                predicted = predict_full_shares(theta, market)
                residual = predicted - market["share"]
                sse += float(residual @ residual)
                n += residual.size

                recovered, _ = invert_market(theta, market, regime, mu)
                cost_rows.extend(recovered)

            demand_rows.append({
                "Data": regime,
                "Scenario mu": mu,
                "beta_w": theta[0],
                "beta_g": theta[1],
                "mu_hat": theta[2],
                "Share RMSE": np.sqrt(sse / n) if n else np.nan,
            })

    # Regress recovered marginal costs on the specified cost shifters, separately
    # for each source dataset and ownership assumption.
    gamma_rows = []
    ct_by_t = {int(t): 0.5 + float(t) / 10.0 for t in range(1, 11)}
    wi = {product: float(bm.W[i]) for i, product in enumerate(PRODUCTS)}
    gi = {product: float(bm.G[i]) for i, product in enumerate(PRODUCTS)}
    costs_df = pd.DataFrame(cost_rows)
    for (regime, assumption), group in costs_df.groupby(
        ["source_regime", "ownership_assumption"], sort=False
    ):
        recovered = group.dropna(subset=["mc_hat"])
        X = np.asarray([
            [(1.0 - wi[row.product]) * ct_by_t[int(row.t)], gi[row.product]]
            for row in recovered.itertuples()
        ], dtype=float)
        y = recovered["mc_hat"].to_numpy(float)
        if len(y) >= 2 and np.linalg.matrix_rank(X) == 2:
            gamma = np.linalg.lstsq(X, y, rcond=None)[0]
            rmse = float(np.sqrt(np.mean((X @ gamma - y) ** 2)))
        else:
            gamma = np.array([np.nan, np.nan])
            rmse = np.nan
        gamma_rows.append({
            "Data": regime,
            "Ownership": assumption,
            "gamma_w": gamma[0],
            "gamma_g": gamma[1],
            "Recovered MCs": len(y),
            "Cost RMSE": rmse,
        })

    demand_table = [[
        row["Data"], row["Scenario mu"], row["beta_w"], row["beta_g"],
        row["mu_hat"], row["Share RMSE"]
    ] for row in demand_rows]
    supply_table = [[
        row["Data"], row["Ownership"], row["gamma_w"], row["gamma_g"],
        row["Recovered MCs"], row["Cost RMSE"]
    ] for row in gamma_rows]

    (outdir / "demand_estimates.md").write_text(
        markdown_table(
            ["Data", r"Scenario $\mu$", r"$\hat{\beta}_w$", r"$\hat{\beta}_g$", r"$\hat{\mu}$", "Share RMSE"],
            demand_table,
        ) + "\n",
        encoding="utf-8",
    )
    (outdir / "supply_estimates.md").write_text(
        markdown_table(
            ["Data", "Ownership", r"$\hat{\gamma}_w$", r"$\hat{\gamma}_g$", "Recovered MCs", "Cost RMSE"],
            supply_table,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote output/demand_estimates.md and output/supply_estimates.md")


if __name__ == "__main__":
    main()
