
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)
CHARS = ["SpicyIndex", "WheatContent", "Veg2meat", "Calories"]
COST = ["VegPIDX", "WheatPIDX", "MeatPIDX", "WageIDX", "ReEsIDX"]
CAMPUS_MAP = {
    "St Johns": "Saint John", "Texas-A&M": "Texas A&M",
    "R-H": "Rose-Hul.", "U. Chicago": "Univ. Chicago"
}


def input_folder(directory=None):
    if directory:
        return Path(directory).expanduser().resolve()
    for root in [Path.cwd(), *Path.cwd().parents]:
        for sub in ["data/input", "pset_2/data/input"]:
            folder = root / sub
            if (folder / "choice_noodles.csv").exists() and (
                (folder / "college-year.xls").exists() or
                (folder / "college-year.csv").exists()
            ):
                return folder
    folder = Path.home() / "Documents/GitHub/IO/pset_2/data/input"
    if (folder / "choice_noodles.csv").exists():
        return folder
    raise FileNotFoundError("Cannot find input files. Set --data-dir.")


def load_data(directory):
    choice = pd.read_csv(directory / "choice_noodles.csv")
    csv_path, xls_path = directory / "college-year.csv", directory / "college-year.xls"
    if xls_path.exists(): college = pd.read_excel(xls_path, engine="xlrd")
    elif csv_path.exists(): college = pd.read_csv(csv_path)
    else: raise FileNotFoundError("college-year.xls not found")
    college["year"] = college["year"].astype(int)
    college["campus"] = college["campus"].replace(CAMPUS_MAP)
    data = choice.merge(
        college, left_on=["University", "year"],
        right_on=["campus", "year"],
        validate="many_to_one", how="left"
    )
    required = COST + ["Share_athletes", "Share_Graduates"]
    if len(data) != len(choice) or data[required].isna().any().any():
        raise ValueError("Campus names or years do not match across the two datasets.")
    if (data["Share"] <= 0).any(): raise ValueError("All observed inside shares must be positive.")
    data = data.sort_values(["University", "year", "dish"]).reset_index(drop=True)
    key = pd.MultiIndex.from_arrays([data.University, data.year])
    data["market"] = pd.factorize(key, sort=False)[0]
    total = data.groupby("market")["Share"].sum()
    if (total >= 1).any():
        raise ValueError("Outside option must have positive share.")
    return data


def prepare(data):
    n, ids = len(data), data["market"].to_numpy()
    M, J = data["market"].nunique(), int(data.groupby("market").size().max())
    counts = data.groupby("market").size().to_numpy()
    offsets = np.cumsum(np.r_[0, counts[:-1]])
    active = np.column_stack([ids, np.arange(n) - offsets[ids]])
    mask = np.zeros((M, J), dtype=bool)
    mask[active[:, 0], active[:, 1]] = True
    chars = data[CHARS].to_numpy(float)
    price = data["Price"].to_numpy(float)
    observed = data["Share"].to_numpy(float)
    X, S, initial = np.zeros((M, J, 4)), np.zeros((M, J)), np.zeros((M, J))
    X[active[:, 0], active[:, 1]] = chars
    S[active[:, 0], active[:, 1]] = observed
    outside = 1 - np.bincount(ids, weights=observed, minlength=M)
    initial[active[:, 0], active[:, 1]] = np.log(observed / outside[ids])

    # Assume independence of athlete and graduate status in each market to obtain their joint shares.
    a, g = data.groupby("market")[["Share_athletes", "Share_Graduates"]].first().to_numpy().T
    if np.any((a < 0) | (a > 1) | (g < 0) | (g > 1)):
        raise ValueError("Demographic shares outside [0,1].")
    weights = np.stack([(1-a)*(1-g), a*(1-g), (1-a)*g, a*g], axis=1)
    dish_names = sorted(data["dish"].unique())
    dish_fe = pd.get_dummies(data["dish"], dtype=float).reindex(dish_names, axis=1).to_numpy()

    shocks = data[COST].to_numpy(float)
    market_sum, same_chain = np.zeros((M, 4)), np.zeros((n, 4))
    np.add.at(market_sum, ids, chars)
    for (_, chain), group in data.groupby(["market", "chain"], sort=False):
        idx = group.index.to_numpy()
        same_chain[idx] = chars[idx].sum(axis=0) - chars[idx]
    other_chain = market_sum[ids] - chars - same_chain
    exposure = np.column_stack([
        data.VegPIDX * data.Veg2meat,
        data.MeatPIDX * (1 - data.Veg2meat),
        data.WheatPIDX * data.WheatContent,
        data.WageIDX * data.Calories,
        data.ReEsIDX * data.Calories
    ])
    raw_Z = np.column_stack([dish_fe, chars, shocks, exposure, same_chain, other_chain])
    U, singular, _ = svd(raw_Z, full_matrices=False)
    keep = singular > singular[0]*1e-10
    Z = U[:, keep]*np.sqrt(n)  # Orthonormal basis of exactly the same instrument span.
    print(f"Products: {n}; markets: {M}; instruments: {Z.shape[1]}", flush=True)
    return dict(df=data, M=M, J=J, n=n, X=X, S=S, mask=mask, initial=initial,
                weights=weights, Z=Z, D=dish_fe, chars=chars, price=price, dish_names=dish_names)


def normal_draws(count, seed):
    count = 2**int(np.ceil(np.log2(count)))
    uniform = qmc.Sobol(4, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    return norm.ppf(np.clip(uniform, 1e-10, 1-1e-10))


class DemandModel:
    def __init__(self, data, name, draws):
        self.data, self.name = data, name
        self.W = np.column_stack([data["D"], *([data["chars"]] if name == "a" else []), data["price"]])
        self.Z, self.n = data["Z"], data["n"]
        self.ZW = self.Z.T @ self.W / self.n
        self.mask = np.asarray(data["mask"])
        self.weight = np.eye(self.Z.shape[1])
        self.cache_theta, self.cache_answer = None, None
        self._compile_jax(draws)

    def _compile_jax(self, draws):
        name = self.name
        x, mask = jnp.asarray(self.data["X"]), jnp.asarray(self.data["mask"])
        observed, initial = jnp.asarray(self.data["S"]), jnp.asarray(self.data["initial"])
        group_weights = jnp.asarray(self.data["weights"])
        nu = np.asarray(draws if name != "b" else np.zeros((1, 4)))
        flags = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        if name == "a": group_weights, flags = jnp.ones((self.data["M"], 1)), np.zeros((1, 2))
        num_types, num_draws = len(flags), len(nu)
        num_params = {"a": 4, "b": 8, "c": 12, "d": 15}[name]
        # d beta / d theta is constant for every draw, including the Cholesky entries.
        db = np.zeros((num_types, num_draws, 4, num_params))
        for t in range(num_types):
            for r in range(num_draws):
                if name != "a":
                    for k in range(4):
                        db[t, r, k, k] = flags[t, 0]
                        db[t, r, k, k+4] = flags[t, 1]
                if name in ("a", "c"):
                    for k in range(4): db[t, r, k, (0 if name == "a" else 8)+k] = nu[r, k]
                if name == "d":
                    for param, row, col in [(8,0,0), (9,1,1), (10,2,1), (11,2,2), (12,3,1), (13,3,2), (14,3,3)]:
                        db[t, r, row, param] = nu[r, col]
        db, nu, flags = jnp.asarray(db), jnp.asarray(nu), jnp.asarray(flags)

        def cholesky(theta):
            if name in ("a", "c"): return jnp.diag(theta[-4:])
            if name == "b": return jnp.zeros((4, 4))
            z = theta[-7:]
            return jnp.array([[z[0], 0., 0., 0.], [0., z[1], 0., 0.],
                              [0., z[2], z[3], 0.], [0., z[4], z[5], z[6]]])

        def shares_and_jacobian(delta, theta):
            random_beta = nu @ cholesky(theta).T
            if name == "a": means = jnp.zeros((1, 4))
            else: means = flags[:, 0, None]*theta[None, :4] + flags[:, 1, None]*theta[None, 4:8]
            beta = means[:, None, :] + random_beta[None, :, :]
            utility = delta[:, :, None, None] + jnp.einsum("mjk,trk->mjtr", x, beta)
            utility = jnp.where(mask[:, :, None, None], utility, -jnp.inf)
            shift = jnp.maximum(0., jnp.max(utility, axis=1, keepdims=True))
            expu = jnp.exp(utility - shift)
            probability = expu / (jnp.exp(-shift) + jnp.sum(expu, axis=1, keepdims=True))
            weight = jnp.broadcast_to(group_weights[:, :, None]/num_draws, (x.shape[0], num_types, num_draws))
            shares = jnp.einsum("mjtr,mtr->mj", probability, weight)
            cross = jnp.einsum("mjtr,mktr,mtr->mjk", probability, probability, weight)
            jacobian = jax.vmap(jnp.diag)(shares) - cross + jax.vmap(jnp.diag)(~mask)
            return shares, jacobian, probability, weight

        def invert_shares(theta):
            def newton_step(state):
                iteration, delta, err = state
                predicted, jacobian, _, _ = shares_and_jacobian(delta, theta)
                correction = jnp.linalg.solve(jacobian, (predicted-observed)[..., None])[..., 0]
                delta = jnp.where(mask, delta-jnp.clip(correction, -4., 4.), 0.)
                return iteration+1, delta, jnp.max(jnp.abs(predicted-observed))
            _, delta, _ = jax.lax.while_loop(
                lambda state: (state[0] < 100) & (state[2] > 1e-11),
                newton_step, (0, initial, jnp.inf))
            predicted, jacobian, probability, weight = shares_and_jacobian(delta, theta)
            mean_chars = jnp.einsum("mjtr,mjk->mtrk", probability, x)
            centered = x[:, :, None, None, :] - mean_chars[:, None, :, :, :]
            ds_dtheta = jnp.einsum("mjtr,mjtrk,trkp,mtr->mjp", probability, centered, db, weight)
            ddelta_dtheta = -jnp.linalg.solve(jacobian, ds_dtheta)
            return delta, ddelta_dtheta, jnp.max(jnp.abs(predicted-observed))
        self.invert = jax.jit(invert_shares)

    def evaluate(self, theta, details=False):
        if self.cache_theta is not None and np.array_equal(theta, self.cache_theta) and not details:
            return self.cache_answer
        delta, derivative, err = self.invert(jnp.asarray(theta))
        full_delta, full_derivative = np.asarray(delta), np.asarray(derivative)
        delta, derivative = full_delta[self.mask], full_derivative[self.mask]
        if not np.isfinite(delta).all() or float(err) > 1e-8:
            raise RuntimeError(f"BLP share inversion did not converge: maximum error {float(err)}")
        Zdelta, Zdtheta = self.Z.T @ delta/self.n, self.Z.T @ derivative/self.n
        gamma = np.linalg.pinv(self.ZW.T @ self.weight @ self.ZW, rcond=1e-12) @ (self.ZW.T @ self.weight @ Zdelta)
        moment = Zdelta - self.ZW @ gamma
        loss = float(moment @ self.weight @ moment)
        gradient = 2*Zdtheta.T @ self.weight @ moment
        if details: return dict(obj=loss, grad=gradient, gamma=gamma, moment=moment,
                                resid=delta-self.W @ gamma, max_share_error=float(err),
                                delta_full=full_delta, ddelta_full=full_derivative)
        self.cache_theta, self.cache_answer = theta.copy(), (loss, gradient)
        return loss, gradient

    def second_step_weight(self, theta):
        residual = self.evaluate(theta, details=True)["resid"]
        market = self.data["df"]["market"].to_numpy()
        scores = np.zeros((self.data["M"], self.Z.shape[1]))
        np.add.at(scores, market, self.Z*residual[:, None])
        omega = scores.T @ scores / self.n
        ridge = 1e-3*np.trace(omega)/len(omega)
        self.weight = np.linalg.inv(omega + ridge*np.eye(len(omega)))
        self.weight *= len(omega)/np.trace(self.weight)
        self.cache_theta, self.cache_answer = None, None


def initial_values(name, previous, seed, starts):
    rng = np.random.default_rng(seed)
    if name == "a":
        first, bounds = np.array([.3, .3, .3, .3]), [(0., 5.)]*4
    elif name == "b":
        first = np.r_[[.15, .2, .1, .1], [.1, .2, .1, .1]]
        bounds = [(-6., 6.)]*8
    elif name == "c":
        first = np.r_[previous["b"][:8] if "b" in previous else np.zeros(8), previous.get("a", np.array([.35]*4))]
        bounds = [(-6., 6.)]*8 + [(0., 5.)]*4
    else:
        old = previous.get("c", np.r_[np.zeros(8), [.3]*4])
        first = np.r_[old[:8], [max(old[8], .05), max(old[9], .05), 0., max(old[10], .05), 0., 0., max(old[11], .05)]]
        bounds = [(-6., 6.)]*8 + [(0., 5.), (0., 5.), (-5., 5.), (0., 5.), (-5., 5.), (-5., 5.), (0., 5.)]
    points = [first]
    for _ in range(starts-1):
        guess = first.copy()
        if name == "a": guess = rng.uniform(.1, 1.5, 4)
        elif name == "b": guess = rng.normal(0., .65, 8)
        elif name == "c":
            guess[:8] += rng.normal(0., .5, 8)
            guess[8:] = rng.uniform(.1, 1.2, 4)
        else:
            guess[:8] += rng.normal(0., .25, 8)
            guess[8:] += rng.normal(0., .15, 7)
            guess[8:][[0,1,3,6]] = np.maximum(.05, guess[8:][[0,1,3,6]])
        points.append(np.clip(guess, [b[0] for b in bounds], [b[1] for b in bounds]))
    return points, bounds


def summarize(model, theta, fit):
    gamma = fit["gamma"]
    out = dict(objective=fit["obj"], moment_norm=float(np.linalg.norm(fit["moment"])),
               max_share_error=fit["max_share_error"], price=float(gamma[-1]),
               dish_fe=dict(zip(model.data["dish_names"], map(float, gamma[:9]))))
    if model.name == "a":
        out["beta_mean"] = dict(zip(CHARS, map(float, gamma[9:13])))
        L = np.diag(theta)
    else:
        out["beta_athlete"] = dict(zip(CHARS, map(float, theta[:4])))
        out["beta_graduate"] = dict(zip(CHARS, map(float, theta[4:8])))
        L = np.zeros((4,4)) if model.name == "b" else (
            np.diag(theta[8:]) if model.name == "c" else np.array([
                [theta[8],0.,0.,0.], [0.,theta[9],0.,0.],
                [0.,theta[10],theta[11],0.], [0.,theta[12],theta[13],theta[14]]]))
    covariance = L @ L.T
    out["sd"] = dict(zip(CHARS, map(float, np.sqrt(np.diag(covariance)))))
    out["covariance"] = covariance.tolist()
    if model.name == "d":
        sd = np.sqrt(np.diag(covariance))
        den = np.outer(sd, sd)
        out["correlation"] = [[float(covariance[i,j]/den[i,j]) if den[i,j] > 1e-12 else None
                               for j in range(4)] for i in range(4)]
    return out


def utilities(data, name, theta, delta, draws, seed=2026):
    """Conditional systematic utilities for all student types and Sobol draws."""
    if name in ("a", "c"):
        L = np.diag(theta[-4:])
    elif name == "b":
        L = np.zeros((4, 4))
    else:
        L = np.array([[theta[8], 0, 0, 0], [0, theta[9], 0, 0],
                      [0, theta[10], theta[11], 0],
                      [0, theta[12], theta[13], theta[14]]])
    nu = normal_draws(draws, seed) if name != "b" else np.zeros((1, 4))
    beta = nu @ L.T
    types = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    if name == "a":
        beta = np.broadcast_to(beta, (4, *beta.shape))
    else:
        beta = beta[None] + (types @ theta[:8].reshape(2, 4))[:, None]
    u = np.asarray(delta)[:, :, None, None] + np.einsum(
        "mjc,trc->mjtr", data["X"], beta)
    return np.where(data["mask"][:, :, None, None], u, -np.inf)


def welfare(data, name, report, draws):
    """Compensating variation from availability of gluten-free dishes."""
    theta = np.asarray(report["parameters"])
    model = DemandModel(data, name, normal_draws(draws, SEED))
    delta, _, err = model.invert(theta)
    if float(err) > 1e-8: raise RuntimeError("Share inversion failed")
    u = utilities(data, name, theta, delta, draws, SEED)
    gf = np.zeros(data["mask"].shape, bool)
    gf[data["mask"]] = data["df"].dish.str.endswith("_gf")
    restricted = np.where(gf[:, :, None, None], -np.inf, u)
    full = np.logaddexp(0, logsumexp(u, axis=1))
    without = np.logaddexp(0, logsumexp(restricted, axis=1))
    cv = ((full - without) / -report["price"]).mean(axis=2)
    avg = np.sum(cv * data["weights"], axis=1)
    if cv.min() < -1e-10: raise AssertionError("Negative welfare gain")
    return avg, cv, gf.sum(axis=1)


def q13_results(data, q12, output_dir, draws):
    years = data["df"].groupby("market", sort=True)["year"].first().to_numpy()
    universities = data["df"].groupby("market", sort=True)["University"].first().to_numpy()
    periods = {"2001-2010": years <= 2010,
               "2011-2015": (years >= 2011) & (years <= 2015),
               "2016-2020": years >= 2016, "2011-2020": years >= 2011}
    names = ["neither", "athlete_only", "graduate_only", "both"]
    out = {"counterfactual": "Remove all gluten-free dishes, hold other prices/characteristics fixed",
           "units": "price units per lunch choice", "aggregation": "equal weight for each campus-year market",
           "group_average": "each type equally averages over the same campus-year markets",
           "models": {}}
    market_rows = []
    for name in "abcd":
        avg, types, count = welfare(data, name, q12["models"][name], draws)
        out["models"][name] = {}
        for period, sel in periods.items():
            out["models"][name][period] = {
                "markets": int(sel.sum()), "average": float(avg[sel].mean()),
                "by_type": dict(zip(names, [float(types[sel, t].mean()) for t in range(4)]))}
        for m in range(data["M"]):
            market_rows.append({"model": name, "year": int(years[m]),
                "campus": universities[m], "gf_products": int(count[m]),
                "average": avg[m], **dict(zip(names, types[m]))})
        print(f"Q13 {name}: 2011-15 {avg[periods['2011-2015']].mean():.4f}, "
              f"2016-20 {avg[periods['2016-2020']].mean():.4f}", flush=True)
    (output_dir / "q13_results.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(market_rows).to_csv(output_dir / "q13_markets.csv", index=False)
    print(f"Q13 results written to {output_dir / 'q13_results.json'}")
    return out


def q15_probabilities(data, name, report, draws, seed, model=None):
    theta = np.asarray(report["parameters"])
    if model is None:
        model = DemandModel(data, name, normal_draws(draws, seed))
    delta, _, err = model.invert(jnp.asarray(theta))
    if float(err) > 1e-8: raise RuntimeError("Share inversion failed")
    u = utilities(data, name, theta, delta, draws, seed)
    shift = np.maximum(0, np.max(u, axis=1, keepdims=True))
    ex = np.exp(u-shift)
    p = ex / (np.exp(-shift)+ex.sum(axis=1, keepdims=True))
    w = np.broadcast_to(data["weights"][:, :, None]/p.shape[-1],
                        (data["M"], 4, p.shape[-1]))
    share = np.einsum("mjtr,mtr->mj", p, w)
    if not np.allclose(share[data["mask"]], data["S"][data["mask"]], atol=1e-8):
        raise AssertionError("Predicted and observed shares differ")
    return p, share, w


def q15_results(data, q12, output_dir, draws, seed):
    """Average product-level elasticity matrices and GF-to-GF price diversion."""
    df, M, J, mask = data["df"], data["M"], data["J"], data["mask"]
    years = df.groupby("market")["year"].first().to_numpy()
    periods = {"2001-2010": years <= 2010,
               "2011-2015": (years >= 2011) & (years <= 2015),
               "2016-2020": years >= 2016}
    labels = sorted(df.dish.unique())
    dish_id = np.full((M, J), -1)
    dish_id[mask] = pd.Categorical(df.dish, categories=labels).codes
    prices = np.zeros((M, J))
    prices[mask] = data["price"]
    gf = np.array([s.endswith("_gf") for s in labels])
    result = {"definition": "E[j,k] = (p_k/s_j) * ds_j/dp_k, averaged equally over markets",
              "ordering": labels, "matrices": {}, "summary": {}}
    rows = []
    for name in "abcd":
        alpha = q12["models"][name]["price"]
        p, shares, w = q15_probabilities(data, name, q12["models"][name], draws, seed)
        pair = np.einsum("mjtr,mktr,mtr->mjk", p, p, w)
        dsdp = alpha*(np.eye(J)[None]*shares[:, :, None]-pair)
        E = dsdp*prices[:, None, :]/np.where(shares > 0, shares, 1)[:, :, None]
        full = np.full((M, len(labels), len(labels)), np.nan)
        diverted = np.full(M, np.nan)
        for m in range(M):
            ix = np.flatnonzero(mask[m]); product_ix = dish_id[m, ix]
            full[m][np.ix_(product_ix, product_ix)] = E[m][np.ix_(ix, ix)]
            g = ix[gf[product_ix]]
            if len(g) > 1:
                diverted[m] = np.mean([dsdp[m, g[g != j], j].sum()/-dsdp[m, j, j] for j in g])
        if not (np.all(np.diagonal(E, axis1=1, axis2=2)[mask] < 0) and
                np.all(E[mask[:, :, None] & mask[:, None, :] & ~np.eye(J, dtype=bool)[None]] >= 0)):
            raise AssertionError(f"Wrong own/cross elasticity signs: {name}")
        result["matrices"][name], result["summary"][name] = {}, {}
        for period, sel in periods.items():
            avail = np.flatnonzero(np.any(np.isfinite(full[sel]), axis=(0, 2)))
            ids = [labels[i] for i in avail]
            mean = full[sel][:, avail][:, :, avail].mean(axis=0)
            if not np.isfinite(mean).all(): raise AssertionError("Missing elasticities")
            own = np.diag(mean)
            off = mean[~np.eye(len(avail), dtype=bool)]
            div = float(np.nanmean(diverted[sel])) if np.isfinite(diverted[sel]).any() else None
            gf_id = np.array([i for i, s in enumerate(ids) if s.endswith("_gf")])
            gf_cross = (float(mean[np.ix_(gf_id, gf_id)][~np.eye(len(gf_id), dtype=bool)].mean())
                        if len(gf_id) > 1 else None)
            result["matrices"][name][period] = {"dishes": ids, "average": mean.tolist(), "markets": int(sel.sum())}
            result["summary"][name][period] = {"mean_own": float(own.mean()),
                "mean_cross": float(off.mean()), "gf_gf_cross": gf_cross,
                "gf_to_other_gf_diversion": div}
            for i, dish_j in enumerate(ids):
                for k, dish_k in enumerate(ids):
                    rows.append((name, period, dish_j, dish_k, mean[i,k], int(sel.sum())))
        x = result["summary"][name]["2016-2020"]
        print(f"Q15 {name}: recent own {x['mean_own']:.3f}, cross {x['mean_cross']:.3f}, "
              f"GF diversion {x['gf_to_other_gf_diversion']:.3f}", flush=True)
    pd.DataFrame(rows, columns=["model", "period", "dish_j", "price_dish_k", "elasticity", "markets"]).to_csv(
        output_dir/"q15_matrices.csv", index=False)
    (output_dir/"q15_results.json").write_text(json.dumps(result, indent=2))
    print(f"Q15 results written to {output_dir/'q15_results.json'}", flush=True)
    return result


# Q16. Survey respondents are sampled among noodle buyers, not all students.
# Observed-choice moments condition on an inside purchase; hypothetical second
# choices condition on the recorded first dish, student type, and another dish.
SURVEY_CAMPUS = {"A&M": "Texas A&M", "Colorado C.": "Colorado",
                 "Rose-Hulman": "Rose-Hul.", "St. John's": "Saint John"}


def survey_data(data, directory):
    survey = pd.read_csv(directory/"survey_noodles.csv", sep="\t")
    survey["coll"] = survey["coll"].replace(SURVEY_CAMPUS)
    survey["Graduate"] = survey["Graduate"].replace({"No": "N"})
    for col in ["Athlete", "Graduate"]:
        if not survey[col].isin(["Y", "N"]).all():
            raise ValueError(f"Unknown survey response in {col}")
    if survey["id"].duplicated().any(): raise ValueError("Duplicate survey id")
    products = data["df"][["market", "University", "year", "dish", "chain"]].copy()
    products["slot"] = data["df"].groupby("market").cumcount()
    for tag, dish, chain in [("obs", "Pick_Obs", "Chain_Obs"),
                              ("hyp", "Pick_Hyp", "Chain_Hyp")]:
        cols = {"market": f"m_{tag}", "dish": dish,
                "chain": f"chain_{tag}", "slot": f"j_{tag}"}
        survey = survey.merge(products.rename(columns=cols),
            left_on=["coll", "year", dish],
            right_on=["University", "year", dish],
            how="left", validate="many_to_one").drop(columns="University")
        if survey[f"j_{tag}"].isna().any() or not (survey[chain] == survey[f"chain_{tag}"]).all():
            raise ValueError(f"Unmatched survey {tag} choice or inconsistent chain")
    if not (survey.m_obs == survey.m_hyp).all():
        raise ValueError("First and second choices have different markets")
    if (survey.j_obs == survey.j_hyp).any():
        raise ValueError("The unavailable first dish cannot be chosen again")
    ids = np.sort(survey.m_obs.unique().astype(int))
    mi = np.searchsorted(ids, survey.m_obs.to_numpy(int))
    j = survey.j_obs.to_numpy(int)
    k = survey.j_hyp.to_numpy(int)
    a = (survey.Athlete == "Y").to_numpy(int)
    g = (survey.Graduate == "Y").to_numpy(int)
    t = a + 2*g
    x = data["X"][ids]
    obs_x, second_x = x[mi,j], x[mi,k]
    f1 = np.column_stack([a, g, a*g, a[:,None]*obs_x, g[:,None]*obs_x])
    chains = np.full((data["M"], data["J"]), -1)
    chains[data["mask"]] = pd.Categorical(data["df"].chain,
        categories=sorted(data["df"].chain.unique())).codes
    gluten = np.zeros_like(data["mask"])
    gluten[data["mask"]] = data["df"].dish.str.endswith("_gf")
    f2 = np.column_stack([(obs_x[:,:,None]*second_x[:,None,:]).reshape(len(j),16),
        (chains[ids[mi],j] == chains[ids[mi],k]).astype(float),
        (gluten[ids[mi],j] & gluten[ids[mi],k]).astype(float)])
    counts = np.zeros((len(ids), data["J"], 4))
    np.add.at(counts, (mi,j,t), 1)
    market_n = np.bincount(mi, minlength=len(ids))
    print(f"Q16 survey: {len(survey)} students, {len(ids)} markets; choices matched", flush=True)
    return dict(ids=ids, m=mi, j=j, t=t, n=len(j), counts=counts,
                market_n=market_n, x=x, gluten=gluten[ids],
                chain=chains[ids], f1=f1, f2=f2,
                observed=np.r_[f1.mean(axis=0), f2.mean(axis=0)])


class MicroMoments:
    def __init__(self, data, survey, name, draws, seed):
        self.s, self.name, self.d = survey, name, data
        self.num = {"a":4, "b":8, "c":12, "d":15}[name]
        self.scale = survey["n"]/data["n"]
        self.micro_weight = {}
        self._compile(draws, seed)

    def _compile(self, draws, seed):
        s, name = self.s, self.name
        x = jnp.asarray(s["x"])
        mask = jnp.asarray(self.d["mask"][s["ids"]])
        group_w = jnp.asarray(self.d["weights"][s["ids"]])
        count = jnp.asarray(s["counts"]/s["n"])
        market_weight = jnp.asarray(s["market_n"]/s["n"])
        gluten = jnp.asarray(s["gluten"], dtype=float)
        same_chain = jnp.asarray(s["chain"][:,:,None] == s["chain"][:,None,:], dtype=float)
        nu = jnp.asarray(normal_draws(draws if name != "b" else 2, seed)
                         if name != "b" else np.zeros((1,4)))
        flags = jnp.array([[0.,0.], [1.,0.], [0.,1.], [1.,1.]])
        a, g = flags[:,0], flags[:,1]

        def prediction(theta, delta):
            if name in ("a", "c"):
                L = jnp.diag(theta[-4:])
            elif name == "b":
                L = jnp.zeros((4,4))
            else:
                z = theta[-7:]
                L = jnp.array([[z[0],0.,0.,0.], [0.,z[1],0.,0.],
                               [0.,z[2],z[3],0.], [0.,z[4],z[5],z[6]]])
            rand = nu@L.T
            means = jnp.zeros((4,4)) if name == "a" else flags@theta[:8].reshape(2,4)
            beta = means[:,None,:] + rand[None,:,:]
            utility = delta[:,:,None,None] + jnp.einsum("mjc,trc->mjtr", x, beta)
            utility = jnp.where(mask[:,:,None,None], utility, -jnp.inf)
            shift = jnp.maximum(0.,jnp.max(utility,axis=1,keepdims=True))
            ex = jnp.exp(utility-shift)
            total = ex.sum(axis=1)
            first_p = ex/(jnp.exp(-shift)+total[:,None])
            mass = first_p.mean(axis=-1)*group_w[:,None,:]
            inside = mass.sum(axis=(1,2))
            f1 = jnp.column_stack([
                jnp.einsum("mjt,t->m", mass, a)/inside,
                jnp.einsum("mjt,t->m", mass, g)/inside,
                jnp.einsum("mjt,t->m", mass, a*g)/inside,
                jnp.einsum("mjt,t,mjc->mc", mass, a, x)/inside[:,None],
                jnp.einsum("mjt,t,mjc->mc", mass, g, x)/inside[:,None]])
            # Conditional ranking among the other noodle dishes, excluding outside.
            remaining = jnp.where(mask[:,:,None,None],
                jnp.maximum(total[:,None]-ex, 1e-12), 1.)
            ex_x = jnp.einsum("mjtr,mjc->mtrc", ex, x)
            second_x = (ex_x[:,None] - ex[:,:,:,:,None]*x[:,:,None,None,:])/remaining[:,:,:,:,None]
            posterior = first_p/jnp.where(mask[:,:,None,None],
                jnp.maximum(first_p.sum(axis=-1,keepdims=True),1e-12), 1.)
            mean_second = jnp.sum(posterior[:,:,:,:,None]*second_x, axis=-2)
            same_total = jnp.einsum("mktr,mjk->mjtr", ex, same_chain)
            same = jnp.sum(posterior*(same_total-ex)/remaining, axis=-1)
            gf_total = jnp.einsum("mjtr,mj->mtr", ex, gluten)
            gf2 = jnp.sum(posterior*(gf_total[:,None]-gluten[:,:,None,None]*ex)/remaining,axis=-1)*gluten[:,:,None]
            outer = x[:,:,None,:,None]*mean_second[:,:,:,None,:]
            f2 = jnp.concatenate([outer.reshape(*count.shape,16),
                                  same[:,:,:,None], gf2[:,:,:,None]], axis=-1)
            return f1, f2

        def moments(theta, delta):
            f1, f2 = prediction(theta, delta)
            return jnp.concatenate([market_weight@f1,
                                    jnp.einsum("mjt,mjtq->q",count,f2)])

        @jax.jit
        def gradient(theta, delta, ddelta):
            moment = moments(theta, delta)
            vectors = jnp.eye(theta.size)
            dx = jnp.moveaxis(ddelta,-1,0)
            tangents = jax.vmap(lambda u, v: jax.jvp(moments,(theta,delta),(u,v))[1])(vectors,dx)
            return moment, tangents.T

        self._predict = jax.jit(prediction)
        self._grad = gradient

    def current(self, theta, fit):
        m = self.s["ids"]
        moment, derivative = self._grad(jnp.asarray(theta),
            jnp.asarray(fit["delta_full"][m]),
            jnp.asarray(fit["ddelta_full"][m]))
        return np.asarray(moment), np.asarray(derivative)

    def update_weight(self, theta, fit, first_only):
        s = self.s
        f1, f2 = self._predict(jnp.asarray(theta),jnp.asarray(fit["delta_full"][s["ids"]]))
        f1, f2 = np.asarray(f1), np.asarray(f2)
        residual = np.column_stack([s["f1"]-f1[s["m"]],
            s["f2"]-f2[s["m"],s["j"],s["t"]]])
        if first_only: residual = residual[:,:11]
        scores = np.zeros((len(s["ids"]),residual.shape[1]))
        np.add.at(scores,s["m"],residual)
        omega = scores.T@scores/s["n"]
        diag = np.diag(np.diag(omega))
        omega = .9*omega + .1*diag
        omega += .01*np.trace(omega)/len(omega)*np.eye(len(omega))
        weight = np.linalg.inv(omega)
        self.micro_weight[first_only] = weight

    def eval_joint(self, theta, model, first_only):
        try:
            fit = model.evaluate(theta,details=True)
            predicted, derivative = self.current(theta,fit)
            size = 11 if first_only else len(predicted)
            gap = self.s["observed"][:size] - predicted[:size]
            W = self.micro_weight[first_only]
            obj = float(fit["obj"] + self.scale*gap@W@gap)
            grad = fit["grad"] - 2*self.scale*derivative[:size].T@W@gap
            if np.isfinite(obj) and np.isfinite(grad).all(): return obj,grad
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            pass
        # Invalid trial points are rejected during the optimizer's line search.
        return 1e4 + float(theta@theta), 2*theta


def elasticity_brief(data, name, report, model, draws, seed):
    """Mean recent price elasticities and gluten-free diversion."""
    p, share, w = q15_probabilities(data, name, report, draws, seed, model)
    pair = np.einsum("mjtr,mktr,mtr->mjk", p, p, w)
    alpha = report["price"]
    dsdp = alpha * (np.eye(data["J"])[None]*share[:, :, None]-pair)
    price = np.zeros_like(share)
    price[data["mask"]] = data["price"]
    e = dsdp * price[:, None, :] / np.maximum(share[:, :, None], 1e-200)
    gf = np.zeros_like(data["mask"])
    gf[data["mask"]] = data["df"].dish.str.endswith("_gf").to_numpy()
    years = data["df"].groupby("market")["year"].first().to_numpy()
    own, cross, div = [], [], []
    for m in np.flatnonzero(years >= 2016):
        ix = np.flatnonzero(data["mask"][m])
        E = e[m][np.ix_(ix, ix)]
        own.extend(np.diag(E))
        cross.extend(E[~np.eye(len(ix), dtype=bool)])
        g = np.flatnonzero(gf[m])
        div.extend([dsdp[m, g[g != j], j].sum() / -dsdp[m, j, j]
                    for j in g])
    return dict(own=float(np.mean(own)), cross=float(np.mean(cross)),
                gf_diversion=float(np.mean(div)))


def q16_results(data, q12, q13, output_dir, directory, draws, seed, maxiter):
    survey = survey_data(data,directory)
    allresults = {"survey_n":survey["n"],"survey_markets":len(survey["ids"]),
                  "first_moments":11,"second_moments":18,
                  "method":"Joint macro and conditional micro GMM, 2-step block weighting",
                  "selection":"Survey conditional on noodle purchases; second choice conditional on first dish and type, restricted to other noodle dishes",
                  "models":{}}
    results_rows=[]
    moments_rows=[]
    years=data["df"].groupby("market")["year"].first().to_numpy()
    recent = years>=2016
    available = years>=2011
    for name in "abcd":
        print(f"Q16 fitting model ({name})",flush=True)
        model = DemandModel(data,name,normal_draws(draws,seed))
        model.second_step_weight(np.asarray(q12["models"][name]["parameters"]))
        micro = MicroMoments(data,survey,name,draws,seed)
        theta = np.asarray(q12["models"][name]["parameters"])
        _, bounds = initial_values(name,{},seed,1)
        report = q12["models"][name]
        stages={"aggregate":report}
        for first_only, stage in [(True,"demographics"),(False,"demographics_second")]:
            micro.update_weight(theta,model.evaluate(theta,details=True),first_only)
            best=None
            for step in range(2):
                res=minimize(lambda x:micro.eval_joint(x,model,first_only),theta,
                    method="L-BFGS-B",jac=True,bounds=bounds,
                    options={"maxiter":maxiter,"gtol":3e-6,"ftol":1e-11,"maxls":25})
                if not np.isfinite(res.fun): raise RuntimeError(f"Q16 {name} bad objective")
                theta=res.x
                print(f"Q16 {name}, {stage}, step {step+1}: obj={res.fun:.7g}; {res.message}",flush=True)
                if step==0:
                    model.second_step_weight(theta)
                    micro.update_weight(theta,model.evaluate(theta,details=True),first_only)
                best=res
            if not best.success:
                raise RuntimeError(f"Q16 {name}, {stage}: optimizer did not converge")
            fit=model.evaluate(theta,details=True)
            report=summarize(model,theta,fit)
            report.update(parameters=theta.tolist(),joint_objective=float(best.fun),
                          optimizer_success=bool(best.success),gradient_norm=float(np.linalg.norm(best.jac)))
            if fit["max_share_error"]>1e-8 or report["price"]>=0:
                raise RuntimeError(f"Q16 {name}, {stage}: inversion or price invalid")
            stages[stage]=report
        summaries={}
        for stage,report in stages.items():
            cv, types, _=welfare(data,name,report,draws)
            e=elasticity_brief(data,name,report,model,draws,seed)
            summaries[stage] = {"price":report["price"],
                "mean_welfare_2011_2020":float(cv[available].mean()),
                "mean_welfare_2016_2020":float(cv[recent].mean()),
                "athlete_only_welfare":float(types[available,1].mean()),
                "graduate_only_welfare":float(types[available,2].mean()),
                "mean_own_elasticity_2016_2020":e["own"],
                "mean_cross_elasticity_2016_2020":e["cross"],
                "gf_to_gf_diversion_2016_2020":e["gf_diversion"]}
            results_rows.append(dict(model=name,stage=stage,**summaries[stage]))
        allresults["models"][name]={"stages":stages,"comparison":summaries}
        for stage in ["aggregate","demographics","demographics_second"]:
            theta=np.asarray(stages[stage]["parameters"])
            fit=model.evaluate(theta,details=True)
            pred,_=micro.current(theta,fit)
            for i,(obs,pr) in enumerate(zip(survey["observed"],pred)):
                moments_rows.append(dict(model=name,stage=stage,kind="first" if i<11 else "second",
                                         moment=i,observed=float(obs),predicted=float(pr)))
        print(f"Q16 ({name}) welfare: "+" / ".join(f"{s}={summaries[s]['mean_welfare_2011_2020']:.4f}" for s in summaries),flush=True)
        (output_dir/"q16_results.json").write_text(json.dumps(allresults,indent=2))
    pd.DataFrame(results_rows).to_csv(output_dir/"q16_comparison.csv",index=False)
    pd.DataFrame(moments_rows).to_csv(output_dir/"q16_moments.csv",index=False)
    print(f"Q16 results written to {output_dir/'q16_results.json'}",flush=True)
    return allresults

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    args.draws, args.seed, args.starts, args.maxiter = 512, 2026, 2, 160
    global SEED
    SEED = args.seed
    directory = input_folder(args.data_dir)
    output_dir = Path(args.out_dir) if args.out_dir else directory.parent.parent / "results/q12"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare(load_data(directory))
    q12_path = output_dir / "q12_results.json"
    saved = json.loads(q12_path.read_text()) if q12_path.exists() else None
    valid = (saved is not None and
             saved["meta"]["draws"] == args.draws and
             saved["meta"]["seed"] == args.seed and
             saved["meta"]["steps"] == 2 and
             saved["meta"]["products"] == data["n"] and
             saved["meta"]["markets"] == data["M"] and
             saved["meta"]["instruments"] == data["Z"].shape[1] and
             all(k in saved["models"] and saved["models"][k].get("optimizer_success")
                 for k in "abcd"))
    if valid:
        print("Reusing existing Q12 estimates", flush=True)
        q12 = saved
    else:
        q12 = {"meta": {"products": data["n"], "markets": data["M"],
               "instruments": data["Z"].shape[1], "draws": args.draws,
               "seed": args.seed, "steps": 2,
               "assumption": "Athlete and graduate status independent conditional on campus-year"},
               "models": {}}
        previous = {}
        for name in "abcd":
            model = DemandModel(data, name, normal_draws(args.draws, args.seed))
            starts, bounds = initial_values(name, previous, args.seed, args.starts)
            best = None
            for i, guess in enumerate(starts, 1):
                res = minimize(model.evaluate, guess, method="L-BFGS-B", jac=True,
                    bounds=bounds, options={"maxiter": args.maxiter, "ftol": 1e-12,
                                             "gtol": 2e-6, "maxls": 20})
                print(f"Q12 {name} start {i}: {res.fun:.9g}", flush=True)
                if best is None or res.fun < best.fun: best = res
            model.second_step_weight(best.x)
            best = minimize(model.evaluate, best.x, method="L-BFGS-B", jac=True,
                bounds=bounds, options={"maxiter": args.maxiter, "ftol": 1e-12,
                                         "gtol": 2e-6, "maxls": 20})
            fit = model.evaluate(best.x, details=True)
            if not best.success or fit["max_share_error"] > 1e-8:
                raise RuntimeError(f"Q12 {name} failed to converge: {best.message}")
            report = summarize(model, best.x, fit)
            report.update(parameters=best.x.tolist(), optimizer_success=bool(best.success),
                          optimizer_message=str(best.message), iterations=int(best.nit),
                          gradient_norm=float(np.linalg.norm(fit["grad"])))
            q12["models"][name] = report
            previous[name] = best.x.copy()
            q12_path.write_text(json.dumps(q12, indent=2))
            print(f"Q12 {name}: price {report['price']:.4f}", flush=True)
    q13 = q13_results(data, q12, output_dir, args.draws)
    q15_results(data, q12, output_dir, args.draws, args.seed)
    q16_results(data, q12, q13, output_dir, directory, args.draws, args.seed, args.maxiter)


if __name__ == "__main__":
    main()
