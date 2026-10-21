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


def welfare(data, name, report, draws, seed):
    """Compensating variation from availability of gluten-free dishes."""
    theta = np.asarray(report["parameters"])
    model = DemandModel(data, name, normal_draws(draws, seed))
    delta, _, err = model.invert(theta)
    if float(err) > 1e-8: raise RuntimeError("Share inversion failed")
    u = utilities(data, name, theta, delta, draws, seed)
    gf = np.zeros(data["mask"].shape, bool)
    gf[data["mask"]] = data["df"].dish.str.endswith("_gf")
    restricted = np.where(gf[:, :, None, None], -np.inf, u)
    full = np.logaddexp(0, logsumexp(u, axis=1))
    without = np.logaddexp(0, logsumexp(restricted, axis=1))
    cv = ((full - without) / -report["price"]).mean(axis=2)
    avg = np.sum(cv * data["weights"], axis=1)
    if cv.min() < -1e-10: raise AssertionError("Negative welfare gain")
    return avg, cv, gf.sum(axis=1)


def q13_results(data, q12, output_dir, draws, seed):
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
        avg, types, count = welfare(data, name, q12["models"][name], draws, seed)
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


def q14_results(q12, q13, output_dir):
    """Diagnostics for the required normal-vs-finite-types discussion (no optional fit)."""
    cov = np.asarray(q12["models"]["d"]["covariance"])
    corr = q12["models"]["d"]["correlation"]
    result = {
        "d_correlations": {"wheat_veg": corr[1][2],
                           "wheat_calories": corr[1][3],
                           "veg_calories": corr[2][3]},
        "d_covariance_eigenvalues_wheat_veg_calories": np.linalg.eigvalsh(cov[1:, 1:]).tolist(),
        "normal_coefficient_sd": {k: q12["models"][k]["sd"] for k in "acd"},
        "welfare_2011_2020": {k: q13["models"][k]["2011-2020"] for k in "abcd"}
    }
    (output_dir / "q14_results.json").write_text(json.dumps(result, indent=2))
    print(f"Q14 diagnostics written to {output_dir / 'q14_results.json'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    draws, seed, starts, maxiter = 512, 2026, 2, 160
    directory = input_folder(args.data_dir)
    output_dir = Path(args.out_dir) if args.out_dir else directory.parent.parent / "results/q12"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare(load_data(directory))
    q12_path = output_dir / "q12_results.json"
    saved = json.loads(q12_path.read_text()) if q12_path.exists() else None
    valid = (saved is not None and
             saved["meta"]["draws"] == draws and saved["meta"]["seed"] == seed and
             saved["meta"]["steps"] == 2 and saved["meta"]["products"] == data["n"] and
             saved["meta"]["markets"] == data["M"] and
             saved["meta"]["instruments"] == data["Z"].shape[1] and
             all(k in saved["models"] and saved["models"][k].get("optimizer_success")
                 for k in "abcd"))
    if valid:
        print("Reusing existing Q12 estimates", flush=True)
        q12 = saved
    else:
        q12 = {"meta": {"products": data["n"], "markets": data["M"],
                "instruments": data["Z"].shape[1], "draws": draws, "seed": seed,
                "steps": 2,
                "assumption": "Athlete and graduate status independent conditional on campus-year"},
                "models": {}}
        previous = {}
        for name in "abcd":
            model = DemandModel(data, name, normal_draws(draws, seed))
            guesses, bounds = initial_values(name, previous, seed, starts)
            best = None
            for i, guess in enumerate(guesses, 1):
                res = minimize(model.evaluate, guess, method="L-BFGS-B", jac=True,
                    bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-12,
                                             "gtol": 2e-6, "maxls": 20})
                print(f"Q12 {name} start {i}: {res.fun:.9g}", flush=True)
                if best is None or res.fun < best.fun: best = res
            model.second_step_weight(best.x)
            best = minimize(model.evaluate, best.x, method="L-BFGS-B", jac=True,
                bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-12,
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
    q13 = q13_results(data, q12, output_dir, draws, seed)
    q14_results(q12, q13, output_dir)


if __name__ == "__main__":
    main()

