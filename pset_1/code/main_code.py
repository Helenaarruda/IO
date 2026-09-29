import jax
import jax.numpy as jnp
import pandas as pd
from scipy.optimize import root
from scipy.optimize import minimize

# ==============================================================================
# Functions from Pietro's code
# ==============================================================================


def single_t_likelihood(model_params, data):
    """Calculates the log-likelihood for a single problem instance."""
    Y,X = data
    # Unpack model parameters
    param1, param2 = model_params['param1'], model_params['param2']
    # Example computation (replace with actual model logic)
    f = jnp.dot(X, param1) + param2 - Y
    # Log-likelihood calculation
    total_log_likelihood = jnp.log(f)
    
    return total_log_likelihood


def likelihood(model_params, data):
    """Calculates the log-likelihood for EACH problem in the batch and returns them as an array."""
    # Vmap the single_likelihood function
    all_likelihoods = jax.vmap(
        single_t_likelihood,
        in_axes=(None, 0)
    )(model_params, data)
    
    return jnp.sum(all_likelihoods)
# ==============================================================================
# Example usage
# ==============================================================================
#if __name__ == "__main__":
#    # Example model parameters
#    model_params = {
#        'param1': jnp.array([0.5, 1.0]),
#        'param2': 2.0
#    }
    
    # Example data: batch of 3 instances
#    data = (
#        jnp.array([1.0, 2.0, 3.0]),  # Y values
#        jnp.array([[1.0, 0.5],       # X values for t= 1
#                   [2.0, 1.5],       # X values for t= 2
#                   [3.0, 2.5]])      # X values for t= 3
#    )
    
    # Calculate likelihood
#    log_likelihood = likelihood(model_params, data)
#    print("Log-Likelihood:", log_likelihood)




# ==============================================================================
# Question 1 - Part ii
# ==============================================================================



# ============================================================
# PARAMETERS
# ============================================================

alpha = 0.5
gamma = 0.8
mu = 0.2
rho = 0.1
beta = 0.6

T = 1000


# ============================================================
# FUNCTION TO SOLVE ONE MARKET
# ============================================================

def solve_market(p, u, nu, eta, scenario,
                 alpha, gamma, mu, rho, beta):

    # --------------------------------------------------------
    # Labor supply:
    #
    # L = exp(
    #     alpha
    #     + gamma*log(w)
    #     + mu*log(u)
    #     + rho*log(u)*log(w)
    #     + eta
    # )
    #
    # Therefore:
    #
    # log(L) =
    #     alpha
    #     + mu*log(u)
    #     + eta
    #     + [gamma + rho*log(u)]*log(w)
    # --------------------------------------------------------

    # We solve for log(w).

    # Scenario 1: perfect competition
    #
    # p * beta * exp(nu) * L^(beta-1) = w

    # Taking logs:
    #
    # log(p) + log(beta) + nu
    # + (beta-1)*log(L)
    # = log(w)

    # Substitute the labor supply equation for log(L):
    #
    # log(L) =
    # alpha + mu*log(u) + eta
    # + [gamma + rho*log(u)]*log(w)

    # This gives an equation in log(w).

    if scenario == 1: 

        numerator = (
            jnp.log(p)
            + jnp.log(beta)
            + nu
            + (beta - 1)
            * (
                alpha
                + mu * jnp.log(u)
                + eta
            )
        )

        denominator = (
            1
            - (beta - 1)
            * (
                gamma
                + rho * jnp.log(u)
            )
        )

        log_w = numerator / denominator


    # --------------------------------------------------------
    # Scenario 2: monopsonist
    #
    # R'(L) =
    # w + L*dw/dL
    #
    # and
    #
    # L*dw/dL =
    # w / [gamma + rho*log(u)]
    #
    # Therefore:
    #
    # R'(L) =
    # w * [
    #     1 + 1/(gamma + rho*log(u))
    # ]
    # --------------------------------------------------------

    elif scenario == 2:

        numerator = (
            jnp.log(p)
            + jnp.log(beta)
            + nu
            + (beta - 1)
            * (
                alpha
                + mu * jnp.log(u)
                + eta
            )
            - jnp.log(
                1
                + 1 / (
                    gamma
                    + rho * jnp.log(u)
                )
            )
        )

        denominator = (
            1
            - (beta - 1)
            * (
                gamma
                + rho * jnp.log(u)
            )
        )

        log_w = numerator / denominator


    else:
        raise ValueError(
            "scenario must be 1 or 2"
        )


    # --------------------------------------------------------
    # Recover wage
    # --------------------------------------------------------

    w = jnp.exp(log_w)


    # --------------------------------------------------------
    # Recover employment using labor supply
    # --------------------------------------------------------

    log_L = (
        alpha
        + gamma * jnp.log(w)
        + mu * jnp.log(u)
        + rho * jnp.log(u) * jnp.log(w)
        + eta
    )

    L = jnp.exp(log_L)


    return w, L


# ============================================================
# GENERATE SHOCKS
# ============================================================

key = jax.random.PRNGKey(12345)

key_u, key_p, key_eta, key_nu, key_scenario = \
    jax.random.split(key, 5)


# u_t ~ LogNormal(0, 0.5)

u = jnp.exp(
    0.5 * jax.random.normal(
        key_u,
        shape=(T,)
    )
)


# p_t ~ LogNormal(0, 0.2)

p = jnp.exp(
    0.2 * jax.random.normal(
        key_p,
        shape=(T,)
    )
)


# eta_t ~ Normal(0, 0.1)

eta = (
    0.1
    * jax.random.normal(
        key_eta,
        shape=(T,)
    )
)


# nu_t ~ Normal(0, 0.2)

nu = (
    0.2
    * jax.random.normal(
        key_nu,
        shape=(T,)
    )
)


# ============================================================
#  Solve using vmap! So no need to apply for and repeat solve 
#  markets funct
# ============================================================


solve_markets = jax.vmap(
    solve_market,
    in_axes=(
        0,      # p
        0,      # u
        0,      # nu
        0,      # eta
        None,   # scenario
        None,   # alpha
        None,   # gamma
        None,   # mu
        None,   # rho
        None    # beta
    )
)



# ============================================================
# SCENARIO 1
# ============================================================

w1, L1 = solve_markets(
    p,
    u,
    nu,
    eta,
    1,
    alpha,
    gamma,
    mu,
    rho,
    beta
)

# ============================================================
# SCENARIO 2
# ============================================================

w2, L2 = solve_markets(
    p,
    u,
    nu,
    eta,
    2,
    alpha,
    gamma,
    mu,
    rho,
    beta
)


# ============================================================
# SCENARIO 3
# ============================================================

scenario_draw = jax.random.uniform(
    key_scenario,
    shape=(T,)
)

scenario_1_market = scenario_draw < 0.5

w3 = jnp.where(
    scenario_1_market,
    w1,
    w2
)

L3 = jnp.where(
    scenario_1_market,
    L1,
    L2
)


# ============================================================
# CREATE DATAFRAMES
# ============================================================

t = jnp.arange(1, T + 1)


df1 = pd.DataFrame({
    "t": t,
    "w": w1,
    "L": L1,
    "u": u,
    "p": p
})


df2 = pd.DataFrame({
    "t": t,
    "w": w2,
    "L": L2,
    "u": u,
    "p": p
})


df3 = pd.DataFrame({
    "t": t,
    "w": w3,
    "L": L3,
    "u": u,
    "p": p
})


# ============================================================
# SAVE
# ============================================================

df1.to_csv("temp/scenario_1.csv", index=False)
df2.to_csv("temp/scenario_2.csv", index=False)
df3.to_csv("temp/scenario_3.csv", index=False)

print("Datasets generated successfully!")


# ============================================================
# Part 2: Estimation
# ============================================================


def single_t_moments(theta, data):
    """
    Calculates the moment conditions for one market t.
    """

    # Unpack parameters directly from theta
    alpha = theta[0]
    gamma = theta[1]
    mu = theta[2]
    rho = theta[3]
    beta = theta[4]

    # Data for market t
    w, L, u, p = data

    # Logs
    log_w = jnp.log(w)
    log_L = jnp.log(L)
    log_u = jnp.log(u)
    log_p = jnp.log(p)

    # Labor supply residual
    eta = (
        log_L
        - alpha
        - gamma * log_w
        - mu * log_u
        - rho * log_u * log_w
    )

    # Firm FOC residual under monopsony
    nu = (
        log_w
        + jnp.log(1 + 1 / (gamma + rho * log_u))
        - log_p
        - jnp.log(beta)
        - (beta - 1) * log_L
    )

    # Instruments
    Z = jnp.array([
        1.0,
        log_u,
        log_p
    ])

    # Six moments for market t
    moments_eta = Z * eta
    moments_nu = Z * nu

    return moments_eta, moments_nu


    # -------------------------------------------------
    # 5. Apply vmap
    # -------------------------------------------------


def all_t_moments(theta, data):
    """
    Calculates the moment conditions for all markets.
    """
    all_moments_eta, all_moments_nu = jax.vmap(
        single_t_moments,
        in_axes=(None, 0)
    )(theta, data)

    return all_moments_eta, all_moments_nu

    # -------------------------------------------------
    # 6. Compute Sample Moments
    # -------------------------------------------------

def sample_moments(theta, data):

    moments_eta, moments_nu = all_t_moments(
        theta, data
    )

    mean_moments_eta = jnp.mean(
        moments_eta,
        axis=0
    )

    mean_moments_nu = jnp.mean(
        moments_nu,
        axis=0
    )

    return jnp.concatenate([
        mean_moments_eta,
        mean_moments_nu
    ])

    # -------------------------------------------------
    # 7. Define GMM function to minimize
    # -------------------------------------------------

def gmm_objective(theta, data):

    moments = sample_moments(
        theta,
        data
    )

    return jnp.sum(moments ** 2)



    # -------------------------------------------------
    # 8. Choose starting value of the parameters
    # -------------------------------------------------

theta_start = jnp.array([
    1.0,   # alpha
    1.0,   # gamma
    1.0,   # mu
    0.0,   # rho
    0.5    # beta
])

gmm_gradient = jax.jit(
    jax.grad(gmm_objective)
)




    # -------------------------------------------------
    # 9. Run for scenario 1
    # -------------------------------------------------


data1 = (
    jnp.array(df1["w"].values),
    jnp.array(df1["L"].values),
    jnp.array(df1["u"].values),
    jnp.array(df1["p"].values)
)


result1 = minimize(
    fun=lambda theta: float(
        gmm_objective(theta, data1)
    ),
    x0=theta_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient(theta, data1)
    ).astype(float),
    method="BFGS"
)

theta_hat_1 = result1.x


# Print results
print("Estimated parameters:")
print("alpha =", theta_hat_1[0])
print("gamma =", theta_hat_1[1])
print("mu    =", theta_hat_1[2])
print("rho   =", theta_hat_1[3])
print("beta  =", theta_hat_1[4])

print("\nNumber of function evaluations:", result1.nfev)
print("Number of gradient evaluations:", result1.njev)
print("Converged:", result1.success)



print("\nGMM objective:")
print(float(gmm_objective(theta_hat_1, data1)))

print("\nEstimated sample moments:")
print(sample_moments(theta_hat_1, data1))


theta_true = jnp.array([
    0.5,
    0.8,
    0.2,
    0.1,
    0.6
])


# Compare with true parameter values]
print("\nGMM objective at true theta:")
print(float(gmm_objective(theta_true, data1)))

print("\nSample moments at true theta:")
print(sample_moments(theta_true, data1))



    # -------------------------------------------------
    # 10. Run for scenario 2 - The true monopsony!!!
    # -------------------------------------------------


data2 = (
    jnp.array(df2["w"].values),
    jnp.array(df2["L"].values),
    jnp.array(df2["u"].values),
    jnp.array(df2["p"].values)
)


result2 = minimize(
    fun=lambda theta: float(
        gmm_objective(theta, data2)
    ),
    x0=theta_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient(theta, data2)
    ).astype(float),
    method="BFGS"
)

theta_hat_2 = result2.x

# Print results
print("Estimated parameters:")
print("alpha =", theta_hat_2[0])
print("gamma =", theta_hat_2[1])
print("mu    =", theta_hat_2[2])
print("rho   =", theta_hat_2[3])
print("beta  =", theta_hat_2[4])

print("\nNumber of function evaluations:", result2.nfev)
print("Number of gradient evaluations:", result2.njev)
print("Converged:", result2.success)


# Compare with true parameter values]
print("\nGMM objective:")
print(float(gmm_objective(theta_hat_2, data2)))

print("\nEstimated sample moments:")
print(sample_moments(theta_hat_2, data2))


theta_true = jnp.array([
    0.5,
    0.8,
    0.2,
    0.1,
    0.6
])

print("\nGMM objective at true theta:")
print(float(gmm_objective(theta_true, data2)))

print("\nSample moments at true theta:")
print(sample_moments(theta_true, data2))




    # -------------------------------------------------
    # 11. Run for scenario 3
    # -------------------------------------------------


data3 = (
    jnp.array(df3["w"].values),
    jnp.array(df3["L"].values),
    jnp.array(df3["u"].values),
    jnp.array(df3["p"].values)
)


result3 = minimize(
    fun=lambda theta: float(
        gmm_objective(theta, data3)
    ),
    x0=theta_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient(theta, data3)
    ).astype(float),
    method="BFGS"
)

theta_hat_3 = result3.x

print("Estimated parameters:")
print("alpha =", theta_hat_3[0])
print("gamma =", theta_hat_3[1])
print("mu    =", theta_hat_3[2])
print("rho   =", theta_hat_3[3])
print("beta  =", theta_hat_3[4])

print("\nNumber of function evaluations:", result3.nfev)
print("Number of gradient evaluations:", result3.njev)
print("Converged:", result3.success)



print("\nGMM objective:")
print(float(gmm_objective(theta_hat_3, data3)))

print("\nEstimated sample moments:")
print(sample_moments(theta_hat_3, data3))


theta_true = jnp.array([
    0.5,
    0.8,
    0.2,
    0.1,
    0.6
])

print("\nGMM objective at true theta:")
print(float(gmm_objective(theta_true, data3)))

print("\nSample moments at true theta:")
print(sample_moments(theta_true, data3))