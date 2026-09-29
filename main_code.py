import jax
import jax.numpy as jnp
import pandas as pd
from scipy.optimize import root

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
    "wt": w1,
    "Lt": L1,
    "ut": u,
    "pt": p
})


df2 = pd.DataFrame({
    "t": t,
    "wt": w2,
    "Lt": L2,
    "ut": u,
    "pt": p
})


df3 = pd.DataFrame({
    "t": t,
    "wt": w3,
    "Lt": L3,
    "ut": u,
    "pt": p
})


# ============================================================
# SAVE
# ============================================================

df1.to_csv("scenario_1.csv", index=False)
df2.to_csv("scenario_2.csv", index=False)
df3.to_csv("scenario_3.csv", index=False)

print("Datasets generated successfully!")