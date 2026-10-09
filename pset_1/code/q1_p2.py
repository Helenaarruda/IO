import jax
import jax.numpy as jnp
import pandas as pd
from scipy.optimize import root
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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
# QUESTION 1 - Part 1, item ii
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

    # --------------------------------------------------------
    # Scenario 1: Perfect Competition
    # --------------------------------------------------------
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
    # Scenario 2: Monopsonist
    # --------------------------------------------------------
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
# Add shocks
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


#  Solve using vmap! So no need to apply for and repeat solve 
#  markets funct

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
# Solve and Export
# ============================================================


# SCENARIO 1
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


# SCENARIO 2
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



# SCENARIO 3
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


# CREATE DATAFRAMES

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


# SAVE
df1.to_csv("temp/scenario_1.csv", index=False)
df2.to_csv("temp/scenario_2.csv", index=False)
df3.to_csv("temp/scenario_3.csv", index=False)

print("Datasets generated successfully!")

# ============================================================
# Part 1 item iii: Graphs
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df1["ut"], df1["wt"],
    alpha=0.5,
    label="Scenario 1"
)

plt.scatter(
    df2["ut"], df2["wt"],
    alpha=0.5,
    label="Scenario 2"
)

plt.scatter(
    df3["ut"], df3["wt"],
    alpha=0.5,
    label="Scenario 3"
)

plt.xlabel("Unemployment ($u_t$)")
plt.ylabel("Wage ($w_t$)")
plt.title("Relationship between wages and unemployment")
plt.legend()
plt.grid(alpha=0.2)

# Save figure
plt.savefig(
    "output/wage_unemployment_scenarios.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# The graph shows that wages are generally higher in Scenario 1 than in Scenario 2. This is consistent with the theoretical 
# results from Part 1(i): under wage-taking, firms take the wage as given and pay the competitive wage, whereas under 
# monopsony firms internalize the effect of their labor demand on wages and therefore choose a lower wage. Scenario 3 lies 
# between the two cases, as it combines wage-taking and monopsonistic markets with equal probability, which is consistent 
# with the theoretical predictions. 
# The relationship between unemployment and wages is less clear in the raw data. In theory, holding the other shocks constant,
# higher unemployment should lead to lower wages because it shifts the labor supply curve outward. Still, the relationship appears more clearly at the extremes, with observations featuring 
# relatively high unemployment tending to have lower wages.


# ============================================================
# Part 2: ESTIMATION
# ============================================================


# define bounds for theta parameters to avoid crazy values
bounds = [
    (-5, 5),       # alpha
    (0.01, 5),     # gamma
    (-5, 5),       # mu
    (-1, 1),       # rho
    (0.01, 0.99)   # beta
]

# i. Two assumptions: 1) E[.] = 0 2)a) The firm behaves as a monopsonist

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
# Apply vmap
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
# Compute Sample Moments
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
# Define GMM function to minimize
# -------------------------------------------------

def gmm_objective(theta, data):

    moments = sample_moments(
        theta,
        data
    )

    return jnp.sum(moments ** 2)



# -------------------------------------------------
# Choose starting value of the parameters
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

theta_true = jnp.array([0.5, 0.8, 0.2, 0.1, 0.6])



# -------------------------------------------------
#  Run for scenarios 1, 2 (true monopsony!) and 3
# -------------------------------------------------


data1 = (
    jnp.array(df1["wt"].values),
    jnp.array(df1["Lt"].values),
    jnp.array(df1["ut"].values),
    jnp.array(df1["pt"].values)
)


result1 = minimize(
    fun=lambda theta: float(
        gmm_objective(theta, data1)
    ),
    x0=theta_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient(theta, data1)
    ).astype(float),
    method="L-BFGS-B",
    bounds=bounds
)

theta_hat_1 = result1.x


data2 = (
    jnp.array(df2["wt"].values),
    jnp.array(df2["Lt"].values),
    jnp.array(df2["ut"].values),
    jnp.array(df2["pt"].values)
)


result2 = minimize(
    fun=lambda theta: float(
        gmm_objective(theta, data2)
    ),
    x0=theta_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient(theta, data2)
    ).astype(float),
    method="L-BFGS-B",
    bounds=bounds
)

theta_hat_2 = result2.x


data3 = (
    jnp.array(df3["wt"].values),
    jnp.array(df3["Lt"].values),
    jnp.array(df3["ut"].values),
    jnp.array(df3["pt"].values)
)


result3 = minimize(
    fun=lambda theta: float(
        gmm_objective(theta, data3)
    ),
    x0=theta_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient(theta, data3)
    ).astype(float),
    method="L-BFGS-B",
    bounds=bounds
)

theta_hat_3 = result3.x



# ============================================================
# Export GMM results - Part 2(i)
# ============================================================

gmm_results_table = pd.DataFrame({
    "Parameter": [
        "alpha",
        "gamma",
        "mu",
        "rho",
        "beta",
        "GMM objective",
        "Function evaluations",
        "Gradient evaluations",
        "Converged"
    ],

    "Scenario 1": [
        theta_hat_1[0],
        theta_hat_1[1],
        theta_hat_1[2],
        theta_hat_1[3],
        theta_hat_1[4],
        float(gmm_objective(theta_hat_1, data1)),
        result1.nfev,
        result1.njev,
        result1.success
    ],

    "Scenario 2": [
        theta_hat_2[0],
        theta_hat_2[1],
        theta_hat_2[2],
        theta_hat_2[3],
        theta_hat_2[4],
        float(gmm_objective(theta_hat_2, data2)),
        result2.nfev,
        result2.njev,
        result2.success
    ],

    "Scenario 3": [
        theta_hat_3[0],
        theta_hat_3[1],
        theta_hat_3[2],
        theta_hat_3[3],
        theta_hat_3[4],
        float(gmm_objective(theta_hat_3, data3)),
        result3.nfev,
        result3.njev,
        result3.success
    ]
})


# Format numerical values with 4 significant digits
gmm_results_table_display = gmm_results_table.copy()

for column in ["Scenario 1", "Scenario 2", "Scenario 3"]:
    gmm_results_table_display[column] = (
        gmm_results_table_display[column]
        .map(lambda x: f"{x:.4g}" if isinstance(x, (float, int)) else x)
    )

    
gmm_results_table_display.to_csv(
    "output/gmm_estimation_results_monopsony.csv",
    index=False
)

print(gmm_results_table_display)


# ============================================================
# Part 2(ii): Estimation under wage-taking Two assumptions: 1) 
# E[.] = 0 2)a) The firm behaves as a price-taker (or wage taker)
# ============================================================


def single_t_moments_wagetaker(theta, data):
    """
    Calculates the moment conditions for one market t
    under the wage-taking assumption.
    """

    # Unpack parameters
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

    # --------------------------------------------------------
    # Labor supply residual: eta_t
    # --------------------------------------------------------

    eta = (
        log_L
        - alpha
        - gamma * log_w
        - mu * log_u
        - rho * log_u * log_w
    )

    # --------------------------------------------------------
    # Firm FOC residual: nu_t
    # Wage-taking:
    #
    # p * beta * exp(nu) * L^(beta-1) = w
    # --------------------------------------------------------

    nu = (
        log_w
        - log_p
        - jnp.log(beta)
        - (beta - 1) * log_L
    )

    # --------------------------------------------------------
    # Instruments
    # --------------------------------------------------------

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
# Apply vmap
# -------------------------------------------------

def all_t_moments_wagetaker(theta, data):

    all_moments_eta, all_moments_nu = jax.vmap(
        single_t_moments_wagetaker,
        in_axes=(None, 0)
    )(theta, data)

    return all_moments_eta, all_moments_nu


# -------------------------------------------------
# Compute sample moments
# -------------------------------------------------

def sample_moments_wagetaker(theta, data):

    moments_eta, moments_nu = all_t_moments_wagetaker(
        theta,
        data
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
# GMM objective
# -------------------------------------------------

def gmm_objective_wagetaker(theta, data):

    moments = sample_moments_wagetaker(
        theta,
        data
    )

    return jnp.sum(moments ** 2)


# -------------------------------------------------
# Gradient using JAX
# -------------------------------------------------

gmm_gradient_wagetaker = jax.jit(
    jax.grad(gmm_objective_wagetaker)
)


# -------------------------------------------------
#  Run for scenarios 1 (wage taker!) , 2 and 3
# -------------------------------------------------

gmm_results = []

result1_wt = minimize(
    fun=lambda theta: float(
        gmm_objective_wagetaker(theta, data1)
    ),
    x0=theta_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient_wagetaker(theta, data1)
    ).astype(float),
    method="L-BFGS-B",
    bounds=bounds
)

theta_hat_1_wt = result1_wt.x

result2_wt = minimize(
    fun=lambda theta: float(
        gmm_objective_wagetaker(theta, data2)
    ),
    x0=theta_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient_wagetaker(theta, data2)
    ).astype(float),
    method="L-BFGS-B",
    bounds=bounds
)

theta_hat_2_wt = result2_wt.x



result3_wt = minimize(
    fun=lambda theta: float(
        gmm_objective_wagetaker(theta, data3)
    ),
    x0=theta_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient_wagetaker(theta, data3)
    ).astype(float),
    method="L-BFGS-B",
    bounds=bounds
)

theta_hat_3_wt = result3_wt.x

# ============================================================
# Export results table
# ============================================================

results_wagetaker = pd.DataFrame({
    "Parameter": [
        "alpha",
        "gamma",
        "mu",
        "rho",
        "beta",
        "GMM objective",
        "Function evaluations",
        "Gradient evaluations",
        "Converged"
    ],

    "Scenario 1": [
        theta_hat_1_wt[0],
        theta_hat_1_wt[1],
        theta_hat_1_wt[2],
        theta_hat_1_wt[3],
        theta_hat_1_wt[4],
        float(gmm_objective_wagetaker(theta_hat_1_wt, data1)),
        result1_wt.nfev,
        result1_wt.njev,
        result1_wt.success
    ],

    "Scenario 2": [
        theta_hat_2_wt[0],
        theta_hat_2_wt[1],
        theta_hat_2_wt[2],
        theta_hat_2_wt[3],
        theta_hat_2_wt[4],
        float(gmm_objective_wagetaker(theta_hat_2_wt, data2)),
        result2_wt.nfev,
        result2_wt.njev,
        result2_wt.success
    ],

    "Scenario 3": [
        theta_hat_3_wt[0],
        theta_hat_3_wt[1],
        theta_hat_3_wt[2],
        theta_hat_3_wt[3],
        theta_hat_3_wt[4],
        float(gmm_objective_wagetaker(theta_hat_3_wt, data3)),
        result3_wt.nfev,
        result3_wt.njev,
        result3_wt.success
    ]
})



# Format numerical values with 4 significant digits
results_wagetaker_display = results_wagetaker.copy()

for column in ["Scenario 1", "Scenario 2", "Scenario 3"]:
    results_wagetaker_display[column] = (
        results_wagetaker_display[column]
        .map(lambda x: f"{x:.4g}" if isinstance(x, (float, int)) else x)
    )

results_wagetaker_display.to_csv(
    "output/gmm_estimation_results_wagetaking.csv",
    index=False
)

print(results_wagetaker_display)


# ============================================================
# Part 2(iii): Estimation with conduct parameter omega
# ============================================================

# We estimate:
# theta = (alpha, gamma, mu, rho, beta)
# and omega, which captures firm conduct.
#
# omega = 0 -> wage-taking
# omega = 1 -> monopsony


# ------------------------------------------------------------
# Moment conditions for one market
# ------------------------------------------------------------

def single_t_moments_conduct(theta_omega, data):
    """
    Calculates the six moment conditions for one market t
    under the conduct specification.

    theta_omega = (alpha, gamma, mu, rho, beta, omega)
    """

    # Unpack parameters
    alpha = theta_omega[0]
    gamma = theta_omega[1]
    mu = theta_omega[2]
    rho = theta_omega[3]
    beta = theta_omega[4]
    omega = theta_omega[5]

    # Data for market t
    w, L, u, p = data

    # Logs
    log_w = jnp.log(w)
    log_L = jnp.log(L)
    log_u = jnp.log(u)
    log_p = jnp.log(p)

    # --------------------------------------------------------
    # Labor supply residual: eta_t
    # --------------------------------------------------------

    eta = (
        log_L
        - alpha
        - gamma * log_w
        - mu * log_u
        - rho * log_u * log_w
    )

    # --------------------------------------------------------
    # Firm FOC under general conduct
    #
    # R'(L) = w + omega * L * dw/dL
    #
    # Since
    #
    # L * dw/dL = w / (gamma + rho * log(u)),
    #
    # we have
    #
    # R'(L) =
    # w * [1 + omega / (gamma + rho * log(u))]
    # --------------------------------------------------------

    nu = (
        log_w
        + jnp.log(
            1
            + omega / (gamma + rho * log_u)
        )
        - log_p
        - jnp.log(beta)
        - (beta - 1) * log_L
    )

    # --------------------------------------------------------
    # Instruments
    # --------------------------------------------------------

    Z = jnp.array([
        1.0,
        log_u,
        log_p
    ])

    # Three moments for eta and three for nu
    moments_eta = Z * eta
    moments_nu = Z * nu

    return moments_eta, moments_nu


# ------------------------------------------------------------
# Apply vmap across markets
# ------------------------------------------------------------

def all_t_moments_conduct(theta_omega, data):

    all_moments_eta, all_moments_nu = jax.vmap(
        single_t_moments_conduct,
        in_axes=(None, 0)
    )(theta_omega, data)

    return all_moments_eta, all_moments_nu


# ------------------------------------------------------------
# Compute sample moments
# ------------------------------------------------------------

def sample_moments_conduct(theta_omega, data):

    moments_eta, moments_nu = all_t_moments_conduct(
        theta_omega,
        data
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


# ------------------------------------------------------------
# GMM objective
# ------------------------------------------------------------

def gmm_objective_conduct(theta_omega, data):

    moments = sample_moments_conduct(
        theta_omega,
        data
    )

    return jnp.sum(moments ** 2)


# ------------------------------------------------------------
# Gradient using JAX
# ------------------------------------------------------------

gmm_gradient_conduct = jax.jit(
    jax.grad(gmm_objective_conduct)
)


# ------------------------------------------------------------
# Starting values
# ------------------------------------------------------------

theta_omega_start = jnp.array([
    1.0,   # alpha
    1.0,   # gamma
    1.0,   # mu
    0.0,   # rho
    0.5,   # beta
    0.5    # omega
])


# ------------------------------------------------------------
# Bounds
#
# Same bounds as Parts 2(i) and 2(ii), plus omega in [0,1].
#
# omega = 0 -> wage-taking
# omega = 1 -> monopsony
# ------------------------------------------------------------

bounds_conduct = [
    (-5, 5),       # alpha
    (0.01, 5),     # gamma
    (-5, 5),       # mu
    (-1, 1),       # rho
    (0.01, 0.99),  # beta
    (0, 1)         # omega
]


# ------------------------------------------------------------
# Run estimation on the three datasets
# ------------------------------------------------------------

result1_conduct = minimize(
    fun=lambda theta: float(
        gmm_objective_conduct(theta, data1)
    ),
    x0=theta_omega_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient_conduct(theta, data1)
    ).astype(float),
    method="L-BFGS-B",
    bounds=bounds_conduct
)

theta_omega_hat_1 = result1_conduct.x


result2_conduct = minimize(
    fun=lambda theta: float(
        gmm_objective_conduct(theta, data2)
    ),
    x0=theta_omega_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient_conduct(theta, data2)
    ).astype(float),
    method="L-BFGS-B",
    bounds=bounds_conduct
)

theta_omega_hat_2 = result2_conduct.x


result3_conduct = minimize(
    fun=lambda theta: float(
        gmm_objective_conduct(theta, data3)
    ),
    x0=theta_omega_start,
    jac=lambda theta: jnp.asarray(
        gmm_gradient_conduct(theta, data3)
    ).astype(float),
    method="L-BFGS-B",
    bounds=bounds_conduct
)

theta_omega_hat_3 = result3_conduct.x


# ------------------------------------------------------------
# Export results table
# ------------------------------------------------------------

results_conduct = pd.DataFrame({
    "Parameter": [
        "alpha",
        "gamma",
        "mu",
        "rho",
        "beta",
        "omega",
        "GMM objective",
        "Function evaluations",
        "Gradient evaluations",
        "Converged"
    ],

    "Scenario 1": [
        theta_omega_hat_1[0],
        theta_omega_hat_1[1],
        theta_omega_hat_1[2],
        theta_omega_hat_1[3],
        theta_omega_hat_1[4],
        theta_omega_hat_1[5],
        float(gmm_objective_conduct(
            theta_omega_hat_1,
            data1
        )),
        result1_conduct.nfev,
        result1_conduct.njev,
        result1_conduct.success
    ],

    "Scenario 2": [
        theta_omega_hat_2[0],
        theta_omega_hat_2[1],
        theta_omega_hat_2[2],
        theta_omega_hat_2[3],
        theta_omega_hat_2[4],
        theta_omega_hat_2[5],
        float(gmm_objective_conduct(
            theta_omega_hat_2,
            data2
        )),
        result2_conduct.nfev,
        result2_conduct.njev,
        result2_conduct.success
    ],

    "Scenario 3": [
        theta_omega_hat_3[0],
        theta_omega_hat_3[1],
        theta_omega_hat_3[2],
        theta_omega_hat_3[3],
        theta_omega_hat_3[4],
        theta_omega_hat_3[5],
        float(gmm_objective_conduct(
            theta_omega_hat_3,
            data3
        )),
        result3_conduct.nfev,
        result3_conduct.njev,
        result3_conduct.success
    ]
})


# ------------------------------------------------------------
# Format numerical values with 4 significant digits
# ------------------------------------------------------------

results_conduct_display = results_conduct.copy()

for column in ["Scenario 1", "Scenario 2", "Scenario 3"]:
    results_conduct_display[column] = (
        results_conduct_display[column]
        .map(
            lambda x: f"{x:.4g}"
            if isinstance(x, (float, int))
            else x
        )
    )


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results_conduct_display.to_csv(
    "output/gmm_estimation_results_conduct.csv",
    index=False
)

print(results_conduct_display)