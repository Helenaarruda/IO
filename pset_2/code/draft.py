

























# ==============================================================================
# Check market coverage before merging
# ==============================================================================

# Define a market as a University-Year pair
choice_markets = choice_noodles[["University", "year"]].drop_duplicates()

college_markets = (
    college_year.rename(columns={"campus": "University"})
    [["University", "year"]]
    .drop_duplicates()
)

survey_markets = (
    survey_noodles.rename(columns={"coll": "University"})
    [["University", "year"]]
    .drop_duplicates()
)

print("Unique University-Year markets")
print(f"Choice data:  {len(choice_markets)}")
print(f"College data: {len(college_markets)}")
print(f"Survey data:  {len(survey_markets)}")



# Check choice data against college-year data
check_college = choice_markets.merge(
    college_markets,
    on=["University", "year"],
    how="left",
    indicator=True
)

print("\nChoice markets without a match in college-year:")
print(
    check_college.loc[
        check_college["_merge"] == "left_only",
        ["University", "year"]
    ].sort_values(["University", "year"])
)

print("\nNumber of unmatched markets in college-year:")
print((check_college["_merge"] == "left_only").sum())


# Check choice data against survey data
check_survey = choice_markets.merge(
    survey_markets,
    on=["University", "year"],
    how="left",
    indicator=True
)

print("\nChoice markets without a match in survey:")
print(
    check_survey.loc[
        check_survey["_merge"] == "left_only",
        ["University", "year"]
    ].sort_values(["University", "year"])
)

print("\nNumber of unmatched markets in survey:")
print((check_survey["_merge"] == "left_only").sum())



# Restrict choice data to the estimation period
choice_period = choice_noodles.loc[
    choice_noodles["year"].between(2010, 2020)
].copy()

# First merge: choice + college-year
merge_college = choice_period.merge(
    college_year.rename(columns={"campus": "University"}),
    on=["University", "year"],
    how="left",
    indicator=True,
    validate="many_to_one"
)

print("\nAfter merging choice with college-year")
print(f"Original choice observations: {len(choice_period)}")
print(f"Matched observations: {(merge_college['_merge'] == 'both').sum()}")
print(f"Unmatched observations: {(merge_college['_merge'] == 'left_only').sum()}")

# Keep only matched observations, replicating the effect of inner merge
merge_college = merge_college.loc[
    merge_college["_merge"] == "both"
].drop(columns="_merge")

# Second merge: add survey
merge_survey = merge_college.merge(
    survey_market,
    on=["University", "year"],
    how="left",
    indicator=True,
    validate="many_to_one"
)

print("\nAfter adding survey data")
print(f"Observations before survey merge: {len(merge_college)}")
print(f"Matched observations: {(merge_survey['_merge'] == 'both').sum()}")
print(f"Unmatched observations: {(merge_survey['_merge'] == 'left_only').sum()}")




# testes exploratorios
def montar_tabelas_cardapio(
    dados: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resume a disponibilidade dos pratos por ano e marca."""
    opcoes_unicas = (
        dados[["year", "chain", "dish"]]
        .drop_duplicates()
        .sort_values(["year", "chain", "dish"])
    )

    # Uma linha para cada ano e marca, com os pratos disponíveis naquele ano.
    cardapio_por_ano = (
        opcoes_unicas.groupby(["year", "chain"], as_index=False)
        .agg(dishes_disponiveis=("dish", ", ".join))
        .sort_values(["year", "chain"])
    )

    # Primeiro e último ano observado para cada combinação marca--prato.
    entrada_dish = (
        opcoes_unicas.groupby(["chain", "dish"], as_index=False)
        .agg(primeiro_ano=("year", "min"), ultimo_ano=("year", "max"))
        .sort_values(["primeiro_ano", "chain", "dish"])
    )

    entrada_gluten_free = entrada_dish.loc[
        entrada_dish["dish"].str.endswith("_gf")
    ].reset_index(drop=True)

    return cardapio_por_ano, entrada_dish, entrada_gluten_free


cardapio_por_ano, entrada_dish, entrada_gluten_free = montar_tabelas_cardapio(
    choice_noodles
)


import pandas as pd

# 1. Basic dimensions
print("CHOICE_NOODLES")
print("Rows:", len(choice_noodles))
print("Universities:", choice_noodles["University"].nunique())
print("Years:", choice_noodles["year"].min(), "-", choice_noodles["year"].max())
print("Unique chain-dish combinations:",
      choice_noodles[["chain", "dish"]].drop_duplicates().shape[0])

print("\nCOLLEGE_YEAR")
print("Rows:", len(college_year))
print("Campuses:", college_year["campus"].nunique())
print("Years:", college_year["year"].min(), "-", college_year["year"].max())


choice_noodles["product"] = (
    choice_noodles["chain"].astype(str) + " - " +
    choice_noodles["dish"].astype(str)
)

# Number of universities where each product is observed in each year
coverage = (
    choice_noodles
    .groupby(["product", "year"])["University"]
    .nunique()
    .reset_index(name="n_universities")
)

print(coverage["n_universities"].describe())

# Distribution of the number of universities per product-year
print(
    coverage["n_universities"]
    .value_counts()
    .sort_index()
)

# Number of observations per product-campus pair
panel_coverage = (
    choice_noodles
    .groupby(["product", "University"])["year"]
    .nunique()
    .reset_index(name="n_years")
)

print(panel_coverage["n_years"].describe())

# How many product-campus pairs are observed in each number of years?
print(panel_coverage["n_years"].value_counts().sort_index())


# Check uniqueness of product-campus-year observations
duplicates = choice_noodles.duplicated(
    subset=["chain", "dish", "University", "year"]
).sum()

print("Duplicate observations:", duplicates)

# Inspect panel coverage by product
product_coverage = (
    choice_noodles.groupby(["chain", "dish"])
    .agg(
        n_years=("year", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_universities=("University", "nunique")
    )
    .reset_index()
)

print(product_coverage.to_string(index=False))


# ==============================================================================
# Part 1: Pricing Equations
# ==============================================================================

# Variables in the first pricing regression
dependent = "Price"
regressors = ["SpicyIndex", "WheatContent", "Veg2meat", "Calories"]

# Keep only observations with non-missing values in the regression variables
reg_data = choice_noodles[[dependent] + regressors].dropna().copy()

# OLS: Price_jmt = beta_0 + beta_1*SpicyIndex + ... + u_jmt
X = sm.add_constant(reg_data[regressors])
y = reg_data[dependent]

# HC1 heteroskedasticity-robust standard errors
model = sm.OLS(y, X).fit(cov_type="HC1")

# Create a clean coefficient table
ci = model.conf_int()
table = pd.DataFrame({
    "Variable": model.params.index,
    "Coefficient": model.params.values,
    "Robust SE (HC1)": model.bse.values,
    "t-statistic": model.tvalues.values,
    "p-value": model.pvalues.values,
    "CI 95% lower": ci[0].values,
    "CI 95% upper": ci[1].values,
})

# Add regression summary statistics
summary = pd.DataFrame({
    "Statistic": ["Observations", "R-squared", "Adjusted R-squared"],
    "Value": [int(model.nobs), model.rsquared, model.rsquared_adj],
})

# Save tables for use in the .qmd
table.to_csv("pset_2/data/output/pricing_regression_1_coefficients.csv", index=False)


print(model.summary())


# --------------------------------------------------
# 1. Prepare and merge the datasets
# --------------------------------------------------

df = choice_noodles.copy()
costs = college_year.copy()

# Check whether campus names match across datasets
print("Universities:", sorted(df["University"].unique()))
print("Campuses:", sorted(costs["campus"].unique()))

df = df.merge(
    costs,
    left_on=["University", "year"],
    right_on=["campus", "year"],
    how="left",
    validate="many_to_one"
)

# Verify that the merge worked
cost_vars = [
    "WageIDX", "ReEsIDX",
    "WheatPIDX", "VegPIDX", "MeatPIDX"
]

# --------------------------------------------------
# 2. Construct product-market-specific cost shocks
# --------------------------------------------------

# These interactions are proxies for product exposure
# to local input-price variation.

df["wheat_shock"] = df["WheatPIDX"] * df["WheatContent"]

df["vegetable_shock"] = df["VegPIDX"] * df["Veg2meat"]

# Veg2meat is a ratio, not an exact ingredient quantity.
# Its inverse is used as a proxy for relative meat intensity.
df["meat_shock"] = np.where(
    df["Veg2meat"] > 0,
    df["MeatPIDX"] / df["Veg2meat"],
    np.nan
)


# --------------------------------------------------
# 3. Construct Hausman-style instrument
# --------------------------------------------------

product_year = ["chain", "dish", "year"]

grouped = df.groupby(product_year)["Price"]

df["price_sum_group"] = grouped.transform("sum")
df["n_markets_group"] = grouped.transform("count")

df["hausman_price"] = (
    (df["price_sum_group"] - df["Price"])
    / (df["n_markets_group"] - 1)
)

# Require at least two other markets
df.loc[
    df["n_markets_group"] < 3,
    "hausman_price"
] = np.nan


# --------------------------------------------------
# 4. Define the three specifications
# --------------------------------------------------

characteristics = [
    "SpicyIndex", "WheatContent", "Veg2meat", "Calories"
]

specifications = {
    "(1) Market costs": characteristics + [
        "WageIDX", "ReEsIDX"
    ],

    "(2) + Input shocks": characteristics + [
        "WageIDX", "ReEsIDX",
        "wheat_shock", "vegetable_shock", "meat_shock"
    ],

    "(3) + Hausman": characteristics + [
        "WageIDX", "ReEsIDX",
        "wheat_shock", "vegetable_shock", "meat_shock",
        "hausman_price"
    ]
}


# --------------------------------------------------
# 5. Estimate OLS regressions with HC1 robust SEs
# --------------------------------------------------

models = {}

for name, regressors in specifications.items():

    # Use the same sample across specifications so that
    # changes in results are not driven by sample composition.
    required = ["Price"] + regressors

    sample = df[required].dropna()

    X = sm.add_constant(sample[regressors])
    y = sample["Price"]

    models[name] = sm.OLS(y, X).fit(cov_type="HC1")

    print(f"\n{name}")
    print("Observations:", int(models[name].nobs))
    print("R-squared:", round(models[name].rsquared, 4))


# --------------------------------------------------
# 6. Create a comparison table
# --------------------------------------------------

all_variables = list(dict.fromkeys(
    ["const"] + [
        variable
        for regressors in specifications.values()
        for variable in regressors
    ]
))

table = pd.DataFrame(index=all_variables)

for name, model in models.items():

    values = []

    for variable in all_variables:

        if variable in model.params.index:
            coef = model.params[variable]
            se = model.bse[variable]

            # Conventional significance notation
            p = model.pvalues[variable]
            stars = "***" if p < 0.01 else (
                "**" if p < 0.05 else (
                    "*" if p < 0.10 else ""
                )
            )

            values.append(f"{coef:.4f}{stars}\n({se:.4f})")

        else:
            values.append("—")

    table[name] = values

table.index.name = "Variable"

# Add model-level statistics
table.loc["Observations"] = [
    f"{int(models[name].nobs)}" for name in models
]

table.loc["R-squared"] = [
    f"{models[name].rsquared:.4f}" for name in models
]

# Save the comparison table
output_path = "pset_2/data/output/pricing_regressions_comparison.csv"

table.to_csv(output_path)

print("\nComparison table:")
print(table.to_string())

# Joint tests using heteroskedasticity-robust covariance (HC1)

# Test A: market-level cost shifters
test_market = model2.f_test(
    "WageIDX = 0, ReEsIDX = 0"
)

# Test B: product-market-specific input-cost shocks
test_inputs = model2.f_test(
    "wheat_shock = 0, vegetable_shock = 0, meat_shock = 0"
)

# Test C: all cost shocks jointly
test_all = model2.f_test(
    "WageIDX = 0, ReEsIDX = 0, "
    "wheat_shock = 0, vegetable_shock = 0, meat_shock = 0"
)

for name, test in [
    ("Market-level cost shifters", test_market),
    ("Input-cost shocks", test_inputs),
    ("All cost shocks", test_all)
]:
    print(f"\n{name}")
    print(f"F-statistic: {float(test.statistic):.3f}")
    print(f"p-value: {float(test.pvalue):.4f}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("pset_2/data/input/choice_noodles.csv")

# Identify gluten-free products
df["GF"] = df["dish"].str.endswith("_gf")

# --------------------------------------------------
# 1A. Market shares by product category
# --------------------------------------------------

# Sum product shares within each university-year
campus_year = (
    df.groupby(["University", "year", "GF"], as_index=False)["Share"]
      .sum()
      .pivot(index=["University", "year"], columns="GF", values="Share")
      .reset_index()
      .rename(columns={False: "Traditional", True: "GF"})
)

campus_year[["Traditional", "GF"]] = (
    campus_year[["Traditional", "GF"]].fillna(0)
)

campus_year["Total_Noodles"] = (
    campus_year["Traditional"] + campus_year["GF"]
)

# Average across universities for each year
category_trends = (
    campus_year.groupby("year")[
        ["GF", "Traditional", "Total_Noodles"]
    ]
    .mean()
    .reset_index()
)

# Plot category market shares over time
plt.figure(figsize=(10, 6))

for variable, label in [
    ("GF", "Gluten-free noodles"),
    ("Traditional", "Traditional noodles"),
    ("Total_Noodles", "All noodles"),
]:
    plt.plot(
        category_trends["year"],
        category_trends[variable],
        marker="o",
        label=label
    )

plt.axvline(2011, linestyle="--", color="gray", alpha=0.7)
plt.axvline(2016, linestyle="--", color="gray", alpha=0.7)

plt.text(2011, plt.ylim()[1] * 0.95, "YourNoodles GF", rotation=90,
         va="top", ha="right", fontsize=9)
plt.text(2016, plt.ylim()[1] * 0.95, "BestNoodles & AllNoodles GF",
         rotation=90, va="top", ha="right", fontsize=9)

plt.xlabel("Year")
plt.ylabel("Average market share")
plt.title("Market Shares by Product Category")
plt.legend()
plt.grid(alpha=0.25)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()


plt.savefig(
    output_dir / "market_shares_by_category.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Save category market shares
category_trends.to_csv(
    output_dir / "market_shares_by_category.csv",
    index=False
)

# --------------------------------------------------
# 1B. Market shares by chain
# --------------------------------------------------

# Sum dish shares within each chain-university-year
chain_campus_year = (
    df.groupby(["chain", "University", "year"], as_index=False)
      .agg(chain_share=("Share", "sum"))
)

# Average chain shares across universities
chain_trends = (
    chain_campus_year
    .groupby(["chain", "year"], as_index=False)
    .agg(mean_share=("chain_share", "mean"))
)

plt.figure(figsize=(10, 6))

for chain, data in chain_trends.groupby("chain"):
    plt.plot(
        data["year"],
        data["mean_share"],
        marker="o",
        label=chain
    )

plt.axvline(2011, linestyle="--", color="gray", alpha=0.7)
plt.axvline(2016, linestyle="--", color="gray", alpha=0.7)

plt.xlabel("Year")
plt.ylabel("Average market share")
plt.title("Market Shares by Noodle Chain")
plt.legend()
plt.grid(alpha=0.25)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

plt.savefig(
    output_dir / "market_shares_by_chain.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Save chain market shares
chain_trends.to_csv(
    output_dir / "market_shares_by_chain.csv",
    index=False
)

# Save campus-level market shares, useful for further analysis
campus_year.to_csv(
    output_dir / "market_shares_by_campus_year.csv",
    index=False
)

chain_campus_year.to_csv(
    output_dir / "market_shares_by_chain_campus_year.csv",
    index=False
)



# Load survey data
survey = pd.read_csv(
    "pset_2/data/input/survey_noodles.csv",
    sep="\t"
)

# Check coding before analysis
print("Graduate categories:")
print(survey["Graduate"].value_counts(dropna=False))

print("\nAthlete categories:")
print(survey["Athlete"].value_counts(dropna=False))

# Indicator for gluten-free observed and hypothetical choices
survey["GF_Obs"] = survey["Pick_Obs"].str.endswith("_gf").astype(int)
survey["GF_Hyp"] = survey["Pick_Hyp"].str.endswith("_gf").astype(int)

# Clean group indicators
survey["Athlete_D"] = np.where(
    survey["Athlete"].eq("Y"), 1,
    np.where(survey["Athlete"].eq("N"), 0, np.nan)
)

survey["Graduate_D"] = np.where(
    survey["Graduate"].eq("Y"), 1,
    np.where(survey["Graduate"].eq("N"), 0, np.nan)
)

# --------------------------------------------------
# 2A. GF choice over time by student group
# --------------------------------------------------

# Athlete: Y = athlete, N = non-athlete
survey["Athlete_D"] = np.where(
    survey["Athlete"].eq("Y"), 1,
    np.where(survey["Athlete"].eq("N"), 0, np.nan)
)

# Graduate: Y = graduate student; N and No = non-graduate student
survey["Graduate_D"] = np.where(
    survey["Graduate"].eq("Y"), 1,
    np.where(survey["Graduate"].isin(["N", "No"]), 0, np.nan)
)

# Check recoding
print("Graduate status after recoding:")
print(survey["Graduate_D"].value_counts(dropna=False))


# --------------------------------------------------
# 2A. GF choice over time by student group
# --------------------------------------------------

# Athlete vs. non-athlete
athlete_trends = (
    survey.dropna(subset=["Athlete_D"])
    .groupby(["year", "Athlete_D"], as_index=False)
    .agg(gf_share=("GF_Obs", "mean"))
)

plt.figure(figsize=(9, 5))

for group, data in athlete_trends.groupby("Athlete_D"):
    label = "Athletes" if group == 1 else "Non-athletes"
    plt.plot(data["year"], data["gf_share"], marker="o", label=label)

plt.axvline(2011, linestyle="--", color="gray", alpha=0.7)
plt.axvline(2016, linestyle="--", color="gray", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Share choosing GF")
plt.title("Observed GF Choice by Athlete Status")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()


plt.savefig(
    output_dir / "gf_choice_by_athlete_status.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

athlete_trends.to_csv(
    output_dir / "gf_choice_by_athlete_status.csv",
    index=False
)

# Graduate vs. non-graduate
graduate_trends = (
    survey.dropna(subset=["Graduate_D"])
    .groupby(["year", "Graduate_D"], as_index=False)
    .agg(gf_share=("GF_Obs", "mean"))
)

plt.figure(figsize=(9, 5))

for group, data in graduate_trends.groupby("Graduate_D"):
    label = "Graduate students" if group == 1 else "Non-graduate students"
    plt.plot(data["year"], data["gf_share"], marker="o", label=label)

plt.axvline(2011, linestyle="--", color="gray", alpha=0.7)
plt.axvline(2016, linestyle="--", color="gray", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Share choosing GF")
plt.title("Observed GF Choice by Graduate Status")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()


plt.xlabel("Year")
plt.ylabel("Share choosing GF")
plt.title("Observed GF Choice by Graduate Status")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    output_dir / "gf_choice_by_graduate_status.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

graduate_trends.to_csv(
    output_dir / "gf_choice_by_graduate_status.csv",
    index=False
)


# --------------------------------------------------
# 2B. Descriptive regressions for GF choice
# --------------------------------------------------

# Athlete model
athlete_sample = survey.dropna(
    subset=["Athlete_D", "GF_Obs", "coll", "year"]
).copy()

model_athlete = smf.ols(
    "GF_Obs ~ C(year) * Athlete_D",
    data=athlete_sample
).fit(
    cov_type="cluster",
    cov_kwds={"groups": athlete_sample["coll"]}
)

print("\nGF choice regression: Athlete status")
print(model_athlete.summary())


# Graduate model: N and No are both coded as 0
graduate_sample = survey.dropna(
    subset=["Graduate_D", "GF_Obs", "coll", "year"]
).copy()

model_graduate = smf.ols(
    "GF_Obs ~ C(year) * Graduate_D",
    data=graduate_sample
).fit(
    cov_type="cluster",
    cov_kwds={"groups": graduate_sample["coll"]}
)

print("\nGF choice regression: Graduate status")
print(model_graduate.summary())

# Save regression coefficient tables
def save_regression_results(model, filename):
    results = pd.DataFrame({
        "coefficient": model.params,
        "std_error": model.bse,
        "z_statistic": model.tvalues,
        "p_value": model.pvalues,
        "ci_lower": model.conf_int()[0],
        "ci_upper": model.conf_int()[1]
    })

    results.index.name = "term"
    results.to_csv(output_dir / filename)


save_regression_results(
    model_athlete,
    "regression_gf_choice_athlete.csv"
)

save_regression_results(
    model_graduate,
    "regression_gf_choice_graduate.csv"
)



# --------------------------------------------------
# Joint tests: are all group-year interactions zero?
# --------------------------------------------------

joint_test_results = []

def joint_interaction_test(model, group_name):
    interaction_names = [
        name for name in model.params.index
        if ":" + group_name in name
    ]

    if not interaction_names:
        print(f"No interactions found for {group_name}.")
        return

    param_names = list(model.params.index)
    R = np.zeros((len(interaction_names), len(param_names)))

    for i, name in enumerate(interaction_names):
        R[i, param_names.index(name)] = 1

    result = model.wald_test(R, use_f=False, scalar=True)

    row = {
        "group": group_name,
        "null_hypothesis": "All year-by-group interactions equal zero",
        "n_restrictions": len(interaction_names),
        "wald_statistic": float(result.statistic),
        "p_value_asymptotic_chi2": float(result.pvalue)
    }

    joint_test_results.append(row)

    print(f"\nJoint test for {group_name}")
    print("H0: All year-by-group interactions equal zero")
    print(f"Number of restrictions: {len(interaction_names)}")
    print(f"Wald statistic: {float(result.statistic):.4f}")
    print(f"p-value (asymptotic chi-square): {float(result.pvalue):.4f}")


joint_interaction_test(model_athlete, "Athlete_D")
joint_interaction_test(model_graduate, "Graduate_D")

# Save joint test results
pd.DataFrame(joint_test_results).to_csv(
    output_dir / "joint_interaction_tests.csv",
    index=False
)



# ==============================================================================
# Exhibit 3 Price elasticities: choices + student survey + input-price data
# ==============================================================================

# The survey covers 2010--2020.  Aggregate it to the same university-year
# market definition used in choice_noodles, then merge all three datasets.
survey_market = (
    survey_noodles.assign(
        athlete=lambda x: x["Athlete"].eq("Y"),
        graduate=lambda x: x["Graduate"].eq("Y"),
        survey_gf_choice=lambda x: x["Pick_Obs"].str.endswith("_gf"),
    )
    .groupby(["coll", "year"], as_index=False)
    .agg(
        survey_athlete_share=("athlete", "mean"),
        survey_graduate_share=("graduate", "mean"),
        survey_gf_choice_share=("survey_gf_choice", "mean"),
    )
    .rename(columns={"coll": "University"})
)

college_market = college_year.rename(columns={"campus": "University"})

elasticity_data = (
    choice_noodles.loc[choice_noodles["year"].between(2010, 2020)].copy()
    .merge(college_market, on=["University", "year"], how="inner", validate="many_to_one")
    .merge(survey_market, on=["University", "year"], how="inner", validate="many_to_one")
)

# Number of GF alternatives available in each market captures their staged entry:
# YN_gf from 2011 and AN_gf/BN_gf from 2016.
gf_options = (
    elasticity_data.assign(is_gf=elasticity_data["dish"].isin(GLUTEN_FREE_DISHES))
    .groupby(["University", "year"], as_index=False)["is_gf"]
    .sum()
    .rename(columns={"is_gf": "gf_options"})
)
elasticity_data = elasticity_data.merge(
    gf_options, on=["University", "year"], how="left", validate="many_to_one"
)


def estimate_log_elasticities(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate dish-specific log-log elasticities with market controls.

    The dependent variable is log market share.  Thus, coefficients on log
    price and the log input-price indexes are elasticities.  The GF-options
    coefficient is an elasticity with respect to 1 + number of GF options.
    """
    variables = {
        "Own price": "Price",
        "Vegetable input": "VegPIDX",
        "Wheat input": "WheatPIDX",
        "Meat input": "MeatPIDX",
        "Wage index": "WageIDX",
        "Real-estate index": "ReEsIDX",
        "GF options": "gf_options",
    }
    rows = []

    for dish, dish_data in data.groupby("dish"):
        log_x = np.column_stack(
            [
                np.log1p(dish_data[column].to_numpy())
                if column == "gf_options"
                else np.log(dish_data[column].to_numpy())
                for column in variables.values()
            ]
        )
        controls = np.column_stack(
            [
                np.ones(len(dish_data)),
                dish_data["year"].to_numpy() - dish_data["year"].min(),
                dish_data["survey_athlete_share"].to_numpy(),
                dish_data["survey_graduate_share"].to_numpy(),
                dish_data["survey_gf_choice_share"].to_numpy(),
            ]
        )
        design = np.column_stack([log_x, controls])
        coefficients, *_ = np.linalg.lstsq(
            design, np.log(dish_data["Share"].to_numpy()), rcond=None
        )
        rows.append({"dish": dish, **dict(zip(variables, coefficients[:len(variables)]))})

    return pd.DataFrame(rows).set_index("dish").sort_index()


elasticity_table = estimate_log_elasticities(elasticity_data)
print("\nEstimated elasticities (2010--2020):")
print(elasticity_table.round(3))

# Heatmap: red means a negative association; blue means a positive association.
fig_heatmap, ax_heatmap = plt.subplots(figsize=(11, 6))
limit = np.abs(elasticity_table.to_numpy()).max()
image = ax_heatmap.imshow(
    elasticity_table.to_numpy(), cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto"
)
ax_heatmap.set_xticks(range(len(elasticity_table.columns)), elasticity_table.columns, rotation=35, ha="right")
ax_heatmap.set_yticks(range(len(elasticity_table.index)), elasticity_table.index)
ax_heatmap.set_title("Estimated elasticities by noodle dish (2010--2020)")

for row in range(elasticity_table.shape[0]):
    for column in range(elasticity_table.shape[1]):
        value = elasticity_table.iloc[row, column]
        color = "white" if abs(value) > limit * 0.55 else "black"
        ax_heatmap.text(column, row, f"{value:.2f}", ha="center", va="center", color=color)

fig_heatmap.colorbar(image, ax=ax_heatmap, label="Elasticity estimate")
fig_heatmap.tight_layout()

# Show the original adoption chart and the elasticity heatmap together.
plt.show()

# Save figure
plt.savefig(
    "pset_2/data/output/raw_elasticities.png",
    dpi=300,
    bbox_inches="tight"
)
