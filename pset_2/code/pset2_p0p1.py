import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.ticker import MaxNLocator

# ==============================================================================
# Part 0 Responses — Input and descriptive evidence
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "pset_2" / "data" / "input"
output_dir = PROJECT_ROOT / "pset_2" / "data" / "output"
output_dir.mkdir(parents=True, exist_ok=True)


def carregar_datasets() -> dict[str, pd.DataFrame]:
    """Carrega os arquivos em um dicionário de DataFrames."""
    return {
        "choice_noodles": pd.read_csv(INPUT_DIR / "choice_noodles.csv"),
        "survey_noodles": pd.read_csv(INPUT_DIR / "survey_noodles.csv", sep="\t"),
        "college_year": pd.read_excel(INPUT_DIR / "college-year.xls"),
    }


datasets = carregar_datasets()

choice_noodles = datasets["choice_noodles"]
survey_noodles = datasets["survey_noodles"]
college_year = datasets["college_year"]


# --------------------------------------------------
# Standardize university names across datasets
# --------------------------------------------------

choice_map = {
    "Texas A&M": "A&M",
    "Univ. Chicago": "University of Chicago",
    "Saint John": "St. John's",
    "Rose-Hul.": "Rose-Hulman",
}

choice_noodles["University"] = (
    choice_noodles["University"].str.strip().replace(choice_map)
)

# Standardize campus in college_year
college_map = {
    "Texas-A&M": "A&M",
    "U. Chicago": "University of Chicago",
    "St Johns": "St. John's",
    "R-H": "Rose-Hulman",
}

college_year["campus"] = (
    college_year["campus"].str.strip().replace(college_map)
)

# Standardize coll in survey_noodles
survey_map = {
    "Colorado C.": "Colorado",
}

survey_noodles["coll"] = (
    survey_noodles["coll"].str.strip().replace(survey_map)
)

# ==============================================================================
# Part 1 Responses — Pricing and reduced-form analysis
# ==============================================================================

# Pricing equations
# ==================================================

# Merge choice data and cost data
# ==================================================

df = choice_noodles.copy()
costs = college_year.copy()

df = df.merge(
    costs,
    left_on=["University", "year"],
    right_on=["campus", "year"],
    how="left",
    validate="many_to_one"
)

cost_vars = [
    "WageIDX", "ReEsIDX",
    "WheatPIDX", "VegPIDX", "MeatPIDX"
]

print("Missing cost indices after merge:")
print(df[cost_vars].isna().sum())


# Construct product-market-specific cost shocks
# ==================================================

df["wheat_shock"] = (
    df["WheatPIDX"] * df["WheatContent"]
)

df["vegetable_shock"] = (
    df["VegPIDX"] * df["Veg2meat"]
)

df["meat_shock"] = np.where(
    df["Veg2meat"] > 0,
    df["MeatPIDX"] / df["Veg2meat"],
    np.nan
)


# Define the four specifications
# ==================================================

dependent = "Price"

characteristics = [
    "SpicyIndex",
    "WheatContent",
    "Veg2meat",
    "Calories"
]

market_shocks = [
    "WageIDX",
    "ReEsIDX"
]

input_shocks = [
    "wheat_shock",
    "vegetable_shock",
    "meat_shock"
]

specifications = {
    "(1) Baseline": characteristics,

    "(2) Market shocks": (
        characteristics + market_shocks
    ),

    "(3) Input-cost shocks": (
        characteristics + input_shocks
    ),

    "(4) All cost shocks": (
        characteristics + market_shocks + input_shocks
    )
}


# Use the same sample for all four regressions
# ==================================================

all_required_vars = list(dict.fromkeys(
    [dependent]
    + characteristics
    + market_shocks
    + input_shocks
))

reg_data = df[all_required_vars].dropna().copy()

print("\nCommon estimation sample:", len(reg_data))


# Estimate all regressions using OLS + HC1
# ==================================================

models = {}

for name, regressors in specifications.items():

    X = sm.add_constant(reg_data[regressors])
    y = reg_data[dependent]

    model = sm.OLS(y, X).fit(cov_type="HC1")
    models[name] = model

    print(f"\n{name}")
    print(f"Observations: {int(model.nobs)}")
    print(f"R-squared: {model.rsquared:.4f}")


# Joint F-tests (robust Wald tests)
# ==================================================

def run_joint_test(model, variables):
    """
    Test whether all coefficients in variables
    are jointly equal to zero, using HC1 covariance.
    """
    restrictions = ", ".join(
        f"{variable} = 0" for variable in variables
    )

    result = model.f_test(restrictions)

    return {
        "F-statistic": float(np.asarray(result.fvalue).item()),
        "p-value": float(np.asarray(result.pvalue).item())
    }


# Each model tests the shocks included in that specification.
# Model 2: market shocks jointly
# Model 3: input-cost shocks jointly
# Model 4: all five cost shocks jointly

test_results = {}

test_results["(2) Market shocks"] = run_joint_test(
    models["(2) Market shocks"],
    market_shocks
)

test_results["(3) Input-cost shocks"] = run_joint_test(
    models["(3) Input-cost shocks"],
    input_shocks
)

test_results["(4) All cost shocks"] = run_joint_test(
    models["(4) All cost shocks"],
    market_shocks + input_shocks
)

# Additional tests within the full specification:
# market shocks jointly, conditional on input-cost shocks
test_market_full = run_joint_test(
    models["(4) All cost shocks"],
    market_shocks
)

# input-cost shocks jointly, conditional on market shocks
test_inputs_full = run_joint_test(
    models["(4) All cost shocks"],
    input_shocks
)


# Save coefficient comparison table
# ==================================================

all_variables = list(dict.fromkeys(
    ["const"] + [
        variable
        for regressors in specifications.values()
        for variable in regressors
    ]
))

coef_table = pd.DataFrame(index=all_variables)

for name, model in models.items():

    values = []

    for variable in all_variables:

        if variable not in model.params.index:
            values.append("—")
            continue

        coef = model.params[variable]
        se = model.bse[variable]
        p = model.pvalues[variable]

        stars = (
            "***" if p < 0.01
            else "**" if p < 0.05
            else "*" if p < 0.10
            else ""
        )

        values.append(
            f"{coef:.4f}{stars} ({se:.4f})"
        )

    coef_table[name] = values

coef_table.index.name = "Variable"

# Append model-level statistics to the coefficient table
coef_table.loc["Observations"] = [
    f"{int(models[name].nobs)}"
    for name in specifications
]

coef_table.loc["R-squared"] = [
    f"{models[name].rsquared:.4f}"
    for name in specifications
]

coef_table.to_csv(
    output_dir / "pricing_regressions_comparison.csv"
)


# Save N, R-squared, and F-test results
# ==================================================

stats_rows = []

for name, model in models.items():

    row = {
        "Specification": name,
        "Observations": int(model.nobs),
        "R-squared": model.rsquared,
        "Adjusted R-squared": model.rsquared_adj,
        "F-statistic (included shocks)": np.nan,
        "p-value (included shocks)": np.nan,
        "F-statistic (market shocks, full model)": np.nan,
        "p-value (market shocks, full model)": np.nan,
        "F-statistic (input shocks, full model)": np.nan,
        "p-value (input shocks, full model)": np.nan
    }

    if name in test_results:
        row["F-statistic (included shocks)"] = (
            test_results[name]["F-statistic"]
        )
        row["p-value (included shocks)"] = (
            test_results[name]["p-value"]
        )

    if name == "(4) All cost shocks":
        row["F-statistic (market shocks, full model)"] = (
            test_market_full["F-statistic"]
        )
        row["p-value (market shocks, full model)"] = (
            test_market_full["p-value"]
        )

        row["F-statistic (input shocks, full model)"] = (
            test_inputs_full["F-statistic"]
        )
        row["p-value (input shocks, full model)"] = (
            test_inputs_full["p-value"]
        )

    stats_rows.append(row)

stats_table = pd.DataFrame(stats_rows)

stats_table.to_csv(
    output_dir / "pricing_regressions_stats_Ftests.csv",
    index=False
)


# Print results
# ==================================================

print("\nCOEFFICIENT COMPARISON")
print(coef_table.to_string())

print("\nMODEL STATISTICS AND JOINT TESTS")
print(stats_table.round(4).to_string(index=False))

print("\nSaved files:")
print(output_dir / "pricing_regressions_comparison.csv")
print(output_dir / "pricing_regressions_stats_Ftests.csv")




# Reduced-form results and Descriptive Exhibits
# ==================================================


# Define adoption year for each chain
# ------------------------------------------------------------

adoption_year = {
    "YourNoodles": 2011,
    "AllNoodles": 2016,
    "BestNoodles": 2016
}


# Aggregate product shares to chain-campus-year
# ------------------------------------------------------------

chain_panel = (
    choice_noodles
    .groupby(["chain", "University", "year"], as_index=False)
    .agg(chain_share=("Share", "sum"))
)

chain_panel["adoption_year"] = (
    chain_panel["chain"].map(adoption_year)
)

chain_panel["event_time"] = (
    chain_panel["year"] - chain_panel["adoption_year"]
)

print("Number of observations:", len(chain_panel))
print("\nObservations by chain:")
print(chain_panel.groupby("chain").size())


# Create event-time indicators
# Reference period: k = -1
# Bin tails to retain all available years
# ------------------------------------------------------------

event_bins = {
    "event_le_m6": chain_panel["event_time"] <= -6,
    "event_m5": chain_panel["event_time"] == -5,
    "event_m4": chain_panel["event_time"] == -4,
    "event_m3": chain_panel["event_time"] == -3,
    "event_m2": chain_panel["event_time"] == -2,
    "event_0": chain_panel["event_time"] == 0,
    "event_p1": chain_panel["event_time"] == 1,
    "event_p2": chain_panel["event_time"] == 2,
    "event_p3": chain_panel["event_time"] == 3,
    "event_p4": chain_panel["event_time"] == 4,
    "event_ge_p5": chain_panel["event_time"] >= 5
}

for variable, condition in event_bins.items():
    chain_panel[variable] = condition.astype(int)

event_terms = list(event_bins.keys())


# Estimate reduced-form event-study regression
# ------------------------------------------------------------

formula = (
    "chain_share ~ "
    + " + ".join(event_terms)
    + " + C(chain) + C(University):C(year)"
)

event_model = smf.ols(
    formula,
    data=chain_panel
).fit(
    cov_type="cluster",
    cov_kwds={"groups": chain_panel["University"]}
)

print("\nEVENT-STUDY REGRESSION")
print(f"Observations: {int(event_model.nobs)}")
print(f"R-squared: {event_model.rsquared:.4f}")

# Pre-treatment coefficients
pre_terms = [
    "event_le_m6",
    "event_m5",
    "event_m4",
    "event_m3",
    "event_m2"
]

# Post-treatment coefficients
post_terms = [
    "event_0",
    "event_p1",
    "event_p2",
    "event_p3",
    "event_p4",
    "event_ge_p5"
]

def joint_wald_test(model, terms):
    restriction = ", ".join(f"{term} = 0" for term in terms)
    result = model.f_test(restriction)

    print(f"F-statistic: {float(np.asarray(result.fvalue).item()):.3f}")
    print(f"p-value: {float(np.asarray(result.pvalue).item()):.4f}")

print("Pre-treatment coefficients jointly equal to zero:")
joint_wald_test(event_model, pre_terms)

print("\nPost-treatment coefficients jointly equal to zero:")
joint_wald_test(event_model, post_terms)

# Extract event-study coefficients and confidence intervals
# ------------------------------------------------------------

event_labels = {
    "event_le_m6": "≤ -6",
    "event_m5": "-5",
    "event_m4": "-4",
    "event_m3": "-3",
    "event_m2": "-2",
    "event_0": "0",
    "event_p1": "1",
    "event_p2": "2",
    "event_p3": "3",
    "event_p4": "4",
    "event_ge_p5": "≥ 5"
}

results = []

for variable in event_terms:
    ci_low, ci_high = event_model.conf_int().loc[variable]

    results.append({
        "Event time": event_labels[variable],
        "Variable": variable,
        "Coefficient": event_model.params[variable],
        "Standard error": event_model.bse[variable],
        "p-value": event_model.pvalues[variable],
        "CI 95% lower": ci_low,
        "CI 95% upper": ci_high
    })

event_results = pd.DataFrame(results)

# Save coefficients for reporting
event_results.to_csv(
    output_dir / "event_study_coefficients.csv",
    index=False
)


# Plot event-study coefficients and 95% confidence intervals
# ------------------------------------------------------------

plot_labels = [
    "≤ -6", "-5", "-4", "-3", "-2",
    "-1 (ref.)", "0", "1", "2", "3", "4", "≥ 5"
]

# Insert the omitted reference period at zero
plot_results = event_results.copy()

reference_row = pd.DataFrame([{
    "Event time": "-1 (ref.)",
    "Variable": "reference",
    "Coefficient": 0.0,
    "Standard error": np.nan,
    "p-value": np.nan,
    "CI 95% lower": np.nan,
    "CI 95% upper": np.nan
}])

plot_results = pd.concat(
    [plot_results, reference_row],
    ignore_index=True
)

plot_results["order"] = plot_results["Event time"].map(
    {label: i for i, label in enumerate(plot_labels)}
)

plot_results = plot_results.sort_values("order")

x = np.arange(len(plot_results))

fig, ax = plt.subplots(figsize=(10, 6))

# Confidence intervals for estimated coefficients
estimated = plot_results["Variable"] != "reference"

ax.errorbar(
    x[estimated],
    plot_results.loc[estimated, "Coefficient"],
    yerr=[
        plot_results.loc[estimated, "Coefficient"]
        - plot_results.loc[estimated, "CI 95% lower"],
        plot_results.loc[estimated, "CI 95% upper"]
        - plot_results.loc[estimated, "Coefficient"]
    ],
    fmt="o",
    capsize=4,
    linewidth=1.3,
    label="Estimate and 95% CI"
)

# Reference period
ref_idx = plot_results.index[
    plot_results["Variable"] == "reference"
][0]
ref_x = plot_results.index.get_loc(ref_idx)

ax.scatter(
    ref_x, 0,
    marker="D",
    label="Reference period (k = -1)"
)

ax.axhline(0, linestyle="--", linewidth=1)
ax.axvline(5.5, linestyle=":", linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(plot_results["Event time"])

ax.set_xlabel("Years relative to GF introduction")
ax.set_ylabel("Change in chain market share relative to k = -1")
ax.set_title("Event Study: Noodle Market Share Around GF Introduction")

ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(
    output_dir / "event_study_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(output_dir / "event_study_coefficients.csv")
print(output_dir / "event_study_plot.png")


# Market-share trends
# --------------------------------------------------
share_data = choice_noodles.copy()

# Identify gluten-free products
share_data["GF"] = share_data["dish"].str.endswith("_gf")

# --------------------------------------------------
# Market shares by product category
# --------------------------------------------------

# Sum product shares within each university-year
campus_year = (
    share_data.groupby(["University", "year", "GF"], as_index=False)["Share"]
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
plt.close()

# Save category market shares
category_trends.to_csv(
    output_dir / "market_shares_by_category.csv",
    index=False
)

# --------------------------------------------------
# Market shares by chain
# --------------------------------------------------

# Sum dish shares within each chain-university-year
chain_campus_year = (
    share_data.groupby(["chain", "University", "year"], as_index=False)
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
plt.close()

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


# Combined market-share figure for the write-up
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for variable, label in [
    ("GF", "Gluten-free noodles"),
    ("Traditional", "Traditional noodles"),
    ("Total_Noodles", "All noodles"),
]:
    axes[0].plot(
        category_trends["year"],
        category_trends[variable],
        marker="o",
        label=label,
    )

for chain, data in chain_trends.groupby("chain"):
    axes[1].plot(
        data["year"],
        data["mean_share"],
        marker="o",
        label=chain,
    )

for ax, title in zip(
    axes,
    ["A. By product category", "B. By noodle chain"],
):
    ax.axvline(2011, linestyle="--", color="gray", alpha=0.7)
    ax.axvline(2016, linestyle="--", color="gray", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

axes[0].set_ylabel("Average market share")
fig.suptitle("Market Shares over Time")
fig.tight_layout()
fig.savefig(
    output_dir / "market_shares_overview.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Student-group heterogeneity
# --------------------------------------------------
survey = survey_noodles.copy()

# Check coding before analysis
print("Graduate categories:")
print(survey["Graduate"].value_counts(dropna=False))

print("\nAthlete categories:")
print(survey["Athlete"].value_counts(dropna=False))

# Indicator for observed gluten-free choices
survey["GF_Obs"] = survey["Pick_Obs"].str.endswith("_gf").astype(int)

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


# GF choice over time by student group
# --------------------------------------------------

# Athlete vs. non-athlete
athlete_trends = (
    survey.dropna(subset=["Athlete_D"])
    .groupby(["year", "Athlete_D"], as_index=False)
    .agg(gf_share=("GF_Obs", "mean"))
)

graduate_trends = (
    survey.dropna(subset=["Graduate_D"])
    .groupby(["year", "Graduate_D"], as_index=False)
    .agg(gf_share=("GF_Obs", "mean"))
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
panels = [
    (axes[0], athlete_trends, "Athlete_D", "Athletes", "Non-athletes", "A. By athlete status"),
    (axes[1], graduate_trends, "Graduate_D", "Graduate students", "Non-graduate students", "B. By graduate status"),
]

for ax, trends, group_col, group_one, group_zero, title in panels:
    for group_value, label in [(0, group_zero), (1, group_one)]:
        data = trends.loc[trends[group_col].eq(group_value)].sort_values("year")
        ax.plot(data["year"], data["gf_share"], marker="o", label=label)

    ax.axvline(2011, linestyle="--", color="gray", alpha=0.7)
    ax.axvline(2016, linestyle="--", color="gray", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

axes[0].set_ylabel("Share choosing GF")
fig.suptitle("Observed Gluten-Free Choice by Student Status")
fig.tight_layout()
fig.savefig(
    output_dir / "gf_choice_by_student_status.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

athlete_trends.to_csv(
    output_dir / "gf_choice_by_athlete_status.csv",
    index=False
)

graduate_trends.to_csv(
    output_dir / "gf_choice_by_graduate_status.csv",
    index=False
)


# --------------------------------------------------
# Second-choice substitution by chain and category
# --------------------------------------------------

survey = survey_noodles.copy()

survey["Obs_Category"] = np.where(
    survey["Pick_Obs"].str.endswith("_gf", na=False),
    "GF", "Traditional"
)
survey["Hyp_Category"] = np.where(
    survey["Pick_Hyp"].str.endswith("_gf", na=False),
    "GF", "Traditional"
)

valid = survey.dropna(
    subset=["Chain_Obs", "Chain_Hyp", "Pick_Obs", "Pick_Hyp"]
).copy()

valid["Observed choice"] = (
    valid["Chain_Obs"] + " - " + valid["Obs_Category"]
)

# Calculate the share choosing GF as the hypothetical second choice
summary = (
    valid.groupby("Observed choice")
    .agg(
        **{
            "Second choice: GF (%)": (
                "Hyp_Category",
                lambda x: x.eq("GF").mean() * 100
            ),
            "N": ("Hyp_Category", "size")
        }
    )
    .reset_index()
)

# Traditional share is the complement of GF share
summary["Second choice: Traditional (%)"] = (
    100 - summary["Second choice: GF (%)"]
)

summary = summary[
    [
        "Observed choice",
        "Second choice: GF (%)",
        "Second choice: Traditional (%)",
        "N"
    ]
].round(1)

print(summary.to_string(index=False))

summary.to_csv(
    output_dir / "second_choice_summary_by_chain.csv",
    index=False
)
