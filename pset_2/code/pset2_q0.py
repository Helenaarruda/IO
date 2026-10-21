import jax
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
from scipy.optimize import root
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# ==============================================================================
# Part 0 - Input
# ==============================================================================
#INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "input"
INPUT_DIR = Path.cwd() / "pset_2" / "data" / "input"


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



# ==============================================================================
# Part I - Exhibit 1: Adoption of gluten-free noodle options over time
# ==============================================================================

# 1. Inspect the available dishes and chains before classifying gluten-free options
print("Available dishes:")
print(sorted(choice_noodles["dish"].dropna().unique()))

print("\nAvailable chains:")
print(sorted(choice_noodles["chain"].dropna().unique()))

print("\nYears in the data:")
print(sorted(choice_noodles["year"].dropna().unique()))

# 2. Inspect the market-share variable
print("\nShare summary:")
print(choice_noodles["Share"].describe())

# After inspecting the dish names, define which dishes are gluten-free.
# Replace the example below with the exact labels in your dataset.
GLUTEN_FREE_DISHES = [
    "AN_gf",
    "BN_gf",
    "YN_gf",
]

if not GLUTEN_FREE_DISHES:
    print(
        "\nDefine GLUTEN_FREE_DISHES using the actual dish labels "
        "before running the exhibit."
    )
else:
    # 3. Create a gluten-free indicator
    df = choice_noodles.copy()
    df["GlutenFree"] = df["dish"].isin(GLUTEN_FREE_DISHES)

    # 4. Calculate total gluten-free and traditional noodle shares
    # within each campus-year market.
    # Share is defined in the assignment as quantity / students eating lunch.
    campus_year = (
        df.groupby(["University", "year", "GlutenFree"], as_index=False)
          ["Share"].sum()
    )

    # Average across campuses, giving each campus equal weight
    annual_shares = (
        campus_year.groupby(["year", "GlutenFree"])["Share"]
        .mean()
        .unstack(fill_value=0)
        .rename(columns={
            False: "Traditional noodles",
            True: "Gluten-free noodles"
        })
    )

  # 5. Plot the evolution over time
fig, ax = plt.subplots(figsize=(10, 6))

# Plot market shares
ax.plot(
    annual_shares.index,
    annual_shares["Traditional noodles"],
    marker="o",
    markersize=5,
    linewidth=2,
    label="Traditional noodles"
)

ax.plot(
    annual_shares.index,
    annual_shares["Gluten-free noodles"],
    marker="o",
    markersize=5,
    linewidth=2,
    label="Gluten-free noodles"
)

# Mark the two important events
ax.axvline(
    x=2010,
    color="gray",
    linestyle="--",
    linewidth=1.4,
    alpha=0.9
)

ax.axvline(
    x=2016,
    color="gray",
    linestyle=":",
    linewidth=1.6,
    alpha=0.9
)

# Add event annotations
ax.text(
    2010.2, 0.97,
    "First GF introduction",
    transform=ax.get_xaxis_transform(),
    ha="left",
    va="top",
    fontsize=9,
    color="dimgray"
)

ax.text(
    2016.2, 0.82,
    "Expansion to other chains",
    transform=ax.get_xaxis_transform(),
    ha="left",
    va="top",
    fontsize=9,
    color="dimgray"
)

# Titles and labels
ax.set_title(
    "Noodle Market Shares Before and After GF Introduction",
    fontsize=13,
    pad=14
)
ax.set_xlabel("Year")
ax.set_ylabel("Average share of students eating lunch")

# Use integer years on the x-axis
years = annual_shares.index.astype(int)
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45, ha="right")

# Set sensible axis limits
ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
ax.set_ylim(bottom=0)

# Formatting
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, loc="best")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# Save figure
plt.savefig(
    "pset_2/data/output/evolution_shares.png",
    dpi=300,
    bbox_inches="tight"
)


# ==============================================================================
# Exhibit 2 Evolution of vote shares by student type
# ==============================================================================


# 1. Prepare survey data
survey = survey_noodles.copy()
survey = survey.loc[survey["Pick_Obs"].notna()].copy()

# Define student groups
survey["Athlete_group"] = np.where(
    survey["Athlete"].eq("Y"),
    "Athletes",
    "Non-athletes"
)

survey["Graduate_group"] = np.where(
    survey["Graduate"].eq("Y"),
    "Graduate students",
    "Undergraduates"
)

# Identify observed GF choices
survey["GlutenFree"] = (
    survey["Pick_Obs"].astype(str).str.lower().str.endswith("_gf")
)

# 2. Calculate GF choice shares by year and student type
athlete_trends = (
    survey.groupby(["year", "Athlete_group"], as_index=False)
    .agg(
        gf_share=("GlutenFree", "mean"),
        n=("GlutenFree", "size")
    )
)

graduate_trends = (
    survey.groupby(["year", "Graduate_group"], as_index=False)
    .agg(
        gf_share=("GlutenFree", "mean"),
        n=("GlutenFree", "size")
    )
)

# 3. Plot trends
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

panels = [
    (
        axes[0],
        athlete_trends,
        "Athlete_group",
        ["Non-athletes", "Athletes"],
        "A. By athlete status"
    ),
    (
        axes[1],
        graduate_trends,
        "Graduate_group",
        ["Undergraduates", "Graduate students"],
        "B. By graduate status"
    )
]

for ax, trends, group_col, groups, title in panels:
    for group in groups:
        group_data = (
            trends.loc[trends[group_col].eq(group)]
            .sort_values("year")
        )

        ax.plot(
            group_data["year"],
            group_data["gf_share"],
            marker="o",
            markersize=4,
            linewidth=2,
            label=group
        )

    # Mark market events
    ax.axvline(
        2010,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8
    )

    ax.axvline(
        2016,
        color="gray",
        linestyle=":",
        linewidth=1.4,
        alpha=0.8
    )

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_xticks(sorted(survey["year"].unique()))
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Share choosing a gluten-free noodle dish")

fig.suptitle(
    "Gluten-Free Choices over Time by Student Type",
    fontsize=13
)

plt.tight_layout()
plt.show()

# Save figure
plt.savefig(
    "pset_2/data/output/choices_bystudent.png",
    dpi=300,
    bbox_inches="tight"
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

# Save a LaTeX table that can be included in the write-up
latex_table = table[["Variable", "Coefficient", "Robust SE (HC1)", "p-value"]].copy()
latex_table["Coefficient"] = latex_table["Coefficient"].map(lambda x: f"{x:.4f}")
latex_table["Robust SE (HC1)"] = latex_table["Robust SE (HC1)"].map(lambda x: f"{x:.4f}")
latex_table["p-value"] = latex_table["p-value"].map(lambda x: f"{x:.4f}")
latex = latex_table.to_latex(
    index=False,
    escape=True,
    caption="Pricing Regression 1: Price on Product Characteristics",
    label="tab:pricing-regression-1",
    column_format="lrrr",
)
("pset_2/data/output/pricing_regression_1.tex").write_text(latex, encoding="utf-8")


print(model.summary())
