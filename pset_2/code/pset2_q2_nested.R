library(readr)
library(fixest)
library(readxl)
library(dplyr)
library(ggplot2)

# ============================================================
# 1. Load data and construct the outside option
# ============================================================

dat <- readr::read_csv("pset_2/data/input/choice_noodles.csv")

dat <- dat |>
  dplyr::mutate(
    Share0 = 1 - sum(Share),
    .by = c(University, year)
  )


# ============================================================
# 2. Load college-year data and merge
# ============================================================

college_year <- readxl::read_xls("pset_2/data/input/college-year.xls") |>
  dplyr::mutate(
    dplyr::across(year, as.numeric)
  )

# Some data cleaning has to be done

college_year <- college_year |>
  dplyr::mutate(
    campus = dplyr::case_match(
      campus,
      "St Johns" ~ "Saint John",
      "Texas-A&M" ~ "Texas A&M",
      "R-H" ~ "Rose-Hul.",
      "U. Chicago" ~ "Univ. Chicago",
      .default = campus
    )
  )

# Merge the datasets

dat <- dat |>
  dplyr::left_join(
    college_year,
    by = c("University" = "campus", "year")
  )


# ============================================================
# 3. Construct conditional shares, nest size, and instruments
# ============================================================

nest_size <- dat |>
  dplyr::distinct(University, year, chain, dish) |>
  dplyr::count(
    University,
    year,
    name = "n_inside"
  ) |>
  dplyr::mutate(
    n_other_nest = n_inside - 1L
  )

dat_nl <- dat |>
  dplyr::left_join(
    nest_size,
    by = c("University", "year")
  ) |>
  dplyr::mutate(
    # Conditional share within the inside-goods nest
    Share_cond = Share / (1 - Share0),

    # Regression variables
    log_share_ratio = log(Share) - log(Share0),
    log_share_cond = log(Share_cond),

    # Additional characteristic × cost-shifter instruments
    IV_wheat = WheatContent * WheatPIDX,
    IV_meat = Veg2meat * MeatPIDX,
    IV_veg = Veg2meat * VegPIDX
  )

# Keep a common estimation sample across all four specifications.
# This also removes observations for which the share logarithms
# or any of the instruments are undefined.

required_vars <- c(
  "Share", "Share0", "Share_cond",
  "Price", "SpicyIndex", "WheatContent", "Veg2meat", "Calories",
  "VegPIDX", "WheatPIDX", "MeatPIDX", "WageIDX", "ReEsIDX",
  "IV_wheat", "IV_meat", "IV_veg", "n_other_nest"
)

dat_nl <- dat_nl |>
  dplyr::filter(
    Share > 0,
    Share0 > 0,
    Share0 < 1,
    Share_cond > 0,
    Share_cond <= 1,
    dplyr::if_all(
      dplyr::all_of(required_vars),
      ~ is.finite(.x)
    )
  )


# ============================================================
# 4. Estimate the four specifications
# ============================================================

# (1) OLS:
# Price and conditional share are treated as exogenous.

m_ols <- fixest::feols(
  log_share_ratio ~
    Price + log_share_cond +
    SpicyIndex + WheatContent + Veg2meat + Calories |
    dish,
  data = dat_nl
)


# Instruments used in every IV specification:
# existing cost shifters.

cost_shifters <- c(
  "VegPIDX",
  "WheatPIDX",
  "MeatPIDX",
  "WageIDX",
  "ReEsIDX"
)


# (2) IV:
# Both price and conditional share are endogenous.
# Use only the original cost-shifter instruments.

m_iv_cost <- fixest::feols(
  log_share_ratio ~
    SpicyIndex + WheatContent + Veg2meat + Calories |
    dish |
    Price + log_share_cond ~
      VegPIDX + WheatPIDX + MeatPIDX + WageIDX + ReEsIDX,
  data = dat_nl
)


# (3) IV:
# Instrument both endogenous regressors with cost shifters
# plus the three characteristic × cost-shifter interactions.

m_iv_interactions <- fixest::feols(
  log_share_ratio ~
    SpicyIndex + WheatContent + Veg2meat + Calories |
    dish |
    Price + log_share_cond ~
      VegPIDX + WheatPIDX + MeatPIDX + WageIDX + ReEsIDX +
      IV_wheat + IV_meat + IV_veg,
  data = dat_nl
)


# (4) IV:
# Add nest size to the full instrument set.
# This is the preferred specification.

m_iv_full <- fixest::feols(
  log_share_ratio ~
    SpicyIndex + WheatContent + Veg2meat + Calories |
    dish |
    Price + log_share_cond ~
      VegPIDX + WheatPIDX + MeatPIDX + WageIDX + ReEsIDX +
      IV_wheat + IV_meat + IV_veg +
      n_other_nest,
  data = dat_nl
)


# ============================================================
# 5. Combined regression table
# ============================================================

fixest::etable(
  m_ols,
  m_iv_cost,
  m_iv_interactions,
  m_iv_full,

  # Group headers: OLS over column (1), IV over columns (2)-(4)
  headers = list(
    "^" = list(
      "OLS" = 1,
      "IV" = 3
    )
  ),

  # Professional coefficient labels
  dict = c(
    "reset",
    "Price" = "Price",
    "fit_Price" = "Price",
    "log_share_cond" = "$\\rho$",
    "fit_log_share_cond" = "$\\rho$",
    "SpicyIndex" = "Spicy index",
    "WheatContent" = "Wheat content",
    "Veg2meat" = "Vegetable-to-meat",
    "Calories" = "Calories"
  ),

  # Specification indicators
  extralines = list(
    "Cost-shifter IVs" = c(
      "No", "Yes", "Yes", "Yes"
    ),
    "Interaction IVs" = c(
      "No", "No", "Yes", "Yes"
    ),
    "Nest-size IV" = c(
      "No", "No", "No", "Yes"
    )
  ),

  # Formatting and output
  tex = TRUE,
  file = "output/nested_logit.tex",
  replace = TRUE,
  create_dirs = TRUE,
  caption = "Nested Logit Demand Estimates",
  label = "tab:nested-logit",
  depvar = FALSE,
  digits = 3,
  se.below = TRUE,
  signif.code = c(
    "***" = 0.01,
    "**" = 0.05,
    "*" = 0.10
  ),
  fitstat = ~ n + r2,
  placement = "H"
)

### Welfare