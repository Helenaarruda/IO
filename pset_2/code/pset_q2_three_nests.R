library(readr)
library(fixest)
library(readxl)
library(dplyr)
library(ggplot2)

# ============================================================
# 1. Load data and construct the outside option
# ============================================================

dat <- readr::read_csv("data/input/choice_noodles.csv")

dat <- dat |>
  dplyr::mutate(
    Share0 = 1 - sum(Share),
    .by = c(University, year)
  )


# ============================================================
# 2. Load college-year data and merge
# ============================================================

college_year <- readxl::read_xls("data/input/college-year.xls") |>
  dplyr::mutate(
    dplyr::across(year, as.numeric)
  ) |>
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

dat <- dat |>
  dplyr::left_join(
    college_year,
    by = c("University" = "campus", "year")
  )


# ============================================================
# 3. Define the three nests
# ============================================================

# Nest 0: outside option (singleton, not included as a product row)
# Nest 1: gluten-free options
# Nest 2: all remaining inside options

gf_dishes <- c("YN_gf", "AN_gf", "BN_gf")

dat <- dat |>
  dplyr::mutate(
    nest = dplyr::if_else(
      dish %in% gf_dishes,
      "Gluten-free",
      "Other"
    )
  )


# ============================================================
# 4. Construct conditional shares and nest-size instrument
# ============================================================

# Nest market share, calculated within University-year-nest.

dat_nl <- dat |>
  dplyr::group_by(University, year, nest) |>
  dplyr::mutate(
    Share_nest = sum(Share, na.rm = TRUE),
    Share_cond = Share / Share_nest
  ) |>
  dplyr::ungroup() |>
  dplyr::mutate(
    log_share_ratio = log(Share) - log(Share0),
    log_share_cond = log(Share_cond)
  )

# Count the number of other chain-dish alternatives in the same nest.

nest_size <- dat_nl |>
  dplyr::distinct(University, year, nest, chain, dish) |>
  dplyr::count(
    University, year, nest,
    name = "n_nest"
  ) |>
  dplyr::mutate(
    n_other_nest = n_nest - 1L
  )

dat_nl <- dat_nl |>
  dplyr::left_join(
    nest_size,
    by = c("University", "year", "nest")
  ) |>
  dplyr::mutate(
    # Product-characteristic × input-cost interactions
    IV_wheat = WheatContent * WheatPIDX,
    IV_meat = Veg2meat * MeatPIDX,
    IV_veg = Veg2meat * VegPIDX
  )


# ============================================================
# 5. Prepare the estimation sample
# ============================================================

required_vars <- c(
  "Share", "Share0", "Share_nest", "Share_cond",
  "Price", "SpicyIndex", "WheatContent", "Veg2meat", "Calories",
  "VegPIDX", "WheatPIDX", "MeatPIDX", "WageIDX", "ReEsIDX",
  "IV_wheat", "IV_meat", "IV_veg", "n_other_nest"
)

dat_nl <- dat_nl |>
  dplyr::filter(
    Share > 0,
    Share0 > 0,
    Share0 < 1,
    Share_nest > 0,
    Share_cond > 0,
    Share_cond <= 1,
    dplyr::if_all(
      dplyr::all_of(required_vars),
      ~ is.finite(.x)
    )
  )


# ============================================================
# 6. Estimate nested logit with one common rho
# ============================================================

# The single log_share_cond regressor is defined relative to
# each product's own nest. Its coefficient is the common rho.
#
# Both Price and log_share_cond are endogenous.
# All cost-shifter, interaction, and nest-size instruments are used.

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
# 7. Regression table
# ============================================================

fixest::etable(
  m_iv_full,

  dict = c(
    "Price" = "Price",
    "fit_Price" = "Price",
    "log_share_cond" = "$\\rho$",
    "fit_log_share_cond" = "$\\rho$",
    "SpicyIndex" = "Spicy index",
    "WheatContent" = "Wheat content",
    "Veg2meat" = "Vegetable-to-meat",
    "Calories" = "Calories"
  ),

  extralines = list(
    "Cost-shifter instruments" = "Yes",
    "Interaction instruments" = "Yes",
    "Nest-size instrument" = "Yes"
  ),

  tex = TRUE,
  file = "output/nested_logit_3nests.tex",
  replace = TRUE,
  create_dirs = TRUE,
  caption = "Nested Logit Demand: Three Nests",
  label = "tab:nested-logit-three-nests",
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


# ============================================================
# 8. Report the estimated price and nesting coefficients
# ============================================================

coefs <- stats::coef(m_iv_full)

rho_name <- intersect(
  c("fit_log_share_cond", "log_share_cond"),
  names(coefs)
)

price_name <- intersect(
  c("fit_Price", "Price"),
  names(coefs)
)

if (length(rho_name) == 0 || length(price_name) == 0) {
  stop(
    "Could not identify the price and conditional-share coefficients. ",
    "Inspect names(coef(m_iv_full))."
  )
}

alpha_hat <- -unname(coefs[price_name[1]])
rho_hat <- unname(coefs[rho_name[1]])

cat("Estimated alpha:", alpha_hat, "\n")
cat("Estimated common rho:", rho_hat, "\n")

if (rho_hat < 0 || rho_hat >= 1) {
  warning(
    "The estimated common rho is outside the standard nested-logit ",
    "range [0, 1). Interpret the structural model accordingly."
  )
}