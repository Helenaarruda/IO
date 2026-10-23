library(readr)
library(fixest)
library(modelsummary)
library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)

# ============================================================
# 1. Load and prepare data
# ============================================================

dat <- readr::read_csv("data/input/choice_noodles.csv")

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

# Outside-option share, calculated within each University-year market

dat <- dat |>
  dplyr::mutate(
    Share0 = 1 - sum(Share),
    .by = c(University, year)
  )

# Conditional share within the nest of inside goods

dat_nl <- dat |>
  dplyr::mutate(
    Share_cond = Share / (1 - Share0),
    log_share_ratio = log(Share) - log(Share0),
    log_share_cond = log(Share_cond)
  ) |>
  dplyr::filter(
    is.finite(Share),
    is.finite(Share0),
    Share > 0,
    Share0 > 0,
    Share0 < 1,
    is.finite(Share_cond),
    Share_cond > 0,
    Share_cond <= 1,
    is.finite(log_share_ratio),
    is.finite(log_share_cond),
    is.finite(Price),
    is.finite(SpicyIndex),
    is.finite(WheatContent),
    is.finite(Veg2meat),
    is.finite(Calories),
    !is.na(dish)
  )


# ============================================================
# 2. Estimate nested-logit OLS
# ============================================================

m_ols <- fixest::feols(
  log_share_ratio ~
    Price + log_share_cond +
    SpicyIndex + WheatContent + Veg2meat + Calories |
    dish,
  data = dat_nl
)

alpha <- -unname(stats::coef(m_ols)["Price"])
rho <- unname(stats::coef(m_ols)["log_share_cond"])

if (!is.finite(alpha) || alpha <= 0) {
  stop("The estimated price coefficient must be negative for the welfare calculation.")
}

if (!is.finite(rho) || rho >= 1) {
  stop("The welfare formula requires rho < 1. Check the estimated conditional-share coefficient.")
}

if (rho < 0) {
  warning(
    "The OLS estimate of rho is negative. The calculations below use ",
    "the formula algebraically, but the estimate is outside the ",
    "standard random-utility nested-logit parameter space [0, 1)."
  )
}

message("Estimated alpha: ", round(alpha, 4))
message("Estimated rho:   ", round(rho, 4))


# ============================================================
# 3. Recover fitted mean utility
# ============================================================

# The fitted value from the regression includes rho * log_share_cond.
# Remove that term to recover the fitted mean-utility component delta.
#
# delta_j = x_j' beta + xi_j
#
# Here we use the fitted component from the OLS regression, consistent
# with the fitted-value approach in your original welfare script.

dat_nl <- dat_nl |>
  dplyr::mutate(
    delta_hat = as.numeric(
      stats::predict(m_ols, newdata = dat_nl)
    ) - rho * log_share_cond
  ) |>
  dplyr::filter(is.finite(delta_hat))


# ============================================================
# 4. Nested-logit consumer surplus
# ============================================================

# Stable calculation of log(1 + exp(x))

log1pexp <- function(x) {
  if (x > 0) {
    x + log1p(exp(-x))
  } else {
    log1p(exp(x))
  }
}

# Welfare in a market:
#
# W = (1 / alpha) *
#     log[1 + {sum_j exp(delta_j / (1-rho))}^{1-rho}]
#
# The outside option has normalized mean utility zero.

nested_logit_welfare <- function(data) {

  data |>
    dplyr::summarise(
      welfare = {
        z <- delta_hat / (1 - rho)

        # Log-sum-exp for numerical stability
        m <- max(z)
        log_sum_exp <- m + log(sum(exp(z - m)))

        log_inside_nest <- (1 - rho) * log_sum_exp

        log1pexp(log_inside_nest) / alpha
      },
      .by = c(University, year)
    )
}


# ============================================================
# 5. Welfare gains: baseline versus counterfactual
# ============================================================

welfare_gain <- function(data) {

  baseline <- nested_logit_welfare(data)

  # Keep the counterfactual rule from your original script
  counterfactual <- data |>
    dplyr::filter(WheatContent != 0) |>
    nested_logit_welfare() |>
    dplyr::rename(welfare_gf = welfare)

  dplyr::left_join(
    baseline,
    counterfactual,
    by = c("University", "year")
  ) |>
    dplyr::mutate(
      welfare_gain = welfare - welfare_gf
    )
}

# Only the OLS demand specification is used.

welfare <- welfare_gain(dat_nl)


# ============================================================
# 6. Output 1: Consumer surplus over time
# ============================================================

dat_plot <- welfare |>
  dplyr::summarise(
    dplyr::across(
      c(welfare, welfare_gf),
      \(x) mean(x, na.rm = TRUE)
    ),
    .by = year
  ) |>
  tidyr::pivot_longer(
    cols = c(welfare, welfare_gf),
    names_to = "measure",
    values_to = "welfare"
  ) |>
  dplyr::mutate(
    measure = dplyr::recode(
      measure,
      welfare = "Baseline",
      welfare_gf = "GF"
    )
  )

label_data <- dat_plot |>
  dplyr::group_by(measure) |>
  dplyr::filter(year == max(year)) |>
  dplyr::ungroup() |>
  dplyr::mutate(label = measure)

p_welfare_over_time <- ggplot2::ggplot(
  dat_plot,
  ggplot2::aes(
    x = year,
    y = welfare,
    color = measure,
    linetype = measure,
    group = measure
  )
) +
  ggplot2::geom_line(linewidth = 1.1) +
  ggplot2::geom_point(size = 1.5) +
  ggplot2::geom_text(
    data = label_data,
    ggplot2::aes(label = label),
    hjust = 0,
    nudge_x = 1.5,
    size = 3.5,
    show.legend = FALSE
  ) +
  ggplot2::scale_color_manual(
    values = c(
      "Baseline" = "#4477AA",
      "GF" = "#CC6677"
    )
  ) +
  ggplot2::scale_linetype_manual(
    values = c(
      "Baseline" = "solid",
      "GF" = "longdash"
    )
  ) +
  ggplot2::scale_x_continuous(
    limits = c(min(dat_plot$year), max(dat_plot$year) + 4),
    breaks = scales::pretty_breaks(n = 8)
  ) +
  ggplot2::labs(
    title = "Consumer Surplus Over Time",
    x = NULL,
    y = "Welfare per consumer"
  ) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 15),
    panel.grid.minor = ggplot2::element_blank(),
    panel.grid.major.x = ggplot2::element_blank(),
    axis.text = ggplot2::element_text(color = "grey25"),
    legend.position = "none",
    plot.margin = ggplot2::margin(5.5, 35, 5.5, 5.5)
  )

print(p_welfare_over_time)

ggplot2::ggsave(
  "output/welfare_over_time_nested.pdf",
  plot = p_welfare_over_time,
  width = 9,
  height = 5,
  units = "in"
)


# ============================================================
# 7. Output 2: Welfare gains over time
# ============================================================

dat_gain_plot <- welfare |>
  dplyr::summarise(
    welfare_gain = mean(welfare_gain, na.rm = TRUE),
    .by = year
  )

gain_labels <- dat_gain_plot |>
  dplyr::filter(year == max(year)) |>
  dplyr::mutate(label = "OLS")

p_welfare_gain <- ggplot2::ggplot(
  dat_gain_plot,
  ggplot2::aes(
    x = year,
    y = welfare_gain
  )
) +
  ggplot2::geom_hline(
    yintercept = 0,
    color = "grey75",
    linewidth = 0.5
  ) +
  ggplot2::geom_line(
    color = "#4477AA",
    linewidth = 1.1
  ) +
  ggplot2::geom_point(
    color = "#4477AA",
    size = 1.6
  ) +
  ggplot2::geom_text(
    data = gain_labels,
    ggplot2::aes(label = label),
    color = "#4477AA",
    nudge_x = 1.2,
    hjust = 0,
    size = 3.8,
    fontface = "bold",
    show.legend = FALSE
  ) +
  ggplot2::scale_x_continuous(
    limits = c(min(dat_gain_plot$year), max(dat_gain_plot$year) + 3),
    breaks = scales::pretty_breaks(n = 8),
    expand = ggplot2::expansion(mult = c(0.01, 0))
  ) +
  ggplot2::labs(
    title = "Welfare Gains vs. Counterfactual Over Time",
    x = NULL,
    y = "Welfare gain per consumer"
  ) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 15),
    plot.title.position = "plot",
    panel.grid.minor = ggplot2::element_blank(),
    panel.grid.major.x = ggplot2::element_blank(),
    axis.text = ggplot2::element_text(color = "grey25"),
    axis.title.y = ggplot2::element_text(
      margin = ggplot2::margin(r = 10)
    ),
    legend.position = "none",
    plot.margin = ggplot2::margin(8, 30, 8, 8)
  )

print(p_welfare_gain)

ggplot2::ggsave(
  "output/welfare_gain_over_time_nested.pdf",
  plot = p_welfare_gain,
  width = 9,
  height = 5,
  units = "in"
)


# ============================================================
# 8. Output 3: Regress welfare gains on student composition
# ============================================================

welfare_reg <- welfare |>
  dplyr::left_join(
    college_year,
    by = c("University" = "campus", "year")
  )

# 2011–2015

reg_ols_mid <- stats::lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = dplyr::filter(
    welfare_reg,
    year >= 2011,
    year <= 2015
  )
)

# 2016 onward

reg_ols_post <- stats::lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = dplyr::filter(
    welfare_reg,
    year >= 2016
  )
)

modelsummary::modelsummary(
  list(
    "2011--2015" = reg_ols_mid,
    "2016 onward" = reg_ols_post
  ),
  coef_map = c(
    "(Intercept)" = "Constant",
    "Share_athletes" = "Share of Athletes",
    "Share_Graduates" = "Share of Graduates"
  ),
  estimate = "{estimate}{stars}",
  statistic = "({std.error})",
  stars = c(
    "*" = 0.10,
    "**" = 0.05,
    "***" = 0.01
  ),
  gof_omit = ".*",
  title = "Welfare Gains and Student Composition",
  notes = paste(
    "Dependent variable: nested-logit welfare gain.",
    "OLS demand estimates are used to construct welfare.",
    "Standard errors in parentheses.",
    "* p < 0.10, ** p < 0.05, *** p < 0.01."
  ),
  output = "output/welfare_gain_by_period_nested.tex"
)


# ============================================================
# 9. Dish-level nested-logit elasticity matrix
# ============================================================

# Each matrix describes the elasticity of the aggregate share of
# dish j with respect to a proportional price change for all
# chain-dish alternatives belonging to dish k.
#
# Product-level nested-logit elasticity:
#
# eps_jk = alpha * p_k *
#   [s_k + rho/(1-rho) * s_{k|1}
#    - 1/(1-rho) * I(j = k)]
#
# We aggregate across alternatives with the same dish label.
# At rho = 0, this reduces to the logit aggregation used in your
# original elasticity_matrix().

elasticity_matrix <- function(alpha, rho, data) {

  if (!is.finite(alpha) || alpha <= 0) {
    stop("The absolute price coefficient must be positive.")
  }

  if (!is.finite(rho) || rho >= 1) {
    stop("The elasticity formula requires rho < 1.")
  }

  dish_market <- data |>
    dplyr::summarise(
      share_dish = sum(Share, na.rm = TRUE),
      price_share = sum(Price * Share, na.rm = TRUE),
      inside_share = 1 - dplyr::first(Share0),
      .by = c(University, year, dish)
    ) |>
    dplyr::filter(
      is.finite(share_dish),
      share_dish > 0,
      is.finite(price_share),
      is.finite(inside_share),
      inside_share > 0
    )

  dish_j <- dish_market |>
    dplyr::transmute(
      University,
      year,
      dish_j = as.character(dish),
      share_j = share_dish,
      inside_share
    )

  dish_k <- dish_market |>
    dplyr::transmute(
      University,
      year,
      dish_k = as.character(dish),
      price_share_k = price_share
    )

  dplyr::inner_join(
    dish_j,
    dish_k,
    by = c("University", "year"),
    relationship = "many-to-many"
  ) |>
    dplyr::mutate(
      elasticity =
        alpha * price_share_k *
          (
            1 + rho / ((1 - rho) * inside_share)
          ) -
          dplyr::if_else(
            dish_j == dish_k,
            alpha * price_share_k /
              ((1 - rho) * share_j),
            0
          )
    ) |>
    dplyr::summarise(
      elasticity = mean(elasticity, na.rm = TRUE),
      .by = c(dish_j, dish_k)
    )
}


# ============================================================
# 10. Define the three time periods
# ============================================================

periods <- list(
  "Before 2011" = dplyr::filter(dat_nl, year < 2011),
  "2011–2015" = dplyr::filter(dat_nl, year >= 2011, year <= 2015),
  "2016 onward" = dplyr::filter(dat_nl, year >= 2016)
)


# ============================================================
# 11. Calculate the three OLS elasticity matrices
# ============================================================

elast_all <- lapply(names(periods), function(period_name) {

  d <- periods[[period_name]]

  elasticity_matrix(alpha, rho, d) |>
    dplyr::mutate(period = period_name)

}) |>
  dplyr::bind_rows()


# ============================================================
# 12. Prepare the three matrices for one horizontal graph
# ============================================================

dish_order <- sort(unique(as.character(dat_nl$dish)))
dish_count <- length(dish_order)

# Complete the grid so each period has the same dish ordering,
# including blank cells for dish pairs not observed in a period.

elast_all <- elast_all |>
  tidyr::complete(
    period = names(periods),
    dish_j = dish_order,
    dish_k = dish_order
  ) |>
  dplyr::mutate(
    period = factor(period, levels = names(periods)),
    dish_j = factor(dish_j, levels = rev(dish_order)),
    dish_k = factor(dish_k, levels = dish_order)
  )

max_abs <- max(abs(elast_all$elasticity), na.rm = TRUE)

if (!is.finite(max_abs) || max_abs == 0) {
  max_abs <- 1
}

elast_all <- elast_all |>
  dplyr::mutate(
    # One decimal; omit values that round to zero
    label = dplyr::if_else(
      is.na(elasticity) | round(elasticity, 1) == 0,
      "",
      sprintf("%.1f", elasticity),
      missing = ""
    ),

    # Use white labels for darker cells
    label_color = dplyr::if_else(
      abs(elasticity) >= 0.55 * max_abs,
      "white",
      "#263238",
      missing = "#263238"
    )
  )

label_size <- if (dish_count <= 8) {
  3.0
} else if (dish_count <= 12) {
  2.5
} else {
  2.0
}


# ============================================================
# 13. Output 4: Three elasticity matrices in one horizontal graph
# ============================================================

p_elasticities <- ggplot2::ggplot(
  elast_all,
  ggplot2::aes(
    x = dish_k,
    y = dish_j,
    fill = elasticity
  )
) +
  ggplot2::geom_tile(
    color = "white",
    linewidth = 0.3
  ) +

  # Numeric labels only for nonzero rounded elasticities
  ggplot2::geom_text(
    ggplot2::aes(
      label = label,
      color = label_color
    ),
    size = label_size,
    show.legend = FALSE
  ) +

  # Outline own-price elasticity cells
  ggplot2::geom_tile(
    data = dplyr::filter(
      elast_all,
      as.character(dish_j) == as.character(dish_k)
    ),
    ggplot2::aes(
      x = dish_k,
      y = dish_j
    ),
    inherit.aes = FALSE,
    fill = NA,
    color = "grey35",
    linewidth = 0.4
  ) +

  # Three time-period matrices placed horizontally
  ggplot2::facet_wrap(
    ~period,
    nrow = 1
  ) +

  ggplot2::scale_fill_gradient2(
    low = "#B6534B",
    mid = "#F7F5F0",
    high = "#287A8A",
    midpoint = 0,
    limits = c(-max_abs, max_abs),
    oob = scales::squish,
    na.value = "white",
    name = "Elasticity",
    labels = scales::label_number(accuracy = 0.1)
  ) +

  ggplot2::scale_color_identity() +
  ggplot2::scale_x_discrete(drop = FALSE) +
  ggplot2::scale_y_discrete(drop = FALSE) +
  ggplot2::coord_fixed() +

  ggplot2::labs(
    title = "Nested-Logit Price Elasticities by Dish",
    subtitle = "OLS estimates; average across University–year markets",
    x = "Price of dish k",
    y = "Demand for dish j",
    caption = paste(
      "Diagonal cells: own-price elasticities.",
      "Off-diagonal cells: cross-price elasticities.",
      "A proportional price change is applied to the alternatives",
      "belonging to dish k.",
      "Labels rounded to one decimal; rounded zeros omitted."
    )
  ) +

  ggplot2::theme_minimal(base_size = 11) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(
      face = "bold",
      size = 15
    ),
    plot.title.position = "plot",
    plot.subtitle = ggplot2::element_text(color = "grey40"),
    plot.caption = ggplot2::element_text(
      color = "grey40",
      size = 8.5,
      hjust = 0
    ),
    strip.text = ggplot2::element_text(
      face = "bold",
      size = 11
    ),
    strip.background = ggplot2::element_rect(
      fill = "#EEF1F3",
      color = NA
    ),
    panel.grid = ggplot2::element_blank(),
    panel.spacing = grid::unit(1, "lines"),
    axis.text.x = ggplot2::element_text(
      angle = 45,
      hjust = 1,
      size = 8
    ),
    axis.text.y = ggplot2::element_text(size = 8),
    axis.title = ggplot2::element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = ggplot2::element_text(face = "bold"),
    legend.key.width = grid::unit(2, "cm"),
    plot.margin = ggplot2::margin(10, 12, 10, 10)
  )

print(p_elasticities)

ggplot2::ggsave(
  "output/elasticities_nested_logit_ols.pdf",
  plot = p_elasticities,
  width = max(15, 1.15 * dish_count + 7),
  height = max(6.5, 0.45 * dish_count + 2.5),
  units = "in"
)