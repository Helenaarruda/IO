library(readr)
library(fixest)
library(modelsummary)
library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)

dir.create("pset_2/output", showWarnings = FALSE, recursive = TRUE)

dat <- readr::read_csv("pset_2/data/input/choice_noodles.csv")

# constructing outside option

dat <- dat |>
  dplyr::mutate(
    Share0 = 1 - sum(Share),
    .by = c(University, year)
  )


# ============================================================
# OLS
# ============================================================

reg <- fixest::feols(
  log(Share) - log(Share0) ~
    Price + SpicyIndex + WheatContent + Veg2meat + Calories |
    dish,
  data = dat
)


# ============================================================
# IV
# ============================================================

college_year <- readxl::read_xls("pset_2/data/input/college-year.xls") |>
  dplyr::mutate(
    dplyr::across(year, as.numeric)
  )

# some data cleaning has to be done

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

# merging the datasets

dat <- dat |>
  dplyr::left_join(
    college_year,
    by = c("University" = "campus", "year")
  )

# Original IV specification: instrument Price using the existing indices

reg_iv_original <- fixest::feols(
  log(Share) - log(Share0) ~
    SpicyIndex + WheatContent + Veg2meat + Calories |
    dish |
    Price ~ VegPIDX + WheatPIDX + MeatPIDX + WageIDX + ReEsIDX,
  data = dat
)

# Construct three additional product-characteristic × input-price instruments.
#
# Wheat: WheatContent × WheatPIDX
# Meat:  Veg2meat × MeatPIDX
# Veg:   Veg2meat × VegPIDX
#
# The supplied script has Veg2meat rather than separate MeatContent
# and VegContent variables, so it is used for the meat and veg interactions.

dat <- dat |>
  dplyr::mutate(
    IV_wheat = WheatContent * WheatPIDX,
    IV_meat = Veg2meat * MeatPIDX,
    IV_veg = Veg2meat * VegPIDX
  )

# Fuller IV specification: add the three interaction instruments
# to the original set of price instruments.

reg_iv_full <- fixest::feols(
  log(Share) - log(Share0) ~
    SpicyIndex + WheatContent + Veg2meat + Calories |
    dish |
    Price ~
      VegPIDX + WheatPIDX + MeatPIDX + WageIDX + ReEsIDX +
      IV_wheat + IV_meat + IV_veg,
  data = dat
)

# The fuller specification is the preferred IV model for all
# subsequent welfare and elasticity calculations.

reg_iv <- reg_iv_full


# ============================================================
# Combined regression table
# ============================================================

fixest::etable(
  reg,
  reg_iv_original,
  reg_iv_full,

  # Group the first column under OLS and the other two under IV
  headers = list(
    "^" = list(
      "OLS" = 1,
      "IV" = 2
    )
  ),

  # Coefficient labels
  dict = c(
    "Price" = "Price",
    "fit_Price" = "Price",
    "SpicyIndex" = "Spicy",
    "WheatContent" = "Wheat",
    "Veg2meat" = "Veg2Meat",
    "Calories" = "Calories"
  ),

  # Specification indicators
  extralines = list(
    "Cost-shifter IVs" = c("No", "Yes", "Yes"),
    "Interaction IVs" = c("No", "No", "Yes")
  ),

  # Professional table formatting
  caption = "Demand Estimates",
  label = "tab:demand",
  tex = TRUE,
  file = "pset_2/output/part2_table.tex",
  replace = TRUE,
  create_dirs = TRUE,
  digits = 3,
  se.below = TRUE,
  depvar = FALSE,
  fitstat = ~ n + r2 + wr2,
  signif.code = c(
    "***" = 0.01,
    "**" = 0.05,
    "*" = 0.10
  ),
  placement = "H"
)


# ============================================================
# Welfare
# ============================================================

logit_welfare <- function(dat, reg) {
  dat |>
    dplyr::mutate(
      delta = predict(reg, newdata = dat)
    ) |>
    dplyr::summarise(
      welfare = log1p(sum(exp(delta))) / -coef(reg)[1],
      .by = c(University, year)
    )
}

# function that computes differences in welfare with and without choices

welfare_gain <- function(dat, reg) {
  gluten <- dat |>
    logit_welfare(reg)

  gluten_free <- dat |>
    dplyr::filter(WheatContent != 0) |>
    logit_welfare(reg) |>
    dplyr::rename(welfare_gf = welfare)

  # merging the two and taking diffs
  dplyr::left_join(
    gluten,
    gluten_free,
    by = c("University", "year")
  ) |>
    dplyr::mutate(
      welfare_gain = welfare - welfare_gf
    )
}

# stacking welfare gains for OLS and the preferred IV estimate

welfare <- purrr::map2(
  list(reg, reg_iv),
  c("OLS", "IV"),
  function(model, name) {
    welfare_gain(dat, model) |>
      dplyr::mutate(estimator = name)
  }
) |>
  dplyr::bind_rows()


# ============================================================
# Diff-in-diff style graph
# ============================================================

dat_plot <- welfare |>
  dplyr::summarise(
    dplyr::across(
      c(welfare, welfare_gf),
      \(x) mean(x, na.rm = TRUE)
    ),
    .by = c(year, estimator)
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
  dplyr::group_by(estimator, measure) |>
  dplyr::filter(year == max(year)) |>
  dplyr::ungroup() |>
  dplyr::mutate(
    label = ifelse(
      measure == "GF",
      paste0(estimator, " (GF)"),
      estimator
    )
  )

ggplot2::ggplot(
  dat_plot,
  ggplot2::aes(
    x = year,
    y = welfare,
    color = estimator,
    linetype = measure,
    group = interaction(estimator, measure)
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
    values = c("OLS" = "#4477AA", "IV" = "#CC6677")
  ) +
  ggplot2::scale_linetype_manual(
    values = c("Baseline" = "solid", "GF" = "longdash")
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

ggplot2::ggsave(
  "output/welfare_over_time.pdf",
  width = 9,
  height = 5,
  units = "in"
)


# ============================================================
# Graph of differences
# ============================================================

dat_plot <- welfare |>
  dplyr::summarise(
    welfare_gain = mean(welfare_gain, na.rm = TRUE),
    .by = c(year, estimator)
  )

gain_labels <- dat_plot |>
  dplyr::filter(year == max(dat_plot$year)) |>
  dplyr::mutate(label = estimator)

ggplot2::ggplot(
  dat_plot,
  ggplot2::aes(
    x = year,
    y = welfare_gain,
    color = estimator,
    group = estimator
  )
) +
  ggplot2::geom_hline(
    yintercept = 0,
    color = "grey75",
    linewidth = 0.5
  ) +
  ggplot2::geom_line(linewidth = 1.1) +
  ggplot2::geom_point(size = 1.6) +
  ggplot2::geom_text(
    data = gain_labels,
    ggplot2::aes(label = label),
    nudge_x = 1.2,
    hjust = 0,
    size = 3.8,
    fontface = "bold",
    show.legend = FALSE
  ) +
  ggplot2::scale_color_manual(
    values = c("OLS" = "#4477AA", "IV" = "#CC6677")
  ) +
  ggplot2::scale_x_continuous(
    limits = c(min(dat_plot$year), max(dat_plot$year) + 3),
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

ggplot2::ggsave(
  "output/welfare_gain_over_time.pdf",
  width = 9,
  height = 5,
  units = "in"
)


# ============================================================
# Regressing welfare gains
# ============================================================

welfare <- welfare |>
  dplyr::left_join(
    college_year,
    by = c("University" = "campus", "year")
  )

# Before 2011

reg_ols_pre <- stats::lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = dplyr::filter(
    welfare,
    estimator == "OLS",
    year < 2011
  )
)

reg_iv_pre <- stats::lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = dplyr::filter(
    welfare,
    estimator == "IV",
    year < 2011
  )
)

# 2011–2015

reg_ols_mid <- stats::lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = dplyr::filter(
    welfare,
    estimator == "OLS",
    year >= 2011,
    year <= 2015
  )
)

reg_iv_mid <- stats::lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = dplyr::filter(
    welfare,
    estimator == "IV",
    year >= 2011,
    year <= 2015
  )
)

# 2016 onward

reg_ols_post <- stats::lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = dplyr::filter(
    welfare,
    estimator == "OLS",
    year >= 2016
  )
)

reg_iv_post <- stats::lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = dplyr::filter(
    welfare,
    estimator == "IV",
    year >= 2016
  )
)

options(tinytable_latex_placement = "H")

modelsummary::modelsummary(
  list(
    "\\textbf{Panel A: OLS}" = list(
      "2011--2015" = reg_ols_mid,
      "2016 onward" = reg_ols_post
    ),
    "\\textbf{Panel B: IV}" = list(
      "2011--2015" = reg_iv_mid,
      "2016 onward" = reg_iv_post
    )
  ),
  shape = "rbind",
  coef_map = c(
    "(Intercept)" = "Constant",
    "Share_athletes" = "Share of Athletes",
    "Share_Graduates" = "Share of Graduates"
  ),
  estimate = "{estimate}{stars}",
  statistic = "({std.error})",
  stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
  gof_omit = ".*",
  title = "Welfare Gains and Student Composition",
  notes = paste(
    "Standard errors in parentheses.",
    "* p < 0.10, ** p < 0.05, *** p < 0.01."
  ),
  output = "output/welfare_gain_by_period.tex"
)


# ============================================================
# 1. Calculate the dish-level elasticity matrix
# ============================================================

elasticity_matrix <- function(alpha, dat) {

  dish_market <- dat |>
    dplyr::summarise(
      share_dish = sum(Share, na.rm = TRUE),
      price_share = sum(Price * Share, na.rm = TRUE),
      .by = c(University, year, dish)
    )

  dish_j <- dish_market |>
    dplyr::transmute(
      University,
      year,
      dish_j = as.character(dish),
      share_j = share_dish
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
      elasticity = dplyr::if_else(
        dish_j == dish_k,
        alpha * price_share_k * (1 - 1 / share_j),
        alpha * price_share_k
      )
    ) |>
    dplyr::summarise(
      elasticity = mean(elasticity, na.rm = TRUE),
      .by = c(dish_j, dish_k)
    )
}


# ============================================================
# 2. OLS and preferred IV price sensitivities
# ============================================================

alpha_ols <- -unname(stats::coef(reg)[1])
alpha_iv <- -unname(stats::coef(reg_iv)[1])


# ============================================================
# 3. Define the three time periods
# ============================================================

periods <- list(
  "Before 2011" = dplyr::filter(dat, year < 2011),
  "2011–2015" = dplyr::filter(dat, year >= 2011, year <= 2015),
  "2016 onward" = dplyr::filter(dat, year >= 2016)
)


# ============================================================
# 4. Calculate all six elasticity matrices
# ============================================================

elast_all <- lapply(names(periods), function(period_name) {

  d <- periods[[period_name]]

  dplyr::bind_rows(
    elasticity_matrix(alpha_ols, d) |>
      dplyr::mutate(estimator = "OLS"),

    elasticity_matrix(alpha_iv, d) |>
      dplyr::mutate(estimator = "IV")
  ) |>
    dplyr::mutate(period = period_name)

}) |>
  dplyr::bind_rows()


# ============================================================
# 5. Prepare labels and a common color scale
# ============================================================

dish_order <- sort(unique(as.character(dat$dish)))
dish_count <- length(dish_order)

max_abs <- max(abs(elast_all$elasticity), na.rm = TRUE)
if (!is.finite(max_abs) || max_abs == 0) max_abs <- 1

elast_all <- elast_all |>
  dplyr::mutate(
    dish_j = factor(dish_j, levels = rev(dish_order)),
    dish_k = factor(dish_k, levels = dish_order),
    estimator = factor(estimator, levels = c("OLS", "IV")),

    # Display one decimal; omit values that round to zero
    label = ifelse(
      is.na(elasticity) | round(elasticity, 1) == 0,
      "",
      sprintf("%.1f", elasticity)
    ),

    # White text on darker tiles, dark text on lighter tiles
    label_color = ifelse(
      abs(elasticity) >= 0.55 * max_abs,
      "white",
      "#263238"
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
# 6. Create a polished two-panel figure for each period
# ============================================================

make_elasticity_plot <- function(period_name) {

  plot_data <- dplyr::filter(
    elast_all,
    period == period_name
  )

  ggplot2::ggplot(
    plot_data,
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

    # Numeric labels only where the rounded elasticity is nonzero
    ggplot2::geom_text(
      ggplot2::aes(
        label = label,
        color = label_color
      ),
      size = label_size,
      show.legend = FALSE
    ) +

    # Subtle border around the own-price elasticities
    ggplot2::geom_tile(
      data = dplyr::filter(
        plot_data,
        as.character(dish_j) == as.character(dish_k)
      ),
      ggplot2::aes(x = dish_k, y = dish_j),
      inherit.aes = FALSE,
      fill = NA,
      color = "grey35",
      linewidth = 0.4
    ) +

    ggplot2::facet_wrap(~estimator, nrow = 1) +

    ggplot2::scale_fill_gradient2(
      low = "#B6534B",
      mid = "#F7F5F0",
      high = "#287A8A",
      midpoint = 0,
      limits = c(-max_abs, max_abs),
      oob = scales::squish,
      name = "Elasticity",
      labels = scales::label_number(accuracy = 0.1)
    ) +

    ggplot2::scale_color_identity() +
    ggplot2::scale_x_discrete(drop = FALSE) +
    ggplot2::scale_y_discrete(drop = FALSE) +
    ggplot2::coord_fixed() +

    ggplot2::labs(
      title = paste("Price Elasticities by Dish:", period_name),
      subtitle = "Average across University–year markets",
      x = "Price of dish k",
      y = "Demand for dish j",
      caption = paste(
        "Diagonal cells: own-price elasticities.",
        "Off-diagonal cells: cross-price elasticities.",
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
        size = 12
      ),
      strip.background = ggplot2::element_rect(
        fill = "#EEF1F3",
        color = NA
      ),
      panel.grid = ggplot2::element_blank(),
      panel.spacing = grid::unit(1.2, "lines"),
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
}


# ============================================================
# 7. Generate and save the three figures
# ============================================================

plots <- lapply(names(periods), make_elasticity_plot)
names(plots) <- names(periods)

file_stems <- c(
  "Before 2011" = "before_2011",
  "2011–2015" = "2011_2015",
  "2016 onward" = "2016_plus"
)

fig_width <- max(12, 0.55 * dish_count + 4)
fig_height <- max(6.5, 0.45 * dish_count + 2.5)

for (period_name in names(plots)) {
  ggplot2::ggsave(
    filename = paste0(
      "output/elasticity_",
      file_stems[[period_name]],
      ".pdf"
    ),
    plot = plots[[period_name]],
    width = fig_width,
    height = fig_height,
    units = "in"
  )
}