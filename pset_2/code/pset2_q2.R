library(readr)
library(fixest)
library(modelsummary)
library(readxl)
library(dplyr)
library(ggplot2)

dat <- readr::read_csv("data/input/choice_noodles.csv")

# constructing outside option

dat <- dat |>
  dplyr::mutate(
    Share0 = 1 - sum(Share),
    .by = c(University, year)
  )

### OLS

reg <- feols(
  Share - Share0 ~ Price + SpicyIndex + WheatContent + Veg2meat + Calories | dish,
  data = dat
)

modelsummary(
  reg,
  coef_map = c(
    "Price"        = "Price",
    "SpicyIndex"   = "Spicy",
    "WheatContent" = "Wheat",
    "Veg2meat"     = "Veg2Meat",
    "Calories"     = "Calories"
  ),
  estimate = "{estimate}{stars}",
  statistic = "({std.error})",
  stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
  gof_map = c("nobs", "r.squared", "r.squared.within"),
  title = "OLS",
  output = "output/part2_table_ols.tex"
)

### IV

college_year <- readxl::read_xls("data/input/college-year.xls") |>
  mutate(across(year, as.numeric))

# merging the datasets

dat <- dat |>
  left_join(college_year, by = c("University" = "campus", "year"))

reg_iv <- feols(
  Share - Share0 ~ SpicyIndex + WheatContent + Veg2meat + Calories |
    dish |
    Price ~ VegPIDX + WheatPIDX + MeatPIDX + WageIDX + ReEsIDX,
  data = dat
)

# Save the regression table as LaTeX
modelsummary(
  reg_iv,
  coef_map = c(
    "fit_Price"        = "Price",
    "SpicyIndex"   = "Spicy",
    "WheatContent" = "Wheat",
    "Veg2meat"     = "Veg2Meat",
    "Calories"     = "Calories"
  ),
  estimate = "{estimate}{stars}",
  statistic = "({std.error})",
  stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
  gof_map = c("nobs", "r.squared", "r.squared.within"),
  title = "IV",
  output = "output/part2_table_iv.tex"
)

### Welfare

logit_welfare <- function(dat, reg) {
  dat |>
    dplyr::mutate(delta = predict(reg, newdata = dat)) |>
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
    filter(WheatContent != 0) |>
    logit_welfare(reg) |>
    rename(welfare_gf = welfare)

  # merging the two and taking diffs
  left_join(gluten, gluten_free, by = c("University", "year")) |>
    mutate(welfare_gain = welfare - welfare_gf)
}

# stacking welfare gains for both estimates
welfare <- purrr::map2(
  list(reg, reg_iv),
  c("OLS", "IV"),
  function(model, name) {
    welfare_gain(dat, model) |>
      mutate(estimator = name)
  }
) |>
  bind_rows()

# Diff-in-diff style graph

dat_plot <- welfare |>
  summarise(
    across(c(welfare, welfare_gf), \(x) mean(x, na.rm = TRUE)),
    .by = c(year, estimator)
  ) |>
  pivot_longer(
    cols = c(welfare, welfare_gf),
    names_to = "measure",
    values_to = "welfare"
  ) |>
  mutate(
    measure = recode(
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
    label = ifelse(measure == "GF",
                   paste0(estimator, " (GF)"),
                   estimator)
  )

ggplot(
  dat_plot,
  aes(
    x = year,
    y = welfare,
    color = estimator,
    linetype = measure,
    group = interaction(estimator, measure)
  )
) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 1.5) +
  geom_text(
    data = label_data,
    aes(label = label),
    hjust = 0,
    nudge_x = 1.5,
    size = 3.5,
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = c("OLS" = "#4477AA", "IV" = "#CC6677")
  ) +
  scale_linetype_manual(
    values = c("Baseline" = "solid", "GF" = "longdash")
  ) +
  scale_x_continuous(
    limits = c(min(dat_plot$year), max(dat_plot$year) + 4),
    breaks = scales::pretty_breaks(n = 8)
  ) +
  labs(
    title = "Consumer Surplus Over Time",
    x = NULL,
    y = "Welfare per consumer"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text = element_text(color = "grey25"),
    legend.position = "none",
    plot.margin = margin(5.5, 35, 5.5, 5.5)
  )

ggsave(
  "output/welfare_over_time.pdf",
  width = 9,
  height = 5,
  units = "in"
)

# Graph of differences

dat_plot <- welfare |>
  summarise(
    welfare_gain = mean(welfare_gain, na.rm = TRUE),
    .by = c(year, estimator)
  )

gain_labels <- dat_plot |>
  filter(year == max(dat_plot$year)) |>
  mutate(label = estimator)

# Plot
ggplot(
  dat_plot,
  aes(x = year, y = welfare_gain, color = estimator, group = estimator)
) +
  geom_hline(
    yintercept = 0,
    color = "grey75",
    linewidth = 0.5
  ) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 1.6) +
  geom_text(
    data = gain_labels,
    aes(label = label),
    nudge_x = 1.2,
    hjust = 0,
    size = 3.8,
    fontface = "bold",
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = c("OLS" = "#4477AA", "IV" = "#CC6677")
  ) +
  scale_x_continuous(
    limits = c(min(dat_plot$year), max(dat_plot$year) + 3),
    breaks = scales::pretty_breaks(n = 8),
    expand = expansion(mult = c(0.01, 0))
  ) +
  labs(
    title = "Welfare Gains vs. Counterfactual Over Time",
    x = NULL,
    y = "Welfare gain per consumer"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.title.position = "plot",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text = element_text(color = "grey25"),
    axis.title.y = element_text(margin = margin(r = 10)),
    legend.position = "none",
    plot.margin = margin(8, 30, 8, 8)
  )

ggsave(
  "output/welfare_gain_over_time.pdf",
  width = 9,
  height = 5,
  units = "in"
)

### Regressing welfare gains

welfare <- welfare |>
  left_join(college_year, by = c("University" = "campus", "year"))

# Before 2011
reg_ols_pre <- lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = filter(welfare, estimator == "OLS", year < 2011)
)

reg_iv_pre <- lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = filter(welfare, estimator == "IV", year < 2011)
)

# 2011–2014
reg_ols_mid <- lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = filter(welfare, estimator == "OLS", year >= 2011, year <= 2015)
)

reg_iv_mid <- lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = filter(welfare, estimator == "IV", year >= 2011, year <= 2015)
)

# 2015 onward
reg_ols_post <- lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = filter(welfare, estimator == "OLS", year >= 2016)
)

reg_iv_post <- lm(
  welfare_gain ~ Share_athletes + Share_Graduates,
  data = filter(welfare, estimator == "IV", year >= 2016)
)

modelsummary(
  list(
    "\\textbf{Panel A: OLS}" = list(
      "2011--2015"  = reg_ols_mid,
      "2016 onward" = reg_ols_post
    ),
    "\\textbf{Panel B: IV}" = list(
      "2011--2015"  = reg_iv_mid,
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
  notes = "Standard errors in parentheses. * p < 0.10, ** p < 0.05, *** p < 0.01.",
  output = "output/welfare_gain_by_period.tex"
)
