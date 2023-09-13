library(ggplot2)
library(tibble)
library(dplyr)

fig_width <- 4
fig_height <- 2

df_poly <- data.frame(
  x = c(-Inf, Inf, Inf, -Inf),
  y = c(0, 0, Inf, Inf)
)

set.seed(1234)

n1 <- 100
n2 <- 100
c1 <- tibble(
  x = rnorm(n1, sd = 0.5),
  y = rnorm(n1, sd = 0.5),
  color = "#d55e00"
)
c2 <- tibble(
  x = rnorm(n2, 4, sd = 0.5),
  y = rnorm(n2, 0, sd = 0.5),
  color = "#0072b2"
)

bind_rows(c1, c2) |>
  ggplot(aes(x, y)) +
  geom_point() +
  scale_color_identity() +
  coord_equal() +
  theme_void()
ggsave("figs/example-no-color.pdf", width = fig_width, height = fig_height)

bind_rows(c1, c2) |>
  ggplot(aes(x, y, color = color)) +
  geom_point() +
  scale_color_identity() +
  coord_equal() +
  theme_void()
ggsave("figs/example-color.pdf", width = fig_width, height = fig_height)

bind_rows(c1, c2) |>
  ggplot(aes(x, y, color = color)) +
  geom_polygon(
    aes(x, y),
    alpha = 0.2,
    data = df_poly,
    inherit.aes = FALSE
  ) +
  geom_hline(yintercept = 0) +
  geom_point() +
  scale_color_identity() +
  coord_equal() +
  theme_void()
ggsave("figs/example-color-clustering.pdf",
  width = fig_width, height = fig_height
)
