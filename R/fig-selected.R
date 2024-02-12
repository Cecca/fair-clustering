compact_plot <- function(data, xaes, yaes, xlab, ylab, ylog = FALSE, flabels = scales::label_number(scale_cut=scales::cut_short_scale()), scales_val = "free_y", ncol = 4) {
  if (ylog) {
    tfun <- "log10"
  } else {
    tfun <- "identity"
  }
  data |>
    ggplot(aes({{ xaes }}, {{ yaes }}, color = algorithm, shape = algorithm, linetype = algorithm)) +
    geom_point(size = 2) +
    geom_line() +
    scale_algorithm() +
    facet_wrap(vars(dataset), scale = scales_val, ncol = ncol) +
    scale_y_continuous(labels = flabels, trans = tfun) +
    scale_x_continuous(trans = "log2") +
    labs(
      x = xlab,
      y = ylab
    ) +
    theme_paper() +
    theme(legend.position = "top", plot.margin = margin(0, 4, 0, 0))
}

two_row_plot <- function(data, ncol = 4) {
  data |>
    filter(algorithm == "Bera-et-al") |>
    print()
  p_radius <- data |>
    compact_plot(k, radius, ylab = "radius", xlab = "", ylog = F, ncol = ncol)
  p_time <- data |>
    compact_plot(k, time_s,
      ylab = "time (s)", xlab = "k", ylog = T,
      flabels = scales::label_number(), scales_val = "fixed", ncol = ncol
    )
  legend <- cowplot::get_legend(p_radius)
  p_radius <- p_radius + theme(
    legend.position = "none"
  )
  p_time <- p_time + theme(legend.position = "none")
  p <- cowplot::plot_grid(
    legend,
    cowplot::plot_grid(
      p_radius,
      p_time,
      align = "v",
      ncol = 1
    ),
    ncol = 1,
    rel_heights = c(1, 10)
  )
}
# ggsave("figs/selected-performance.pdf", width=6, height=4)
