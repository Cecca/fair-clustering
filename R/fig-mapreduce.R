plot_mr <- function(data) {
  p_time <- data |>
    filter(k==32) |>
    filter(parallelism %in% c(2,4,8,16)) |>
    filter(tau %in% c(1,2)) |>
    select(dataset, algorithm, parallelism, time_s, time_assignment_s, tau, radius) |>
    print() |>
    ggplot(aes(parallelism, time_s, group=tau, color=algorithm, shape=algorithm)) +
    geom_area(aes(y=time_assignment_s, fill=algorithm), color=NA, alpha=0.5, position="identity") +
    geom_point() +
    geom_line() +
    facet_wrap(vars(dataset), scales='free_y', ncol=4) +
    scale_algorithm() +
    scale_y_continuous(limits=c(0,NA)) +
    scale_x_continuous(trans="log2") +
    labs(
      x = "parallelism",
      y = "time (s)"
    ) +
    theme_paper() +
    theme(
      plot.margin = margin(0,4,0,0),
      legend.margin = margin(0,0,0,0)
    )

  p_radius <- data |>
    filter(k==32) |>
    filter(parallelism %in% c(2,4,8,16)) |>
    filter(tau %in% c(1,2)) |>
    select(dataset, algorithm, parallelism, time_s, time_assignment_s, tau, radius) |>
    print() |>
    ggplot(aes(parallelism, radius, group=tau, color=algorithm, shape=algorithm)) +
    geom_point() +
    geom_line() +
    facet_wrap(vars(dataset), scales='free_y', ncol=4) +
    scale_algorithm() +
    scale_y_continuous(labels=scales::label_number_si()) +
    scale_x_continuous(trans="log2") +
    labs(
      x = "parallelism",
      y = "radius"
    ) +
    theme_paper() +
    theme(
      plot.margin = margin(0,4,0,0),
      legend.margin = margin(0,0,0,0)
    )

  legend <- cowplot::get_legend(p_radius)
  p_time <- p_time + theme(legend.position="none")
  p_radius <- p_radius + theme(legend.position="none")

  plot_grid(
    legend,
    plot_grid(p_time, p_radius),
    ncol=1,
    rel_heights=c(1,10)
  )
}

