plot_streaming <- function(data, baseline_data, psize = 2) {
  do_plot <- function(data, column, ylab,
                      scale_val = "free", include_baseline = TRUE,
                      ytrans = "identity",
                      labelsfun = scales::label_number_auto()) {
    baseline <- baseline_data |>
      filter(algorithm %in% c("KFC", "coreset")) |>
      filter(k == 32) |>
      select(dataset, k, algorithm, radius, time_s, tau) |>
      print(n = 100) |>
      filter(is.na(tau) | (tau >= 32)) |>
      semi_join(select(data, dataset, k)) |>
      group_by(dataset) |>
      slice_min({{ column }})

    baseline |>
      select(algorithm, dataset, radius, time_s) |>
      print()

    plotdata <- data |>
      drop_na(streaming_memory_bytes) |>
      mutate(
        param = if_else(is.na(tau), epsilon, tau),
        bytes_per_point = streaming_memory_bytes / n
      ) |>
      group_by(algorithm, dataset, k, param) |>
      filter(radius != min(radius), radius != max(radius)) |>
      reframe(
        {{ column }},
        streaming_memory_bytes = mean(streaming_memory_bytes)
      ) |>
      arrange(param)

    labels <- plotdata |>
      group_by(dataset, algorithm) |>
      filter(streaming_memory_bytes == max(streaming_memory_bytes))

    p <- plotdata |>
      ggplot(
        aes(
          streaming_memory_bytes, {{ column }},
          color = algorithm, shape = algorithm
        )
      ) +
      geom_path(stat = "summary") +
      geom_point(stat = "summary", size = psize) +
      # geom_text(
      #   aes(
      #     label = param
      #   ),
      #   data = labels,
      #   size = 2,
      #   stat = "summary",
      #   nudge_x = 1,
      #   ha = 0
      # ) +
      scale_algorithm() +
      labs(x = "memory (bytes)", y = ylab) +
      scale_y_continuous(trans = ytrans, labels = labelsfun) +
      scale_x_continuous(
        trans = "log",
        labels = scales::label_number(scale_cut=scales::cut_short_scale()),
        guide = guide_axis(n.dodge = 1),
        n.breaks = 4
      ) +
      facet_wrap(vars(dataset), scale = scale_val, ncol = 4) +
      theme_paper() +
      theme(
        legend.position = "top",
        plot.margin = margin(0, 10, 0, 0)
      )

    if (include_baseline) {
      p <- p + geom_hline(aes(yintercept = {{ column }}), linetype = "dashed", data = baseline)
    }
    p
  }

  p_radius <- do_plot(data, radius, ylab = "radius", labelsfun = scales::label_number(scales_cut=scales::cut_short_scale()))
  p_time <- do_plot(
    data, time_s,
    ylab = "time (s)", ytrans = "log",
    labelsfun = scales::number_format(accuracy = 1)
  )
  legend <- get_legend(p_radius)
  p_radius <- p_radius + theme(legend.position = "none")
  p_time <- p_time + theme(legend.position = "none")

  plot_grid(
    legend,
    plot_grid(p_time, p_radius),
    ncol = 1,
    rel_heights = c(1, 10)
  )
}
