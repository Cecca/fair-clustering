library(targets)
source("R/functions.R")
source("R/fig-selected.R")
source("R/fig-streaming.R")
source("R/fig-mapreduce.R")
tar_option_set(packages = c(
  "tidyverse",
  "cowplot",
  "ggrepel",
  "stringr",
  "kableExtra"
))

selected_datasets <- c("hmda", "census1990", "athlete", "diabetes")
large_datasets <- c("hmda", "census1990")

list(
  ##########################################################################
  # Main paper
  ##########################################################################
  #
  # Sequential
  tar_target(
    selected_data,
    get_data() |>
      filter(dataset %in% selected_datasets) |>
      filter(time_s <= 3600)
  ),
  tar_target(sequential_figure, {
    selected_data |>
      filter(algorithm != "dummy") |>
      filter((tau %in% c(1, 32)) | (is.na(tau))) |>
      mutate(algorithm = if_else(!is.na(tau),
        str_c(algorithm, " (", tau, "k)"),
        algorithm,
      )) |>
      two_row_plot()
    ggsave("figs/selected-performance.pdf", width = 6, height = 4)
  }),

  # MapReduce
  tar_target(
    mr_data,
    get_data(mr = TRUE, do_summarise = FALSE) |> filter(dataset %in% large_datasets)
  ),
  tar_target(mr_figure, {
    plot_mr(mr_data)
    ggsave("figs/mapreduce.pdf", width = 6, height = 2)
  }),

  # Streaming
  tar_target(
    streaming_data,
    get_data(streaming = TRUE, do_summarise = FALSE) |>
      filter(k == 32, dataset %in% large_datasets) |>
      filter(is.na(tau) | (tau <= 512)) |>
      filter(is.na(epsilon) | (epsilon >= 0.001))
  ),
  tar_target(streaming_figure, {
    plot_streaming(streaming_data, selected_data)
    ggsave("figs/streaming.pdf", width = 6, height = 2)
  }),

  ##########################################################################
  # Appendix
  ##########################################################################
  #
  # Sequential
  tar_target(
    all_data,
    get_data() |>
      filter(time_s <= 3600)
  ),
  tar_target(sequential_figure_appendix, {
    all_data |>
      filter(algorithm != "dummy") |>
      filter((tau %in% c(1, 32)) | (is.na(tau))) |>
      mutate(algorithm = if_else(!is.na(tau),
        str_c(algorithm, " (", tau, "k)"),
        algorithm,
      )) |>
      two_row_plot(ncol = 5)
    ggsave("figs/all-sequential-performance.pdf", width = 12, height = 8)
  }),

  # Streaming
  tar_target(
    all_streaming_data,
    get_data(streaming = TRUE, do_summarise = TRUE) |> filter(k == 32)
  ),
  tar_target(all_streaming_figure, {
    plot_streaming(streaming_data, selected_data)
    ggsave("figs/all-streaming.pdf", width = 6, height = 2)
  }),

  # Additive violation
  tar_target(
    table_additive_violation,
    all_data |>
      filter(k == 32) |>
      filter((tau == 32) | is.na(tau)) |>
      filter(algorithm != "dummy") |>
      select(dataset, algorithm, additive_violation) |>
      pivot_wider(names_from = "algorithm", values_from = "additive_violation") |>
      kbl(format = "latex", booktabs=T) |>
      str_replace_all("NA", "-") |>
      write_file("figs/additive_violation.tex")
  ),

  # Coreset radius ratios
  tar_target(
    table_coreset_radius_ratio,
    all_data |>
      filter(algorithm == "coreset", tau %in% c(1, 32), k == 32) |>
      select(dataset, tau, coreset_radius) |>
      pivot_wider(names_from = "tau", values_from = "coreset_radius") |>
      mutate(radius_ratio = `1` / `32`) |>
      rename(`1k` = `1`, `32k` = `32`) |>
      select(dataset, radius_ratio) |>
      # kbl(format = "markdown", digits = 2)
      kbl(format = "latex", booktabs=T, digits=2) |>
      str_replace_all("NA", "-") |>
      write_file("figs/coreset_radius_ratio.tex")
  ),

  # Unfair radius
  tar_target(
    table_unfair_radius,
    all_data |>
      filter(algorithm == "unfair", k %in% c(16, 64)) |>
      select(dataset, k, radius) |>
      pivot_wider(names_from = "k", values_from = "radius") |>
      mutate(radius_ratio = `16` / `64`) |>
      filter(dataset %in% c("bank", "diabetes"))
  )
)
