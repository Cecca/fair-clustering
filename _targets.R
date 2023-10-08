library(targets)
source("R/functions.R")
source("R/fig-selected.R")
source("R/fig-streaming.R")
source("R/fig-mapreduce.R")
tar_option_set(packages = c(
  "tidyverse",
  "cowplot",
  "stringr"
))

selected_datasets <- c("hmda", "census1990", "athlete", "diabetes")
large_datasets <- c("hmda", "census1990")

list(
  # Sequential
  tar_target(selected_data, 
    get_data() |> 
      filter(dataset %in% selected_datasets) |>
      filter(time_s <= 3600)
  ),
  tar_target(sequential_figure, {
    selected_data |>
      filter(algorithm != "dummy") |>
      filter(( tau %in% c(1, 32) ) | (is.na(tau))) |>
      mutate(algorithm = if_else(!is.na(tau),
        str_c(algorithm, " (", tau, ")"),
        algorithm,
      )) |>
      two_row_plot()
    ggsave("figs/selected-performance.pdf", width=6, height=4)
  }),

  # MapReduce
  tar_target(mr_data, get_data(mr=TRUE) |> filter(dataset %in% large_datasets)),
  tar_target(mr_figure, {
    plot_mr(mr_data)
    ggsave("figs/mapreduce.pdf", width=6, height=2)
  }),

  # Streaming
  tar_target(streaming_data, 
    get_data(streaming=TRUE, do_summarise=TRUE) |> filter(k == 32, dataset %in% large_datasets)
  ),
  tar_target(streaming_figure, {
    plot_streaming(streaming_data, selected_data)
    ggsave("figs/streaming.pdf", width=6, height=2)
  })
)
