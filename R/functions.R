theme_paper <- function() {
  theme_bw() +
    theme(
      legend.position = "top",
      strip.background = element_blank(),
      strip.text = element_text(hjust=0),
      plot.margin = margin(0,0,0,0)
    )
}

scale_algorithm <- function() {
  list(
    scale_color_manual(values = c(
      "coreset" = "#5778a4",
      "coreset (1)" = "#5778a4",
      "coreset (32)" = "#5778a4",
      "coreset-MR" = "#5778a4",
      "coreset-stream" = "#5778a4",
      "KFC" = "#e49444",
      "unfair" = "#85b6b2",
      "Bera-et-al" = "#d1615d",
      "Bera-et-al-MR" = "#d1615d",
      "Bera-et-al-stream" = "#d1615d",
      "dummy" = "black"
    ), aesthetics=c("color", "fill")),
    scale_shape_manual(values = c(
      "coreset" = 19,
      "coreset (1)" = 19,
      "coreset (32)" = 19,
      "coreset-MR" = 19,
      "coreset-stream" = 19,
      "KFC" = 17,
      "unfair" = 18,
      "Bera-et-al" = 15,
      "Bera-et-al-MR" = 15,
      "Bera-et-al-stream" = 15,
      "dummy" = 8
    )),
    scale_linetype_manual(values = c(
      "coreset" = "solid",
      "coreset (1)" = "dotted",
      "coreset (32)" = "solid",
      "coreset-MR" = "solid",
      "coreset-stream" = "solid",
      "KFC" = "solid",
      "unfair" = "solid",
      "Bera-et-al" = "solid",
      "Bera-et-al-MR" = "solid",
      "Bera-et-al-stream" = "solid",
      "dummy" = "solid"
    ))
  )
}

imgdir <- "imgs"

imgpath <- function(key) {
  str_c(imgdir, key, "clustering.png", sep="/")
}


dataset_stats <- function() {
  con <- DBI::dbConnect(RSQLite::SQLite(), "results.db")
  stats <- tbl(con, "dataset_stats") |> collect()
  DBI::dbDisconnect(con)
  stats
}

get_data <- function(delta_val = 0.01, mr = FALSE, streaming = FALSE, do_summarise = TRUE) {
  con <- DBI::dbConnect(RSQLite::SQLite(), "results.db")

  if (mr) {
    algofilter <- "algorithm like '%mr%'"
    additional_col <- "json_extract(params, '$.parallelism') as parallelism, 0 as streaming_memory_bytes,"
  } else if (streaming) {
    algofilter <- "algorithm like '%stream%'"
    additional_col <- "1 as parallelism, json_extract(additional_metrics, '$.streaming_memory_bytes') as streaming_memory_bytes,"
  } else {
    algofilter <- "algorithm not like '%mr%' and algorithm not like '%stream%'"
    additional_col <- "1 as parallelism, 0 as streaming_memory_bytes,"
  }

  q <- str_glue(
      "select *, 
       json_extract(params, '$.tau') / k as tau,
       json_extract(params, '$.epsilon') as epsilon,
       {additional_col}
       cast(additional_metrics -> '$.coreset_radius' as real) as coreset_radius,
       cast(additional_metrics -> '$.time_coreset_s' as real) as coreset_time_s,
       cast(additional_metrics -> '$.time_assignment_s' as real) as time_assignment_s
       from results
       where dataset not like '%std'
       and dataset not like 'census1990'
       and {algofilter}")

  smallest_radius <- tbl(con, sql("select dataset, k, min(radius) as best_unfair_radius from results group by dataset, k")) |> collect()

  results <- tbl(con, sql(q)) |> 
    inner_join(tbl(con, "dataset_stats")) |>
    collect() |>
    inner_join(smallest_radius) |>
    mutate(
      scaled_radius = radius / best_unfair_radius,
      dataset = case_when(
        dataset == "reuter_50_50" ~ "reuter",
        dataset == "census1990_age" ~ "census1990",
        T ~ dataset
      ),
      algorithm = case_when(
        algorithm == "coreset-fair-k-center" ~ "coreset",
        algorithm == "kfc-k-center" ~ "KFC",
        algorithm == "unfair-k-center" ~ "unfair",
        algorithm == "bera-et-al-k-center" ~ "Bera-et-al",
        algorithm == "bera-mr-fair-k-center" ~ "Bera-et-al-MR",
        algorithm == "bera-streaming-fair-k-center" ~ "Bera-et-al-stream",
        algorithm == "mr-coreset-fair-k-center" ~ "coreset-MR",
        algorithm == "streaming-coreset-fair-k-center" ~ "coreset-stream",
        T ~ algorithm
      ),
      algorithm = factor(algorithm, ordered=TRUE, levels=c(
        "unfair", 
        "Bera-et-al", 
        "Bera-et-al-MR", 
        "Bera-et-al-stream", 
        "KFC", 
        "coreset",
        "coreset-MR", 
        "coreset-stream", 
        "dummy"
      )),
      # timeout_s = if_else(time_s > 30*60, 30*60, timeout_s),
      timed_out = !is.na(timeout_s),
      # time_s = if_else(timed_out, timeout_s, time_s),
      scaled_time_spp = time_s / n,
      scaled_coreset_time_spp = coreset_time_s / n,
      img_path = imgpath(hdf5_key),
      coreset_size_frac = tau * k / n,
      dataset = fct_reorder(dataset, desc(n))
    ) |>
    filter(!timed_out)

  if (do_summarise) {
    results <- results |>
      group_by(dataset, algorithm, k, delta, tau, epsilon, parallelism, timed_out, coreset_size_frac, n, dimensions) |>
      summarise(
        across(c(radius, scaled_radius, coreset_radius, 
                  time_s, coreset_time_s, time_assignment_s, streaming_memory_bytes
              ), mean),
        additive_violation = max(additive_violation)
      ) |>
      mutate(streaming_memory_bytes = as.double(streaming_memory_bytes)) |>
      ungroup()
  }

  if (!is.na(delta_val)) {
    message(paste("filtering by delta=", delta_val))
    results <- filter(results, delta == delta_val)
  }

  DBI::dbDisconnect(con)
  results
}


