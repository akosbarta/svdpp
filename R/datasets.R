#' Simulate ratings from a latent factor model
#'
#' Draws ratings \eqn{\mu + b_u + b_i + p_u^\top q_i + \varepsilon} for
#' random user-item pairs, rounded and clipped to the 1--5 star scale.
#' Useful for examples and tests. Call [set.seed()] first for reproducible
#' data.
#'
#' @param n_users,n_items Number of users and items.
#' @param n_ratings Number of distinct user-item pairs to rate.
#' @param factors Number of true latent factors.
#' @param noise Standard deviation of the rating noise.
#'
#' @return A data frame with columns `user`, `item` (both integer) and
#'   `rating`.
#' @examples
#' set.seed(1)
#' head(simulate_ratings())
#' @export
simulate_ratings <- function(n_users = 300L, n_items = 150L, n_ratings = 6000L,
                             factors = 3L, noise = 0.5) {
  n_users <- check_number(n_users, "n_users", 1, int = TRUE)
  n_items <- check_number(n_items, "n_items", 1, int = TRUE)
  n_ratings <- check_number(n_ratings, "n_ratings", 1, int = TRUE)
  factors <- check_number(factors, "factors", 1, int = TRUE)
  noise <- check_number(noise, "noise", 0)
  n_cells <- as.numeric(n_users) * n_items
  if (n_ratings > n_cells) {
    stop(sprintf(paste0("`n_ratings` (%d) cannot exceed the number of ",
                        "user-item pairs, `n_users` * `n_items` (%.0f)."),
                 n_ratings, n_cells), call. = FALSE)
  }
  cell <- sample.int(n_cells, n_ratings) - 1
  u <- as.integer(cell %/% n_items) + 1L
  i <- as.integer(cell %% n_items) + 1L
  bu <- stats::rnorm(n_users, 0, 0.4)
  bi <- stats::rnorm(n_items, 0, 0.4)
  P <- matrix(stats::rnorm(n_users * factors, 0, 0.6), n_users)
  Q <- matrix(stats::rnorm(n_items * factors, 0, 0.6), n_items)
  r <- 3.5 + bu[u] + bi[i] + rowSums(P[u, , drop = FALSE] * Q[i, , drop = FALSE]) +
    stats::rnorm(n_ratings, 0, noise)
  data.frame(user = u, item = i, rating = pmin(5, pmax(1, round(r))))
}

#' Download a MovieLens ratings data set
#'
#' Downloads one of the MovieLens data sets from GroupLens
#' (<https://grouplens.org/datasets/movielens/>) and returns its ratings.
#' The data are not shipped with the package because their licence does not
#' allow redistribution; read the licence on the GroupLens site before use.
#'
#' @param size `"100k"`, `"1m"` or `"10m"`.
#' @param dir Directory to download and unpack into. If the archive is
#'   already there it is not downloaded again. The default is a temporary
#'   directory; pass a persistent one to keep the data between sessions.
#'
#' @return A data frame with columns `user`, `item`, `rating` and
#'   `timestamp`, or `NULL` (with a message) if the data could not be
#'   downloaded, for example without an internet connection.
#' @examples
#' \donttest{
#' ml <- movielens("100k")
#' if (!is.null(ml)) head(ml)
#' }
#' @export
movielens <- function(size = c("100k", "1m", "10m"), dir = tempdir()) {
  size <- match.arg(size)
  info <- switch(size,
    "100k" = list(zip = "ml-100k.zip", file = "ml-100k/u.data", sep = "\t",
                  mb = 5),
    "1m" = list(zip = "ml-1m.zip", file = "ml-1m/ratings.dat", sep = "::",
                mb = 6),
    "10m" = list(zip = "ml-10m.zip", file = "ml-10M100K/ratings.dat",
                 sep = "::", mb = 63))
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  zip <- file.path(dir, info$zip)
  path <- file.path(dir, info$file)
  if (!file.exists(path)) {
    if (!file.exists(zip)) {
      url <- paste0("https://files.grouplens.org/datasets/movielens/", info$zip)
      old <- options(timeout = max(60 + 10 * info$mb, getOption("timeout")))
      on.exit(options(old), add = TRUE)
      ok <- tryCatch(
        utils::download.file(url, zip, mode = "wb", quiet = TRUE) == 0,
        error = function(e) FALSE, warning = function(w) FALSE)
      if (!ok) {
        unlink(zip)
        message("Could not download ", url,
                ". Check your internet connection and try again.")
        return(invisible(NULL))
      }
    }
    unz <- tryCatch(utils::unzip(zip, files = info$file, exdir = dir),
                    error = function(e) character(), warning = function(w) character())
    if (!length(unz) || !file.exists(path)) {
      unlink(zip)
      message("The downloaded file ", zip, " could not be unpacked; ",
              "it has been removed. Please try again.")
      return(invisible(NULL))
    }
  }
  if (info$sep == "\t") {
    x <- utils::read.table(path, sep = "\t",
                           colClasses = c("integer", "integer", "numeric", "integer"))
  } else {
    x <- utils::read.table(path, sep = ":",
                           colClasses = c("integer", "NULL", "integer", "NULL",
                                          "numeric", "NULL", "integer"))
  }
  names(x) <- c("user", "item", "rating", "timestamp")
  x
}
