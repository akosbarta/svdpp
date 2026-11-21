#' Fit a baseline predictor
#'
#' Fits the baseline estimates of Koren (2008, Sec. 2.1),
#' \eqn{b_{ui} = \mu + b_u + b_i}, where \eqn{\mu} is the mean rating, by
#' solving
#' \deqn{\min_{b} \sum_{(u,i)} (r_{ui} - \mu - b_u - b_i)^2 +
#'   \lambda_u \sum_u b_u^2 + \lambda_i \sum_i b_i^2}
#' exactly with alternating least squares. These baselines are a model in
#' their own right and are also the fixed offsets \eqn{b_{uj}} used inside
#' [asvd()].
#'
#' @inheritParams svdpp
#' @param lambda Named numeric vector `c(user = , item = )` of the
#'   regularisation for user and item biases. The defaults (15 and 10) are
#'   the same as Surprise's `BaselineOnly`.
#' @param maxit Maximum number of alternating sweeps.
#' @param tol Stop when no bias changes by more than this.
#'
#' @return An object of class `"baseline_predictor"` (and `"cf_model"`).
#' @examples
#' set.seed(1)
#' d <- simulate_ratings()
#' fit <- baseline_predictor(d)
#' fit
#' head(predict(fit, d))
#' @export
baseline_predictor <- function(data, user = "user", item = "item",
                               rating = "rating",
                               lambda = c(user = 15, item = 10),
                               maxit = 100L, tol = 1e-8) {
  lambda <- check_baseline_lambda(lambda, "lambda")
  maxit <- check_number(maxit, "maxit", 1, int = TRUE)
  tol <- check_number(tol, "tol", 0)
  t0 <- proc.time()[["elapsed"]]
  d <- prepare_ratings(data, user, item, rating)
  b <- fit_baseline(d, lambda, maxit, tol)
  new_cf_model(
    "baseline_predictor", d, b$mu, b$bu, b$bi,
    Q = matrix(0, d$ni, 0L), U = matrix(0, d$nu, 0L),
    extra = list(lambda = lambda, iterations = b$iterations),
    columns = c(user = user, item = item, rating = rating),
    call = match.call(), time = proc.time()[["elapsed"]] - t0,
    label = "Baseline predictor")
}

fit_baseline <- function(d, lambda, maxit = 100L, tol = 1e-8) {
  mu <- mean(d$r)
  nU <- tabulate(d$u, d$nu)
  nI <- tabulate(d$i, d$ni)
  bu <- numeric(d$nu)
  bi <- numeric(d$ni)
  it <- 0L
  for (it in seq_len(maxit)) {
    bi_new <- group_sum(d$r - mu - bu[d$u], d$i, d$ni) / (lambda[["item"]] + nI)
    bu_new <- group_sum(d$r - mu - bi_new[d$i], d$u, d$nu) / (lambda[["user"]] + nU)
    delta <- max(abs(bi_new - bi), abs(bu_new - bu))
    bi <- bi_new
    bu <- bu_new
    if (delta < tol) break
  }
  list(mu = mu, bu = bu, bi = bi, iterations = it)
}

check_baseline_lambda <- function(lambda, name = "baseline_lambda") {
  if (!is.numeric(lambda) || length(lambda) != 2L || anyNA(lambda) ||
      any(!is.finite(lambda)) || any(lambda < 0) || is.null(names(lambda)) ||
      !setequal(names(lambda), c("user", "item"))) {
    stop(sprintf("`%s` must be c(user = , item = ) with non-negative values.",
                 name), call. = FALSE)
  }
  lambda
}
