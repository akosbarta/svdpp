## Internals shared by all model fitters.

## Argument checks with informative errors. `min` is inclusive unless
## `strict`; `int` requires a whole number and returns an integer.
check_number <- function(x, name, min = -Inf, strict = FALSE, int = FALSE) {
  ok <- is.numeric(x) && length(x) == 1L && !is.na(x) && is.finite(x) &&
    (if (strict) x > min else x >= min) && (!int || x == trunc(x))
  if (!ok) {
    stop(sprintf("`%s` must be a single %s %s %s.", name,
                 if (int) "whole number" else "number",
                 if (strict) ">" else ">=", format(min)), call. = FALSE)
  }
  if (int) as.integer(x) else as.numeric(x)
}

check_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("`%s` must be TRUE or FALSE.", name), call. = FALSE)
  }
  x
}

check_factors <- function(factors) {
  if (is.character(factors)) {
    stop(sprintf("`factors` must be a single whole number >= 1, not a string. %s",
                 positional_hint()), call. = FALSE)
  }
  check_number(factors, "factors", 1, int = TRUE)
}

## Calls written for svdpp 0.1.0, such as svdpp(d, "u", "i", "r"), now pass
## column names to `factors` (or `lambda`); say why.
positional_hint <- function() {
  paste0("Since svdpp 0.2.0 the column names come after the model ",
         "arguments; give them by name, e.g. user = \"u\", item = \"i\", ",
         "rating = \"r\".")
}

check_lambda <- function(lambda) check_number(lambda, "lambda", 0)

## Validate the estimator settings shared by svdpp() and asvd().
check_control <- function(lr, epochs, shuffle, maxit, init_sd, verbose) {
  list(lr = check_number(lr, "lr", 0, strict = TRUE),
       epochs = check_number(epochs, "epochs", 1, int = TRUE),
       shuffle = check_flag(shuffle, "shuffle"),
       maxit = check_number(maxit, "maxit", 1, int = TRUE),
       init_sd = check_number(init_sd, "init_sd", 0),
       verbose = check_flag(verbose, "verbose"))
}

## Stop if SGD has blown up (training error no longer finite).
check_diverged <- function(rmse, epoch) {
  if (!is.finite(rmse)) {
    stop(sprintf(paste0("SGD diverged in epoch %d: the training error is no ",
                        "longer finite. Use a smaller `lr` (and/or a larger ",
                        "`lambda`)."), epoch), call. = FALSE)
  }
}

## Sum x within groups g in 1..n (empty groups give 0).
group_sum <- function(x, g, n) {
  out <- numeric(n)
  s <- rowsum(x, g)
  out[as.integer(rownames(s))] <- s[, 1]
  out
}

## Users without ratings and items without ratings have no data term for
## their biases and q_i; zero them (SGD would otherwise leave the random init).
zero_unrated <- function(est, d) {
  rated_u <- tabulate(d$u, d$nu) > 0
  rated_i <- tabulate(d$i, d$ni) > 0
  est$bu[!rated_u] <- 0
  est$bi[!rated_i] <- 0
  est$Q[!rated_i, ] <- 0
  if (!is.null(est$P)) est$P[!rated_u, ] <- 0
  if (!is.null(est$X)) est$X[!rated_i, ] <- 0
  est
}

## Build a fitted model. Every model predicts
##   mu + bu[u] + bi[i] + Q[i, ] . U[u, ]
## so predict() and recommend() only need these pieces; model-specific
## parameters go in `extra`.
new_cf_model <- function(class, d, mu, bu, bi, Q, U, extra, columns, call,
                         time, label) {
  names(bu) <- d$users
  names(bi) <- d$items
  dimnames(U) <- list(d$users, NULL)
  dimnames(Q) <- list(d$items, NULL)
  pred <- mu + bu[d$u] + bi[d$i] + rowdot(Q, d$i, U, d$u)
  structure(
    c(list(mu = mu, bu = bu, bi = bi, Q = Q, U = U),
      extra,
      list(users = d$users, items = d$items,
           user_ids = d$user_ids, item_ids = d$item_ids,
           train = list(u = d$u, i = d$i, r = d$r),
           rated = rated_matrix(d$i, d$u, d$ni, d$nu),
           rating_range = range(d$r), columns = columns,
           train_rmse = sqrt(mean((d$r - pred)^2)),
           n_ratings = length(d$r), time = time, label = label, call = call)),
    class = c(class, "cf_model"))
}

## Items x users pattern matrix of training ratings, for top-N exclusion.
rated_matrix <- function(i, u, ni, nu) {
  methods::as(sparseMatrix(i = i, j = u, dims = c(ni, nu)), "CsparseMatrix")
}

## CSR-style access to N(u): returns list(p, idx) so that the items of user u
## are idx[(p[u] + 1):p[u + 1]].
user_index <- function(M) {
  tM <- methods::as(t(M), "CsparseMatrix")
  list(p = tM@p, idx = tM@i + 1L)
}

## Template sparse residual matrix with the same pattern as the ratings;
## E@x <- e[perm] fills it without re-sorting.
residual_template <- function(d) {
  E <- sparseMatrix(i = d$u, j = d$i, x = seq_along(d$u), dims = c(d$nu, d$ni))
  list(E = E, perm = as.integer(E@x))
}

## Wrap an objective closure for stats::optim(L-BFGS-B).
run_lbfgs <- function(obj, theta0, maxit, verbose) {
  opt <- stats::optim(theta0,
                      fn = function(th) obj$fg(th)$f,
                      gr = function(th) obj$fg(th)$g,
                      method = "L-BFGS-B",
                      control = list(maxit = maxit, trace = as.integer(verbose),
                                     REPORT = 10L))
  if (!is.finite(opt$value)) {
    stop("L-BFGS-B failed: the objective is not finite.", call. = FALSE)
  }
  if (opt$convergence %in% c(51L, 52L)) {
    warning(sprintf("L-BFGS-B reported a problem (code %d): %s",
                    opt$convergence, opt$message), call. = FALSE)
  }
  p <- obj$unpack(opt$par)
  p$convergence <- list(code = opt$convergence, message = opt$message,
                        iterations = opt$counts[["function"]],
                        objective = opt$value)
  p
}
