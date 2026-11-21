#' @export
print.cf_model <- function(x, ...) {
  cat(x$label, "\n", sep = "")
  cat(sprintf("  %d users, %d items, %d ratings", length(x$users),
              length(x$items), x$n_ratings))
  if (!is.null(x$factors)) cat(sprintf(", %d factors", x$factors))
  cat("\n")
  if (identical(x$method, "sgd")) {
    cat(sprintf("  estimator: SGD, %d epochs, lr = %g, lambda = %g\n",
                x$control$epochs, x$control$lr, x$lambda))
  } else if (identical(x$method, "lbfgs")) {
    cat(sprintf("  estimator: L-BFGS-B, %d iterations (code %d), lambda = %g\n",
                x$convergence$iterations, x$convergence$code, x$lambda))
    if (identical(x$convergence$code, 1L)) {
      cat("  (stopped at `maxit` before converging; increase `maxit` for a tighter fit)\n")
    }
  } else if (inherits(x, "baseline_predictor")) {
    cat(sprintf("  alternating least squares, %d sweeps, lambda: user = %g, item = %g\n",
                x$iterations, x$lambda[["user"]], x$lambda[["item"]]))
  }
  cat(sprintf("  training RMSE: %.4f   fit time: %.1fs\n", x$train_rmse, x$time))
  invisible(x)
}

#' Predict ratings from a fitted model
#'
#' Works for every model in the package: [svdpp()], [asvd()] and
#' [baseline_predictor()].
#'
#' @param object A fitted model.
#' @param newdata A data frame with the user and item columns used when
#'   fitting (other columns are ignored). If omitted, predictions are made
#'   for the ratings the model was fitted on, in the order of `data`.
#' @param clip Clip predictions to the range of the training ratings?
#' @param ... Unused.
#'
#' @details Users or items not seen during fitting (or with missing ids)
#'   contribute nothing beyond
#'   the global mean: their biases and factors are taken as zero. A user seen
#'   only through `implicit` feedback is predicted from the implicit term.
#'   To score new users of an [asvd()] model from their ratings, add them
#'   first with [fold_in()].
#'
#'   `fitted()` and `residuals()` refer to the ratings the model was fitted
#'   on (for a matrix `data`, its entries in the order the matrix stores
#'   them: column-major for a base `matrix` or a `dgCMatrix`). They are not
#'   clipped, so `sqrt(mean(residuals(fit)^2))` is the
#'   training RMSE. `nobs()` is the number of those ratings. After
#'   [fold_in()], they use the updated user representations but still refer
#'   to the original ratings, and the training RMSE shown by `print()` is
#'   recomputed to match.
#'
#' @return `predict()` and `fitted()`: a numeric vector of predicted
#'   ratings, one per row of `newdata` (or per training rating).
#'   `residuals()`: the training ratings minus the fitted values. `nobs()`:
#'   the number of training ratings.
#' @examples
#' set.seed(1)
#' d <- simulate_ratings()
#' fit <- svdpp(d, factors = 5, epochs = 10)
#' predict(fit, data.frame(user = c("1", "2"), item = c("3", "4")))
#' head(fitted(fit))
#' sqrt(mean(residuals(fit)^2)) # training RMSE
#' nobs(fit)
#' @export
predict.cf_model <- function(object, newdata, clip = TRUE, ...) {
  check_flag(clip, "clip")
  if (missing(newdata) || is.null(newdata)) {
    u <- object$train$u
    i <- object$train$i
  } else {
    cols <- object$columns[c("user", "item")]
    check_columns(newdata, cols, "newdata", allow_na = TRUE)
    ## Missing ids are allowed and treated like unknown users/items.
    u <- match(as_id(newdata[[cols[["user"]]]]), object$users,
               incomparables = NA)
    i <- match(as_id(newdata[[cols[["item"]]]]), object$items,
               incomparables = NA)
  }
  ok <- !is.na(u) & !is.na(i)
  pred <- object$mu +
    ifelse(is.na(u), 0, object$bu[u]) +
    ifelse(is.na(i), 0, object$bi[i])
  pred[ok] <- pred[ok] + rowdot(object$Q, i[ok], object$U, u[ok])
  pred <- unname(pred)
  if (clip) {
    pred <- pmin(pmax(pred, object$rating_range[1]), object$rating_range[2])
  }
  pred
}

#' @rdname predict.cf_model
#' @export
fitted.cf_model <- function(object, ...) predict(object, clip = FALSE)

#' @rdname predict.cf_model
#' @export
residuals.cf_model <- function(object, ...) {
  object$train$r - predict(object, clip = FALSE)
}

#' @rdname predict.cf_model
#' @export
nobs.cf_model <- function(object, ...) length(object$train$r)

#' Top-N recommendations
#'
#' @param object A fitted model.
#' @param users Ids of the users to recommend for. Defaults to all users.
#' @param n Number of items per user.
#' @param exclude_rated Leave out items the user rated in the training data?
#' @param ... Passed to methods.
#'
#' @return A data frame with columns `user`, `item`, `score` and `rank`,
#'   sorted by user and rank. Unknown users are dropped with a warning.
#'   `user` has the type of `users` and `item` the type of the item ids in
#'   the training data (for example integer or factor), so the result can be
#'   joined back onto them. For a matrix `data` without names the ids are
#'   the row and column numbers.
#' @examples
#' set.seed(1)
#' d <- simulate_ratings()
#' fit <- svdpp(d, factors = 5, epochs = 10)
#' recommend(fit, users = c("1", "2"), n = 3)
#' @export
recommend <- function(object, ...) UseMethod("recommend")

#' @rdname recommend
#' @export
recommend.cf_model <- function(object, users = object$user_ids, n = 10L,
                               exclude_rated = TRUE, ...) {
  n <- check_number(n, "n", 1, int = TRUE)
  check_flag(exclude_rated, "exclude_rated")
  if (!is.atomic(users) || !is.null(dim(users))) {
    stop("`users` must be a vector of user ids.", call. = FALSE)
  }
  u <- match(as_id(users), object$users, incomparables = NA)
  if (anyNA(u)) {
    warning(sprintf("%d unknown user(s) dropped.", sum(is.na(u))), call. = FALSE)
    users <- users[!is.na(u)]
    u <- u[!is.na(u)]
  }
  n <- min(n, length(object$items))
  rp <- object$rated@p
  ri <- object$rated@i + 1L
  base <- object$mu + object$bi
  top_items <- vector("list", length(u))
  top_scores <- vector("list", length(u))
  ## Score users in chunks so the items x users score matrix stays small.
  chunk <- 512L
  start <- 1L
  while (start <= length(u)) {
    idx <- start:min(length(u), start + chunk - 1L)
    S <- object$Q %*% t(object$U[u[idx], , drop = FALSE]) # items x users
    for (j in seq_along(idx)) {
      uu <- u[idx[j]]
      sc <- base + object$bu[uu] + S[, j]
      if (exclude_rated && rp[uu + 1L] > rp[uu]) {
        sc[ri[(rp[uu] + 1L):rp[uu + 1L]]] <- -Inf
      }
      top <- order(sc, decreasing = TRUE)[seq_len(n)]
      top <- top[is.finite(sc[top])]
      top_items[[idx[j]]] <- top
      top_scores[[idx[j]]] <- unname(sc[top])
    }
    start <- start + chunk
  }
  len <- lengths(top_items)
  top <- as.integer(unlist(top_items, use.names = FALSE))
  data.frame(user = rep(users, len),
             item = object$item_ids[top],
             score = as.numeric(unlist(top_scores, use.names = FALSE)),
             rank = sequence(len),
             stringsAsFactors = FALSE)
}
