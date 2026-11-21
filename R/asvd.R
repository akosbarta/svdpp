#' Fit an Asymmetric-SVD recommender model
#'
#' Fits Asymmetric-SVD (Koren 2008, Sec. 4.1, Eq. 13). Users have no factor
#' vector of their own; they are represented through the items they rated
#' and the items they gave implicit feedback on:
#' \deqn{\hat r_{ui} = \mu + b_u + b_i + q_i^\top \Big(|R(u)|^{-1/2}
#'   \sum_{j \in R(u)} (r_{uj} - b_{uj}) x_j + |N(u)|^{-1/2}
#'   \sum_{j \in N(u)} y_j\Big).}
#' The offsets \eqn{b_{uj} = \mu + b_u + b_j} inside the sum are fixed
#' baseline estimates (see [baseline_predictor()]); the \eqn{b_u, b_i}
#' outside it are learned with the other parameters.
#'
#' Because users are not parameterised, a new user can be scored as soon as
#' their ratings are known, without refitting: see [fold_in()]. Predictions
#' are also explainable as a sum of contributions of the user's past
#' ratings.
#'
#' @section Estimators:
#' * `method = "sgd"`: gradient descent as in the paper, user by user. Since
#'   the user representation does not change while that user's ratings are
#'   processed, all the updates for one user are vectorised. The only
#'   sequential part, the \eqn{b_u} update, is a linear recurrence computed
#'   with [stats::filter()]. The \eqn{x_j} and \eqn{y_j} are updated once per
#'   user with the gradient summed over the user's ratings, and shrunk once
#'   per user rather than once per rating. Defaults follow the paper's
#'   Netflix settings: 30 epochs, `lr = 0.002`, `lambda = 0.04`.
#' * `method = "lbfgs"`: minimises \eqn{\sum (r_{ui} - \hat r_{ui})^2 +
#'   \lambda \|\theta\|^2} with L-BFGS-B and a sparse-matrix gradient, as in
#'   [svdpp()]. Default `lambda = 20`; choose it on validation data.
#'
#' @inheritParams svdpp
#' @param lr SGD learning rate.
#' @param epochs Number of SGD passes over the data.
#' @param baseline_lambda Regularisation of the fixed baseline estimates
#'   \eqn{b_{uj}}, passed to [baseline_predictor()].
#'
#' @return An object of class `"asvd"` (and `"cf_model"`), a list with the
#'   global mean `mu`, learned biases `bu` and `bi`, factor matrices `Q`,
#'   `X` and `Y`, the user representation `U`, the fixed baseline
#'   (`baseline`), the training RMSE, and the user and item ids.
#'
#' @references
#' Koren, Y. (2008). Factorization meets the neighborhood: a multifaceted
#' collaborative filtering model. *KDD '08*, 426--434.
#' \doi{10.1145/1401890.1401944}
#'
#' Paterek, A. (2007). Improving regularized singular value decomposition
#' for collaborative filtering. *Proceedings of KDD Cup and Workshop*.
#'
#' @seealso [fold_in()], [svdpp()], [predict.cf_model()], [recommend()]
#' @examples
#' set.seed(1)
#' d <- simulate_ratings(n_users = 200, n_items = 100, n_ratings = 4000)
#' test <- d[1:400, ]
#' fit <- asvd(d[-(1:400), ], factors = 5, lr = 0.01)
#' fit
#' sqrt(mean((predict(fit, test) - test$rating)^2))
#' @export
asvd <- function(data, user = "user", item = "item", rating = "rating",
                 implicit = NULL, factors = 20L,
                 method = c("sgd", "lbfgs"), lambda = NULL,
                 lr = 0.002, epochs = 30L, shuffle = TRUE,
                 maxit = 300L, init_sd = 0.1,
                 baseline_lambda = c(user = 15, item = 10),
                 verbose = FALSE) {
  method <- match.arg(method)
  factors <- check_factors(factors)
  lambda <- check_lambda(if (is.null(lambda)) {
    if (method == "sgd") 0.04 else 20
  } else lambda)

  ctl <- check_control(lr, epochs, shuffle, maxit, init_sd, verbose)
  lr <- ctl$lr
  epochs <- ctl$epochs
  shuffle <- ctl$shuffle
  maxit <- ctl$maxit
  init_sd <- ctl$init_sd
  verbose <- ctl$verbose
  baseline_lambda <- check_baseline_lambda(baseline_lambda)

  d <- prepare_ratings(data, user, item, rating, implicit)

  t0 <- proc.time()[["elapsed"]]
  base <- fit_baseline(d, baseline_lambda)
  ## Fixed offsets r_uj - b_uj, scaled by |R(u)|^{-1/2}.
  n_r <- tabulate(d$u, d$nu)
  w <- (d$r - base$mu - base$bu[d$u] - base$bi[d$i]) / sqrt(n_r[d$u])
  est <- if (method == "sgd") {
    sgd_asvd(d, w, factors, lambda, lr, epochs, shuffle, init_sd,
             verbose)
  } else {
    lbfgs_asvd(d, w, factors, lambda, maxit, init_sd, verbose)
  }
  est <- zero_unrated(est, d)
  Rw <- sparseMatrix(i = d$u, j = d$i, x = w, dims = c(d$nu, d$ni))
  U <- as.matrix(Rw %*% est$X + normalise_implicit(d$N) %*% est$Y)
  dimnames(est$X) <- dimnames(est$Y) <- list(d$items, NULL)
  names(base$bu) <- d$users
  names(base$bi) <- d$items

  new_cf_model(
    "asvd", d, est$mu, est$bu, est$bi, est$Q, U,
    extra = list(
      X = est$X, Y = est$Y,
      baseline = list(mu = base$mu, bu = base$bu, bi = base$bi,
                      lambda = baseline_lambda),
      method = method, factors = factors, lambda = lambda,
      control = if (method == "sgd") {
        list(lr = lr, epochs = epochs, shuffle = shuffle, init_sd = init_sd)
      } else {
        list(maxit = maxit, init_sd = init_sd)
      },
      history = est$history, convergence = est$convergence),
    columns = c(user = user, item = item, rating = rating),
    call = match.call(), time = proc.time()[["elapsed"]] - t0,
    label = "Asymmetric-SVD (Koren 2008)")
}

## Per-user vectorised SGD. `w` holds |R(u)|^{-1/2} (r_uj - b_uj) per rating.
sgd_asvd <- function(d, w, k, lambda, lr, epochs, shuffle, init_sd, verbose) {
  mu <- mean(d$r)
  bu <- numeric(d$nu)
  bi <- numeric(d$ni)
  Q <- matrix(stats::rnorm(d$ni * k, 0, init_sd), d$ni, k)
  X <- matrix(stats::rnorm(d$ni * k, 0, init_sd), d$ni, k)
  Y <- matrix(stats::rnorm(d$ni * k, 0, init_sd), d$ni, k)

  ord <- order(d$u)
  item <- d$i[ord]
  r <- d$r[ord]
  w <- w[ord]
  cnt <- tabulate(d$u, d$nu)
  last <- cumsum(cnt)
  first <- last - cnt + 1L
  users <- which(cnt > 0L)
  NI <- user_index(d$N)
  np <- NI$p
  nidx <- NI$idx
  ## b_u after a rating: b <- b + lr * (e - lambda * b), e = s - b, so
  ## b_next = rho * b + lr * s.
  rho <- 1 - lr * lambda - lr
  decay <- 1 - lr * lambda

  history <- numeric(epochs)
  for (ep in seq_len(epochs)) {
    sse <- 0
    visit <- if (shuffle) users[sample.int(length(users))] else users
    for (uu in visit) {
      idx <- first[uu]:last[uu]
      Ru <- item[idx]
      wu <- w[idx]
      Nu <- nidx[(np[uu] + 1L):np[uu + 1L]]
      sN <- 1 / sqrt(length(Nu))
      XR <- X[Ru, , drop = FALSE]
      YN <- Y[Nu, , drop = FALSE]
      QR <- Q[Ru, , drop = FALSE]
      Uu <- colSums(wu * XR) + sN * colSums(YN)
      s <- r[idx] - mu - bi[Ru] - drop(QR %*% Uu)
      b_after <- as.numeric(stats::filter(lr * s, rho, method = "recursive",
                                          init = bu[uu]))
      e <- s - c(bu[uu], b_after)[seq_along(s)]
      bu[uu] <- b_after[length(b_after)]
      sse <- sse + sum(e * e)
      bi[Ru] <- decay * bi[Ru] + lr * e
      g <- colSums(e * QR)
      Q[Ru, ] <- decay * QR + lr * outer(e, Uu)
      X[Ru, ] <- decay * XR + lr * outer(wu, g)
      Y[Nu, ] <- decay * YN + rep(lr * sN * g, each = length(Nu))
    }
    history[ep] <- sqrt(sse / length(r))
    check_diverged(history[ep], ep)
    if (verbose) message(sprintf("epoch %3d  train RMSE %.4f", ep, history[ep]))
  }
  list(mu = mu, bu = bu, bi = bi, Q = Q, X = X, Y = Y,
       history = history, convergence = NULL)
}

## Full-batch objective; theta = c(bu, bi, Q, X, Y).
asvd_objective <- function(d, w, k, lambda) {
  u <- d$u
  i <- d$i
  r <- d$r
  nu <- d$nu
  ni <- d$ni
  mu <- mean(r)
  Rw <- sparseMatrix(i = u, j = i, x = w, dims = c(nu, ni))
  tRw <- methods::as(t(Rw), "CsparseMatrix")
  Nn <- normalise_implicit(d$N)
  tNn <- methods::as(t(Nn), "CsparseMatrix")
  tmpl <- residual_template(d)
  E <- tmpl$E
  perm <- tmpl$perm
  sizes <- c(nu, ni, ni * k, ni * k, ni * k)
  ends <- cumsum(sizes)
  starts <- ends - sizes + 1L
  ## Cache of the last evaluated point: optim() requests f and g
  ## separately at the same theta.
  cache <- new.env(parent = emptyenv())

  unpack <- function(theta) {
    list(mu = mu,
         bu = theta[starts[1]:ends[1]],
         bi = theta[starts[2]:ends[2]],
         Q = matrix(theta[starts[3]:ends[3]], ni, k),
         X = matrix(theta[starts[4]:ends[4]], ni, k),
         Y = matrix(theta[starts[5]:ends[5]], ni, k))
  }

  fg <- function(theta) {
    if (identical(theta, cache$theta)) return(cache$out)
    p <- unpack(theta)
    U <- as.matrix(Rw %*% p$X + Nn %*% p$Y)
    e <- r - (mu + p$bu[u] + p$bi[i] + rowdot(p$Q, i, U, u))
    E@x <- e[perm]
    EQ <- as.matrix(E %*% p$Q)
    g <- -2 * c(rowSums(E), colSums(E), as.matrix(crossprod(E, U)),
                as.matrix(tRw %*% EQ), as.matrix(tNn %*% EQ)) +
      2 * lambda * theta
    cache$theta <- theta
    cache$out <- list(f = sum(e^2) + lambda * sum(theta^2), g = g)
    cache$out
  }
  list(fg = fg, unpack = unpack, n_par = sum(sizes))
}

lbfgs_asvd <- function(d, w, k, lambda, maxit, init_sd, verbose) {
  obj <- asvd_objective(d, w, k, lambda)
  theta0 <- c(numeric(d$nu + d$ni), stats::rnorm(3 * d$ni * k, 0, init_sd))
  run_lbfgs(obj, theta0, maxit, verbose)
}

#' Add or update users without refitting
#'
#' Computes the representation of new (or changed) users from their ratings
#' and implicit feedback, keeping all item parameters fixed. This is
#' possible for [asvd()] models because users are represented only through
#' the items they rated.
#'
#' @param object A fitted model.
#' @param data A data frame with the user, item and rating columns used when
#'   fitting. For each user in `data`, it must hold all of that user's
#'   ratings: it replaces what the model knew about them.
#' @param implicit Optional extra implicit feedback, as in [asvd()].
#' @param ... Unused.
#'
#' @details For each user, the fixed baseline bias is computed from their
#'   ratings with the item baselines held fixed,
#'   \eqn{b_u = \sum_j (r_{uj} - \mu - b_j) / (\lambda_u + |R(u)|)}, and the
#'   user representation follows Eq. 13. New users use this baseline bias as
#'   their \eqn{b_u}; users already in the model keep their fitted
#'   \eqn{b_u}. Ratings of items unknown to the model are ignored with a
#'   warning.
#'
#' @return The model with the users in `data` (and `implicit`) added or
#'   updated.
#' @examples
#' set.seed(1)
#' d <- simulate_ratings()
#' old <- d[d$user != 1, ]
#' fit <- asvd(old, factors = 5, lr = 0.01)
#' fit2 <- fold_in(fit, d[d$user == 1, ])
#' predict(fit2, data.frame(user = 1, item = 1:5))
#' @export
fold_in <- function(object, ...) UseMethod("fold_in")

#' @rdname fold_in
#' @export
fold_in.asvd <- function(object, data, implicit = NULL, ...) {
  cols <- object$columns
  check_columns(data, cols, "data")
  du <- as_id(data[[cols[["user"]]]])
  di <- match(as_id(data[[cols[["item"]]]]), object$items)
  r <- check_rating(data[[cols[["rating"]]]], cols[["rating"]])
  iu <- character()
  ii <- integer()
  if (!is.null(implicit)) {
    check_columns(implicit, cols[c("user", "item")], "implicit")
    iu <- as_id(implicit[[cols[["user"]]]])
    ii <- match(as_id(implicit[[cols[["item"]]]]), object$items)
  }
  unknown <- sum(is.na(di)) + sum(is.na(ii))
  if (unknown) {
    warning(sprintf("%d row(s) with items unknown to the model ignored.",
                    unknown), call. = FALSE)
  }
  ulev <- unique(c(du, iu))
  keep <- !is.na(di)
  u <- match(du[keep], ulev)
  i <- di[keep]
  r <- r[keep]
  if (anyDuplicated((u - 1) * length(object$items) + i)) {
    stop("`data` contains more than one rating for some user-item pair.",
         call. = FALSE)
  }
  ni <- length(object$items)
  nn <- length(ulev)
  b <- object$baseline
  n_r <- tabulate(u, nn)
  bu_base <- group_sum(r - b$mu - b$bi[i], u, nn) / (b$lambda[["user"]] + n_r)
  w <- (r - b$mu - bu_base[u] - b$bi[i]) / sqrt(n_r[u])
  Rw <- sparseMatrix(i = u, j = i, x = w, dims = c(nn, ni))
  ki <- !is.na(ii)
  N <- sparseMatrix(i = c(u, match(iu[ki], ulev)), j = c(i, ii[ki]), x = 1,
                    dims = c(nn, ni))
  N@x[] <- 1
  Unew <- as.matrix(Rw %*% object$X + normalise_implicit(N) %*% object$Y)

  pos <- match(ulev, object$users)
  is_new <- is.na(pos)
  users <- c(object$users, ulev[is_new])
  pos[is_new] <- length(object$users) + seq_len(sum(is_new))

  U <- rbind(object$U, matrix(0, sum(is_new), ncol(object$U)))
  U[pos, ] <- Unew
  rownames(U) <- users
  bu <- c(object$bu, bu_base[is_new])
  names(bu) <- users
  b$bu <- c(b$bu, stats::setNames(bu_base[is_new], ulev[is_new]))
  b$bu[ulev] <- bu_base

  ## Replace the rated sets of the updated users.
  old <- Matrix::summary(object$rated)
  old <- old[!(old$j %in% pos), c("i", "j")]
  object$rated <- rated_matrix(c(old$i, i), c(old$j, pos[u]), ni, length(users))

  object$U <- U
  object$bu <- bu
  object$users <- users
  object$baseline <- b
  object
}

#' @rdname fold_in
#' @export
fold_in.default <- function(object, ...) {
  stop(sprintf(paste0("fold_in() is not available for objects of class '%s'. ",
                      "Only asvd() models can add users without refitting; ",
                      "refit other models with the new ratings."),
               class(object)[1]), call. = FALSE)
}
