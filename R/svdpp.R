#' Fit an SVD++ recommender model
#'
#' Fits the SVD++ model of Koren (2008, Sec. 4.2). The predicted rating of
#' user \eqn{u} for item \eqn{i} is
#' \deqn{\hat r_{ui} = \mu + b_u + b_i + q_i^\top \Big(p_u + |N(u)|^{-1/2}
#'   \sum_{j \in N(u)} y_j\Big),}
#' where \eqn{N(u)} is the set of items with implicit feedback from \eqn{u}.
#' By default \eqn{N(u)} is the set of items \eqn{u} rated; extra implicit
#' feedback (for example views or purchases without a rating) can be added
#' with `implicit`.
#'
#' @section Estimators:
#' * `method = "sgd"`: stochastic gradient descent as in the paper, one rating
#'   at a time. Ratings are processed user by user: the implicit term is
#'   computed once per user and the \eqn{y_j} are updated once at the end of
#'   that user's ratings, with the gradient accumulated over them. This makes
#'   the cost per epoch linear in the number of ratings, whereas recomputing
#'   the implicit sum for every rating costs \eqn{\sum_u |R(u)||N(u)|}.
#'   `lambda` is the per-update regularisation (default 0.02).
#' * `method = "lbfgs"`: minimises the full regularised squared error
#'   \eqn{\sum (r_{ui} - \hat r_{ui})^2 + \lambda \|\theta\|^2} with
#'   [stats::optim()] (`"L-BFGS-B"`), computing the gradient with sparse
#'   matrix products. Because the penalty is on the summed loss, `lambda` is
#'   on a different scale from the SGD one (default 20); choose it on
#'   validation data.
#'
#' Both estimators start from random factor values and SGD visits users in
#' random order, so call [set.seed()] before fitting for reproducible
#' results.
#'
#' Users with implicit feedback but no ratings get \eqn{p_u = 0}, so their
#' predictions use only the implicit term. Items with no ratings get
#' \eqn{q_i = 0}, \eqn{b_i = 0}.
#'
#' @section Input formats:
#' `data` can be
#' * a data frame in long format, one row per rating, with user, item and
#'   rating columns (named by `user`, `item` and `rating`). This is the
#'   natural format for rating data, which is mostly missing;
#' * a users x items `matrix` (or dense \pkg{Matrix}) with `NA` for missing
#'   ratings; or
#' * a users x items sparse matrix from the \pkg{Matrix} package (for
#'   example a `dgCMatrix`), in which the stored entries are the ratings and
#'   entries that are not stored are missing (not zero).
#'
#' For a matrix, the row and column names are the user and item ids, or the
#' row and column numbers if there are none. Every row and column is a user
#' or item of the model, in matrix order, including those without ratings
#' (they get zero biases and factors). The fitted model then predicts for a
#' data frame with columns `user` and `item` (see [predict.cf_model()]).
#'
#' `implicit` can be a data frame with the `user` and `item` columns or a
#' users x items matrix. If `data` is a matrix and `implicit` has no row
#' (column) names, it must have the same rows (columns) as `data` and takes
#' its names.
#'
#' @param data Ratings: a data frame with one row per user-item pair, or a
#'   users x items matrix. See *Input formats*.
#' @param factors Number of latent factors.
#' @param method Estimator, `"sgd"` or `"lbfgs"`. See *Estimators*.
#' @param lambda Regularisation strength. `NULL` uses the default for
#'   `method`.
#' @param implicit Optional extra implicit feedback, added to \eqn{N(u)}:
#'   a data frame with the `user` and `item` columns, or a users x items
#'   logical or numeric matrix whose non-zero (`TRUE`) entries mark feedback.
#' @param lr SGD learning rate.
#' @param epochs Number of SGD passes over the data.
#' @param shuffle Visit users in a random order in each SGD epoch?
#' @param maxit Maximum number of L-BFGS-B iterations.
#' @param init_sd Standard deviation of the normal initialisation of the
#'   factor matrices. Biases start at zero.
#' @param user,item,rating Names of the user, item and rating columns of a
#'   data frame `data`. User and item ids can be of any type; they are
#'   matched as character strings. For a matrix `data` they are the column
#'   names the fitted model expects in `newdata` when predicting.
#' @param verbose Print progress?
#'
#' @return An object of class `"svdpp"` (and `"cf_model"`), a list with the
#'   global mean `mu`, biases `bu` and `bi`, factor matrices `P`, `Q` and
#'   `Y`, the user representation `U` (\eqn{p_u} plus the implicit term), the
#'   training RMSE, and the user and item ids.
#'
#' @references
#' Koren, Y. (2008). Factorization meets the neighborhood: a multifaceted
#' collaborative filtering model. *KDD '08*, 426--434.
#' \doi{10.1145/1401890.1401944}
#'
#' @seealso [asvd()], [baseline_predictor()], [predict.cf_model()],
#'   [recommend()], [simulate_ratings()]
#' @examples
#' set.seed(1)
#' d <- simulate_ratings(n_users = 200, n_items = 100, n_ratings = 4000)
#' test <- d[1:400, ]
#' fit <- svdpp(d[-(1:400), ], factors = 5)
#' fit
#' sqrt(mean((predict(fit, test) - test$rating)^2))
#'
#' fit2 <- svdpp(d[-(1:400), ], 5, "lbfgs")
#' sqrt(mean((predict(fit2, test) - test$rating)^2))
#'
#' # A users x items matrix with NA for missing ratings
#' m <- matrix(NA, 4, 3, dimnames = list(paste0("u", 1:4), c("a", "b", "c")))
#' m[cbind(c(1, 1, 2, 3, 3, 4), c(1, 2, 2, 1, 3, 3))] <- c(5, 3, 4, 2, 5, 1)
#' fit3 <- svdpp(m, factors = 2, epochs = 50)
#' predict(fit3, data.frame(user = "u2", item = "c"))
#' @export
svdpp <- function(data, factors = 20L, method = c("sgd", "lbfgs"),
                  lambda = NULL, implicit = NULL,
                  lr = 0.007, epochs = 20L, shuffle = TRUE,
                  maxit = 300L, init_sd = 0.1,
                  user = "user", item = "item", rating = "rating",
                  verbose = FALSE) {
  factors <- check_factors(factors)
  method <- match.arg(method)
  lambda <- check_lambda(if (is.null(lambda)) {
    if (method == "sgd") 0.02 else 20
  } else lambda)

  ctl <- check_control(lr, epochs, shuffle, maxit, init_sd, verbose)
  lr <- ctl$lr
  epochs <- ctl$epochs
  shuffle <- ctl$shuffle
  maxit <- ctl$maxit
  init_sd <- ctl$init_sd
  verbose <- ctl$verbose

  d <- prepare_ratings(data, user, item, rating, implicit)

  t0 <- proc.time()[["elapsed"]]
  est <- if (method == "sgd") {
    sgd_svdpp(d, factors, lambda, lr, epochs, shuffle, init_sd,
              verbose)
  } else {
    lbfgs_svdpp(d, factors, lambda, maxit, init_sd, verbose)
  }
  est <- zero_unrated(est, d)
  U <- est$P + as.matrix(normalise_implicit(d$N) %*% est$Y)
  dimnames(est$P) <- list(d$users, NULL)
  dimnames(est$Y) <- list(d$items, NULL)

  new_cf_model(
    "svdpp", d, est$mu, est$bu, est$bi, est$Q, U,
    extra = list(
      P = est$P, Y = est$Y,
      method = method, factors = factors, lambda = lambda,
      control = if (method == "sgd") {
        list(lr = lr, epochs = epochs, shuffle = shuffle, init_sd = init_sd)
      } else {
        list(maxit = maxit, init_sd = init_sd)
      },
      history = est$history, convergence = est$convergence),
    columns = c(user = user, item = item, rating = rating),
    call = match.call(), time = proc.time()[["elapsed"]] - t0,
    label = "SVD++ (Koren 2008)")
}

## SGD with per-user grouping of the implicit term.
sgd_svdpp <- function(d, k, lambda, lr, epochs, shuffle, init_sd, verbose) {
  mu <- mean(d$r)
  bu <- numeric(d$nu)
  bi <- numeric(d$ni)
  P <- matrix(stats::rnorm(d$nu * k, 0, init_sd), d$nu, k)
  Q <- matrix(stats::rnorm(d$ni * k, 0, init_sd), d$ni, k)
  Y <- matrix(stats::rnorm(d$ni * k, 0, init_sd), d$ni, k)

  ord <- order(d$u)
  item <- d$i[ord]
  r <- d$r[ord]
  cnt <- tabulate(d$u, d$nu)
  last <- cumsum(cnt)
  first <- last - cnt + 1L
  users <- which(cnt > 0L)
  NI <- user_index(d$N)
  np <- NI$p
  nidx <- NI$idx

  history <- numeric(epochs)
  for (ep in seq_len(epochs)) {
    sse <- 0
    visit <- if (shuffle) users[sample.int(length(users))] else users
    for (uu in visit) {
      Nu <- nidx[(np[uu] + 1L):np[uu + 1L]]
      s <- 1 / sqrt(length(Nu))
      YN <- Y[Nu, , drop = FALSE]
      z <- s * colSums(YN)
      gy <- numeric(k)
      pu <- P[uu, ]
      b_u <- bu[uu]
      for (t in first[uu]:last[uu]) {
        ii <- item[t]
        qi <- Q[ii, ]
        e <- r[t] - (mu + b_u + bi[ii] + sum(qi * (pu + z)))
        sse <- sse + e * e
        b_u <- b_u + lr * (e - lambda * b_u)
        bi[ii] <- bi[ii] + lr * (e - lambda * bi[ii])
        Q[ii, ] <- qi + lr * (e * (pu + z) - lambda * qi)
        pu <- pu + lr * (e * qi - lambda * pu)
        gy <- gy + e * s * qi
      }
      bu[uu] <- b_u
      P[uu, ] <- pu
      Y[Nu, ] <- YN + lr * (rep(gy, each = length(Nu)) - lambda * YN)
    }
    history[ep] <- sqrt(sse / length(r))
    check_diverged(history[ep], ep)
    if (verbose) message(sprintf("epoch %3d  train RMSE %.4f", ep, history[ep]))
  }
  list(mu = mu, bu = bu, bi = bi, P = P, Q = Q, Y = Y,
       history = history, convergence = NULL)
}

## Objective and gradient of the full-batch SVD++ problem as one closure.
## theta = c(bu, bi, P, Q, Y). fg() returns list(f, g) and caches the last
## point, because optim() asks for f and g separately at the same theta.
svdpp_objective <- function(d, k, lambda) {
  u <- d$u
  i <- d$i
  r <- d$r
  nu <- d$nu
  ni <- d$ni
  mu <- mean(r)
  Nn <- normalise_implicit(d$N)
  tNn <- methods::as(t(Nn), "CsparseMatrix")
  tmpl <- residual_template(d)
  E <- tmpl$E
  perm <- tmpl$perm
  sizes <- c(nu, ni, nu * k, ni * k, ni * k)
  ends <- cumsum(sizes)
  starts <- ends - sizes + 1L
  ## Cache of the last evaluated point: optim() requests f and g
  ## separately at the same theta.
  cache <- new.env(parent = emptyenv())

  unpack <- function(theta) {
    list(mu = mu,
         bu = theta[starts[1]:ends[1]],
         bi = theta[starts[2]:ends[2]],
         P = matrix(theta[starts[3]:ends[3]], nu, k),
         Q = matrix(theta[starts[4]:ends[4]], ni, k),
         Y = matrix(theta[starts[5]:ends[5]], ni, k))
  }

  fg <- function(theta) {
    if (identical(theta, cache$theta)) return(cache$out)
    p <- unpack(theta)
    U <- p$P + as.matrix(Nn %*% p$Y)
    e <- r - (mu + p$bu[u] + p$bi[i] + rowdot(p$Q, i, U, u))
    E@x <- e[perm]
    EQ <- as.matrix(E %*% p$Q)
    g <- -2 * c(rowSums(E), colSums(E), EQ,
                as.matrix(crossprod(E, U)), as.matrix(tNn %*% EQ)) +
      2 * lambda * theta
    cache$theta <- theta
    cache$out <- list(f = sum(e^2) + lambda * sum(theta^2), g = g)
    cache$out
  }
  list(fg = fg, unpack = unpack, n_par = sum(sizes))
}

lbfgs_svdpp <- function(d, k, lambda, maxit, init_sd, verbose) {
  obj <- svdpp_objective(d, k, lambda)
  theta0 <- c(numeric(d$nu + d$ni),
              stats::rnorm((d$nu + 2 * d$ni) * k, 0, init_sd))
  run_lbfgs(obj, theta0, maxit, verbose)
}
