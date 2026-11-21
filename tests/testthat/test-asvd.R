set.seed(42)
d <- simulate_ratings(n_users = 150, n_items = 80, n_ratings = 3000)
test_idx <- 1:300
train <- d[-test_idx, ]
test <- d[test_idx, ]
rmse <- function(fit, x) sqrt(mean((predict(fit, x) - x$rating)^2))
mean_rmse <- sqrt(mean((test$rating - mean(train$rating))^2))

test_that("baseline predictor solves the regularised least squares problem", {
  b <- baseline_predictor(train, lambda = c(user = 3, item = 2))
  u <- match(as.character(train$user), b$users)
  i <- match(as.character(train$item), b$items)
  e <- train$rating - b$mu - b$bu[u] - b$bi[i]
  # stationarity: sum of residuals per user/item equals lambda * bias
  expect_equal(as.vector(tapply(e, u, sum)), unname(3 * b$bu), tolerance = 1e-6)
  expect_equal(as.vector(tapply(e, i, sum)), unname(2 * b$bi), tolerance = 1e-6)
  expect_lt(rmse(b, test), mean_rmse)
  expect_equal(ncol(b$U), 0L)
  expect_equal(nrow(recommend(b, users = "1", n = 3)), 3)
  expect_error(baseline_predictor(train, lambda = 5), "c\\(user")
})

test_that("ASVD analytic gradient matches finite differences", {
  set.seed(3)
  tiny <- simulate_ratings(n_users = 8, n_items = 6, n_ratings = 25)
  imp <- data.frame(user = c(1, 2, 9), item = c(7, 1, 2))
  prep <- svdpp:::prepare_ratings(tiny, "user", "item", "rating", imp)
  set.seed(2)
  w <- rnorm(length(prep$r))
  obj <- svdpp:::asvd_objective(prep, w, k = 2, lambda = 0.3)
  th <- rnorm(obj$n_par, 0, 0.3)
  g <- obj$fg(th)$g
  h <- 1e-6
  num <- vapply(seq_along(th), function(j) {
    e <- replace(numeric(length(th)), j, h)
    (obj$fg(th + e)$f - obj$fg(th - e)$f) / (2 * h)
  }, numeric(1))
  expect_equal(g, num, tolerance = 1e-6)
})

test_that("vectorised ASVD SGD equals a plain per-rating loop", {
  set.seed(5)
  tiny <- simulate_ratings(n_users = 10, n_items = 8, n_ratings = 40)
  set.seed(7)
  fit <- asvd(tiny, factors = 2, epochs = 2, lr = 0.05, lambda = 0.1,
              shuffle = FALSE)
  # Reference: same algorithm written one rating at a time.
  prep <- svdpp:::prepare_ratings(tiny, "user", "item", "rating")
  base <- svdpp:::fit_baseline(prep, c(user = 15, item = 10))
  nr <- tabulate(prep$u, prep$nu)
  w <- (prep$r - base$mu - base$bu[prep$u] - base$bi[prep$i]) / sqrt(nr[prep$u])
  k <- 2; lr <- 0.05; lam <- 0.1
  set.seed(7)
  Q <- matrix(rnorm(prep$ni * k, 0, .1), prep$ni)
  X <- matrix(rnorm(prep$ni * k, 0, .1), prep$ni)
  Y <- matrix(rnorm(prep$ni * k, 0, .1), prep$ni)
  bu <- numeric(prep$nu); bi <- numeric(prep$ni); mu <- mean(prep$r)
  for (ep in 1:2) for (uu in seq_len(prep$nu)) {
    rows <- which(prep$u == uu)
    rows <- rows[order(rows)]
    Ru <- prep$i[rows]
    Nu <- which(prep$N[uu, ] > 0)
    Uu <- colSums(w[rows] * X[Ru, , drop = FALSE]) +
      colSums(Y[Nu, , drop = FALSE]) / sqrt(length(Nu))
    g <- numeric(k)
    for (t in seq_along(rows)) {
      ii <- Ru[t]
      e <- prep$r[rows[t]] - (mu + bu[uu] + bi[ii] + sum(Q[ii, ] * Uu))
      bu[uu] <- bu[uu] + lr * (e - lam * bu[uu])
      bi[ii] <- bi[ii] + lr * (e - lam * bi[ii])
      g <- g + e * Q[ii, ]
      Q[ii, ] <- Q[ii, ] + lr * (e * Uu - lam * Q[ii, ])
    }
    X[Ru, ] <- X[Ru, , drop = FALSE] + lr * (outer(w[rows], g) - lam * X[Ru, , drop = FALSE])
    Y[Nu, ] <- Y[Nu, , drop = FALSE] +
      lr * (matrix(g / sqrt(length(Nu)), length(Nu), k, byrow = TRUE) - lam * Y[Nu, , drop = FALSE])
  }
  expect_equal(unname(fit$Q), Q, tolerance = 1e-10)
  expect_equal(unname(fit$X), X, tolerance = 1e-10)
  expect_equal(unname(fit$Y), Y, tolerance = 1e-10)
  expect_equal(unname(fit$bu), bu, tolerance = 1e-10)
})

test_that("ASVD fits beat the global mean", {
  set.seed(1)
  f1 <- asvd(train, factors = 3, lr = 0.01)
  f2 <- asvd(train, factors = 3, method = "lbfgs", lambda = 5)
  expect_s3_class(f1, c("asvd", "cf_model"))
  expect_lt(rmse(f1, test), 0.9 * mean_rmse)
  expect_lt(rmse(f2, test), 0.9 * mean_rmse)
  expect_equal(f2$convergence$code, 0L)
  expect_lt(f1$history[30], f1$history[1])
  set.seed(1)
  q1 <- asvd(train, factors = 3, epochs = 2)$Q
  set.seed(1)
  q2 <- asvd(train, factors = 3, epochs = 2)$Q
  expect_identical(q1, q2)
})

test_that("fold_in reproduces fitted users and adds new ones", {
  set.seed(1)
  f <- asvd(train, factors = 3, lr = 0.01, epochs = 5)
  # Folding in an existing user's own ratings gives back their U.
  one <- train[train$user == 1, ]
  g <- fold_in(f, one)
  expect_equal(g$U["1", ], f$U["1", ], tolerance = 1e-6)
  expect_equal(g$bu, f$bu)

  # A brand-new user is added and gets personalised predictions.
  newu <- transform(one, user = "new")
  h <- fold_in(f, newu)
  expect_true("new" %in% h$users)
  expect_equal(h$U["new", ], f$U["1", ], tolerance = 1e-6)
  expect_equal(predict(h, data.frame(user = "new", item = "5")) -
                 h$bu[["new"]],
               predict(f, data.frame(user = "1", item = "5")) - f$bu[["1"]],
               tolerance = 1e-6)
  rec <- recommend(h, users = "new", n = 5)
  expect_false(any(rec$item %in% as.character(one$item)))

  expect_warning(fold_in(f, data.frame(user = "x", item = "nope", rating = 3)),
                 "unknown")
})
