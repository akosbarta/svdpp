set.seed(42)
d <- simulate_ratings(n_users = 150, n_items = 80, n_ratings = 3000)
test_idx <- 1:300
train <- d[-test_idx, ]
test <- d[test_idx, ]
rmse <- function(fit, x) sqrt(mean((predict(fit, x) - x$rating)^2))
mean_rmse <- sqrt(mean((test$rating - mean(train$rating))^2))

test_that("analytic gradient matches finite differences", {
  set.seed(3)
  tiny <- simulate_ratings(n_users = 8, n_items = 6, n_ratings = 25)
  imp <- data.frame(user = c(1, 2, 9), item = c(7, 1, 2)) # new user and item
  prep <- svdpp:::prepare_ratings(tiny, "user", "item", "rating", imp)
  obj <- svdpp:::svdpp_objective(prep, k = 2, lambda = 0.3)
  set.seed(1)
  th <- rnorm(obj$n_par, 0, 0.3)
  g <- obj$fg(th)$g
  h <- 1e-6
  num <- vapply(seq_along(th), function(j) {
    e <- replace(numeric(length(th)), j, h)
    (obj$fg(th + e)$f - obj$fg(th - e)$f) / (2 * h)
  }, numeric(1))
  expect_equal(g, num, tolerance = 1e-6)
})

test_that("SGD fit beats the global mean and is reproducible", {
  set.seed(1)
  f1 <- svdpp(train, factors = 3)
  set.seed(1)
  f2 <- svdpp(train, factors = 3)
  expect_s3_class(f1, "svdpp")
  expect_identical(f1$Q, f2$Q)
  expect_lt(rmse(f1, test), 0.9 * mean_rmse)
  expect_length(f1$history, 20)
  expect_lt(f1$history[20], f1$history[1])
})

test_that("L-BFGS fit beats the global mean", {
  set.seed(1)
  f <- svdpp(train, factors = 3, method = "lbfgs", lambda = 5)
  expect_lt(rmse(f, test), 0.9 * mean_rmse)
  expect_equal(f$convergence$code, 0L)
})

test_that("predict handles unknown users and items, clipping and ids", {
  set.seed(1)
  f <- svdpp(train, factors = 3, epochs = 5)
  nd <- data.frame(user = c("1", "nobody", "1"), item = c("2", "3", "nothing"))
  p <- predict(f, nd, clip = FALSE)
  expect_length(p, 3)
  expect_equal(p[2], f$mu + unname(f$bi["3"]))
  expect_equal(p[3], f$mu + unname(f$bu["1"]))
  expect_true(all(predict(f, test) >= 1 & predict(f, test) <= 5))
})

test_that("character ids and custom column names work", {
  x <- data.frame(who = paste0("u", train$user), what = paste0("i", train$item),
                  stars = train$rating)
  f <- svdpp(x, user = "who", item = "what", rating = "stars", factors = 2,
             epochs = 3)
  p <- predict(f, data.frame(who = "u1", what = "i1"))
  expect_true(is.finite(p))
})

test_that("implicit feedback adds users and items", {
  imp <- data.frame(user = c(1, 1, 9999), item = c(8888, 2, 1))
  set.seed(1)
  f <- svdpp(train, implicit = imp, factors = 3, epochs = 3)
  expect_true("9999" %in% f$users)
  expect_true("8888" %in% f$items)
  expect_equal(unname(f$P["9999", ]), c(0, 0, 0))
  expect_equal(unname(f$Q["8888", ]), c(0, 0, 0))
  # implicit-only user is predicted from the implicit term, not just mu
  p <- predict(f, data.frame(user = "9999", item = "5"), clip = FALSE)
  expect_false(isTRUE(all.equal(p, f$mu + unname(f$bi["5"]))))
})

test_that("input validation", {
  expect_error(svdpp(as.matrix(train)), "data frame")
  expect_error(svdpp(train, user = "x"), "not found")
  expect_error(svdpp(rbind(train, train[1, ])), "more than one rating")
  bad <- train
  bad$rating[1] <- NA
  expect_error(svdpp(bad), "missing values")
  expect_error(svdpp(train, factors = 0), "`factors` must be")
})

test_that("recommend returns top-n unrated items", {
  set.seed(1)
  f <- svdpp(train, factors = 3, epochs = 5)
  rec <- recommend(f, users = c("1", "2"), n = 5)
  expect_equal(nrow(rec), 10)
  expect_named(rec, c("user", "item", "score", "rank"))
  rated1 <- as.character(train$item[train$user == 1])
  expect_false(any(rec$item[rec$user == "1"] %in% rated1))
  expect_true(all(diff(rec$score[rec$user == "1"]) <= 0))
  expect_equal(rec$score[1], predict(f, rec[1, ], clip = FALSE))
  expect_warning(r2 <- recommend(f, users = c("1", "zzz"), n = 2), "unknown")
  expect_equal(nrow(r2), 2)
})
