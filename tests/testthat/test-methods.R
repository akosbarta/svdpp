set.seed(5)
d <- simulate_ratings(n_users = 50, n_items = 30, n_ratings = 600)

test_that("predict() without newdata, fitted(), residuals() and nobs()", {
  set.seed(1)
  f <- svdpp(d, 3, epochs = 3)
  expect_equal(predict(f), predict(f, d))
  expect_equal(fitted(f), predict(f, d, clip = FALSE))
  expect_equal(residuals(f), d$rating - fitted(f))
  expect_equal(resid(f), residuals(f))
  expect_equal(sqrt(mean(residuals(f)^2)), f$train_rmse)
  expect_identical(nobs(f), nrow(d))
  b <- baseline_predictor(d)
  expect_equal(sqrt(mean(residuals(b)^2)), b$train_rmse)
  expect_equal(nobs(b), nrow(d))
})

test_that("fitted() follows the input order for a shuffled data frame", {
  x <- d[sample(nrow(d)), ]
  set.seed(1)
  f <- asvd(x, 2, epochs = 2)
  expect_equal(fitted(f), predict(f, x, clip = FALSE))
})

test_that("recommend() returns ids in their original type", {
  set.seed(1)
  f <- svdpp(d, 2, epochs = 2)
  r <- recommend(f, n = 2)
  expect_type(r$user, "integer")
  expect_type(r$item, "integer")
  expect_setequal(unique(r$user), unique(d$user))
  expect_true(all(r$item %in% d$item))
  expect_type(recommend(f, users = c(1, 2), n = 2)$user, "double")
  expect_type(recommend(f, users = c("1", "2"), n = 2)$user, "character")

  x <- d
  x$user <- factor(paste0("u", x$user))
  x$item <- factor(paste0("i", x$item))
  set.seed(1)
  g <- svdpp(x, 2, epochs = 2)
  r <- recommend(g, n = 1)
  expect_s3_class(r$user, "factor")
  expect_s3_class(r$item, "factor")
  expect_identical(levels(r$item), levels(x$item))

  x <- d
  x$item <- as.Date("2020-01-01") + x$item
  set.seed(1)
  h <- svdpp(x, 2, epochs = 1)
  expect_s3_class(recommend(h, users = 1, n = 2)$item, "Date")
})

test_that("implicit-only ids keep their type", {
  imp <- data.frame(user = 999L, item = 998L)
  set.seed(1)
  f <- svdpp(d, 2, epochs = 1, implicit = imp)
  expect_type(f$user_ids, "integer")
  expect_true(999L %in% f$user_ids)
  expect_true(998L %in% f$item_ids)
  ## Mixed types fall back to character ids.
  g <- svdpp(d, 2, epochs = 1, implicit = data.frame(user = "x", item = 1L))
  expect_type(g$user_ids, "character")
  expect_true("x" %in% recommend(g, users = "x", n = 1)$user)
})

test_that("fold_in() keeps id types", {
  set.seed(1)
  f <- asvd(d, 2, epochs = 2)
  g <- fold_in(f, data.frame(user = 1000L, item = 1:3, rating = c(5, 4, 1)))
  expect_type(g$user_ids, "integer")
  expect_identical(tail(g$user_ids, 1), 1000L)
  expect_identical(length(g$user_ids), length(g$users))
  r <- recommend(g, n = 1)
  expect_type(r$user, "integer")
  expect_true(1000L %in% r$user)
  h <- fold_in(f, data.frame(user = "new", item = 1, rating = 5))
  expect_type(h$user_ids, "character")
})

test_that("matrix input gives character or integer ids", {
  m <- matrix(c(5, NA, 3, 4, 2, NA), 2, 3, byrow = TRUE)
  set.seed(1)
  f <- svdpp(m, 1, epochs = 1)
  expect_type(recommend(f, n = 1, exclude_rated = FALSE)$item, "integer")
  dimnames(m) <- list(c("a", "b"), c("x", "y", "z"))
  set.seed(1)
  g <- svdpp(m, 1, epochs = 1)
  expect_type(recommend(g, n = 1)$item, "character")
  expect_identical(fitted(g), predict(g, clip = FALSE))
  expect_length(fitted(g), 4)
})

test_that("ids keep their type when nothing new is added", {
  set.seed(1)
  f <- asvd(d, 2, epochs = 2)
  g <- fold_in(f, data.frame(user = "1", item = 1:3, rating = 3))
  expect_type(g$user_ids, "integer")
  g <- fold_in(f, data.frame(user = 1000, item = 1:3, rating = 3))
  expect_type(g$user_ids, "integer")
  h <- svdpp(d, 2, epochs = 1,
             implicit = data.frame(user = as.character(d$user[1:3]),
                                   item = as.character(d$item[1:3])))
  expect_type(h$user_ids, "integer")
  expect_type(h$item_ids, "integer")
})

test_that("fold_in() updates the training RMSE", {
  set.seed(1)
  f <- asvd(d, 2, epochs = 2)
  g <- fold_in(f, data.frame(user = 1L, item = 1:3, rating = c(5, 1, 5)))
  expect_equal(g$train_rmse, sqrt(mean(residuals(g)^2)))
})
