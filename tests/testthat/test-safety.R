set.seed(11)
d <- simulate_ratings(n_users = 60, n_items = 40, n_ratings = 800)

test_that("arguments are validated with informative errors", {
  expect_error(svdpp(d, epochs = -1), "`epochs` must be")
  expect_error(svdpp(d, epochs = 0), "`epochs` must be")
  expect_error(svdpp(d, epochs = 2.5), "whole number")
  expect_error(svdpp(d, lr = "a"), "`lr` must be")
  expect_error(svdpp(d, lr = 0), "`lr` must be")
  expect_error(svdpp(d, lambda = -1), "`lambda` must be")
  expect_error(svdpp(d, shuffle = NA), "`shuffle` must be TRUE or FALSE")
  expect_error(svdpp(d, maxit = 0, method = "lbfgs"), "`maxit` must be")
  expect_error(svdpp(d, init_sd = Inf), "`init_sd` must be")
  expect_error(asvd(d, baseline_lambda = 3), "baseline_lambda")
  expect_error(baseline_predictor(d, tol = -1), "`tol` must be")
  expect_error(svdpp(d, method = "als"), "should be one of")
})

test_that("non-finite ratings are rejected", {
  x <- d
  x$rating[3] <- Inf
  expect_error(svdpp(x), "infinite")
  x$rating[3] <- NaN
  expect_error(svdpp(x), "missing values")
})

test_that("divergent SGD stops with a clear error instead of returning NaN", {
  set.seed(1)
  expect_error(svdpp(d, factors = 3, lr = 10, epochs = 5), "diverged")
  set.seed(1)
  expect_error(asvd(d, factors = 3, lr = 50, epochs = 5), "diverged")
})

test_that("ids match whatever their storage type", {
  x <- data.frame(user = c(100000L, 2L, 100000L), item = c(1L, 2L, 2L),
                  rating = c(5, 1, 4))
  f <- baseline_predictor(x)
  expect_true("100000" %in% f$users)
  p_int <- predict(f, data.frame(user = 100000L, item = 1L), clip = FALSE)
  p_dbl <- predict(f, data.frame(user = 1e5, item = 1), clip = FALSE)
  p_chr <- predict(f, data.frame(user = "100000", item = "1"), clip = FALSE)
  expect_equal(p_dbl, p_int)
  expect_equal(p_chr, p_int)
  expect_false(isTRUE(all.equal(p_int, f$mu + unname(f$bi["1"]))))
  expect_equal(nrow(recommend(f, users = 1e5, n = 1, exclude_rated = FALSE)), 1)
})

test_that("predict copes with empty and missing input", {
  set.seed(1)
  f <- svdpp(d, factors = 2, epochs = 2)
  expect_identical(predict(f, d[0, ]), numeric(0))
  p <- predict(f, data.frame(user = c(NA, "1"), item = c("1", NA)), clip = FALSE)
  expect_equal(p[1], f$mu + unname(f$bi["1"]))
  expect_equal(p[2], f$mu + unname(f$bu["1"]))
  expect_error(predict(f, as.matrix(d)), "data frame")
  expect_error(predict(f, data.frame(u = 1, i = 1)), "not found")
  expect_error(predict(f, d, clip = NA), "`clip`")
})

test_that("recommend handles edge cases", {
  set.seed(1)
  f <- svdpp(d, factors = 2, epochs = 2)
  expect_error(recommend(f, users = "1", n = 0), "`n` must be")
  expect_error(recommend(f, users = "1", n = -2), "`n` must be")
  r <- suppressWarnings(recommend(f, users = c("zzz", "yyy")))
  expect_equal(nrow(r), 0)
  expect_named(r, c("user", "item", "score", "rank"))
  big <- recommend(f, users = "1", n = 1e6)
  expect_equal(nrow(big), length(f$items) - sum(d$user == 1))
  all_users <- recommend(f, n = 2)
  expect_equal(nrow(all_users), 2 * length(f$users))
  expect_equal(all_users$rank, rep(1:2, length(f$users)))
})

test_that("fold_in gives a clear error for models that cannot fold in", {
  set.seed(1)
  f <- svdpp(d, factors = 2, epochs = 1)
  expect_error(fold_in(f, d), "Only asvd\\(\\) models")
})

test_that("single-rating and single-user data work", {
  expect_s3_class(svdpp(d[1, ], factors = 2, epochs = 2), "svdpp")
  expect_s3_class(asvd(d[d$user == d$user[1], ], factors = 2, epochs = 2), "asvd")
  expect_s3_class(svdpp(d[1:5, ], factors = 2, method = "lbfgs"), "svdpp")
})

test_that("malformed inputs give informative errors", {
  x <- d
  x$rating <- as.character(x$rating)
  expect_error(svdpp(x), "must be numeric, not character.*as.numeric")
  x$rating[2] <- "four"
  expect_error(svdpp(x), "must be numeric, not character")
  x <- d
  x$rating <- factor(x$rating)
  expect_error(svdpp(x), "not factor.*as.character")
  x <- d
  x$rating <- x$rating > 3
  expect_error(svdpp(x), "not logical")
  x <- d
  x$user <- I(lapply(x$user, function(z) c(z, z)))
  expect_error(svdpp(x), "plain vector")
  x <- d
  x$user <- as.complex(x$user)
  expect_error(svdpp(x), "plain vector")
  expect_error(svdpp(as.matrix(d)), "long format")
  expect_error(svdpp(NULL), "class 'NULL'")
  expect_error(svdpp(as.list(d)), "class 'list'")
  expect_error(svdpp(d[0, ]), "no ratings")
  expect_error(svdpp(d, user = 1), "`user` must be a single column name")
  expect_error(svdpp(d, user = c("user", "item")), "`user` must be a single")
  expect_error(svdpp(d, user = "item"), "different columns")
  set.seed(1)
  f <- svdpp(d, factors = 2, epochs = 1)
  expect_error(predict(f, as.matrix(d)), "It is a matrix")
  expect_error(predict(f, data.frame(user = I(list(1, 2)), item = 1:2)),
               "plain vector")
  expect_error(recommend(f, users = list(1, 2)), "vector of user ids")
  expect_error(recommend(f, users = d[1:2, ]), "vector of user ids")
  expect_error(fold_in(asvd(d, factors = 2, epochs = 1), transform(d, rating = "5")),
               "must be numeric")
})

test_that("odd but valid ids work", {
  x <- d
  x$item <- as.Date("2020-01-01") + x$item
  set.seed(1)
  f <- svdpp(x, factors = 2, epochs = 1)
  expect_length(predict(f, x[1:3, ]), 3)
  x <- d
  x$user <- factor(paste0("u", x$user))
  x$rating <- as.integer(x$rating)
  expect_s3_class(svdpp(x, factors = 2, epochs = 1), "svdpp")
})

test_that("simulate_ratings validates its arguments", {
  expect_error(simulate_ratings(n_users = -1), "`n_users` must be")
  expect_error(simulate_ratings(n_items = 2.5), "`n_items` must be")
  expect_error(simulate_ratings(noise = -1), "`noise` must be")
  expect_error(simulate_ratings(n_users = 2, n_items = 2, n_ratings = 5),
               "cannot exceed")
  x <- simulate_ratings(n_users = 5, n_items = 4, n_ratings = 20)
  expect_equal(nrow(unique(x[c("user", "item")])), 20)
  expect_type(x$user, "integer")
  expect_true(all(x$rating %in% 1:5))
})
