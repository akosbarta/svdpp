set.seed(7)
d <- simulate_ratings(n_users = 40, n_items = 25, n_ratings = 400)
d$user <- paste0("u", d$user)
d$item <- paste0("i", d$item)
users <- sort(unique(d$user))
items <- sort(unique(d$item))
m <- matrix(NA_real_, length(users), length(items),
            dimnames = list(users, items))
m[cbind(match(d$user, users), match(d$item, items))] <- d$rating
s <- Matrix::sparseMatrix(i = match(d$user, users), j = match(d$item, items),
                          x = d$rating, dimnames = list(users, items))

## Same fit from a long data frame: ratings are reordered, so compare the
## deterministic parts (baseline, L-BFGS objective), not SGD paths.
test_that("dense and sparse matrices give the same model as a data frame", {
  b_df <- baseline_predictor(d)
  b_m <- baseline_predictor(m)
  b_s <- baseline_predictor(s)
  expect_equal(b_m$bu[users], b_df$bu[users])
  expect_equal(b_s$bi[items], b_df$bi[items])
  expect_equal(b_m$n_ratings, nrow(d))
  set.seed(1)
  f_m <- svdpp(m, 2, "lbfgs", maxit = 20)
  set.seed(1)
  f_s <- svdpp(s, 2, "lbfgs", maxit = 20)
  expect_equal(f_m$convergence$objective, f_s$convergence$objective)
  nd <- data.frame(user = d$user[1:5], item = d$item[1:5])
  expect_length(predict(f_m, nd), 5)
})

test_that("factors and method can be given by position", {
  set.seed(1)
  f <- svdpp(d, 3, "lbfgs", maxit = 5)
  expect_equal(f$factors, 3L)
  expect_identical(f$method, "lbfgs")
  set.seed(1)
  a <- asvd(m, 2, "sgd", epochs = 2)
  expect_equal(a$factors, 2L)
  expect_identical(baseline_predictor(d, c(user = 1, item = 1))$lambda,
                   c(user = 1, item = 1))
})

test_that("matrices without dimnames use row and column numbers as ids", {
  m2 <- unname(m)
  set.seed(1)
  f <- svdpp(m2, factors = 2, epochs = 2)
  expect_identical(f$users, as.character(seq_len(nrow(m2))))
  p <- predict(f, data.frame(user = 1, item = 1))
  expect_length(p, 1)
  expect_true(is.finite(p))
})

test_that("stored entries of a sparse matrix are ratings, including zeros", {
  z <- Matrix::sparseMatrix(i = c(1, 1, 2), j = c(1, 2, 2), x = c(0, 4, 2))
  f <- baseline_predictor(z)
  expect_equal(f$n_ratings, 3)
  expect_equal(f$mu, 2)
})

test_that("implicit feedback can be a matrix", {
  imp <- !is.na(m)
  imp[1, ] <- TRUE
  imp_df <- data.frame(user = users[1], item = items)
  set.seed(1)
  f_m <- svdpp(m, 2, "lbfgs", implicit = imp, maxit = 5)
  set.seed(1)
  f_d <- svdpp(m, 2, "lbfgs", implicit = imp_df, maxit = 5)
  expect_equal(f_m$convergence$objective, f_d$convergence$objective)
  expect_error(svdpp(m, implicit = matrix("a", 2, 2, dimnames = list(users[1:2], items[1:2]))),
               "logical or numeric")
})

test_that("fold_in() accepts a matrix of new ratings", {
  set.seed(1)
  f <- asvd(d, factors = 2, epochs = 2)
  new <- matrix(c(5, NA, 1), 1, 3, dimnames = list("new", items[1:3]))
  g <- fold_in(f, new)
  expect_true("new" %in% g$users)
  h <- fold_in(f, data.frame(user = "new", item = items[c(1, 3)],
                             rating = c(5, 1)))
  expect_equal(g$U["new", ], h$U["new", ])
})

test_that("bad matrices give informative errors", {
  expect_error(svdpp(matrix(NA_real_, 3, 3)), "no ratings")
  expect_error(svdpp(matrix("5", 2, 2)), "numeric matrix")
  expect_error(svdpp(matrix(TRUE, 2, 2)), "numeric matrix")
  dup <- matrix(1, 2, 2, dimnames = list(c("a", "a"), c("x", "y")))
  expect_error(svdpp(dup), "row names .* unique")
  expect_error(svdpp(as.matrix(data.frame(user = 1:2, item = 1:2,
                                           rating = 3:4))), "long format")
  set.seed(1)
  f <- svdpp(m, factors = 2, epochs = 1)
  expect_error(predict(f, m), "It is a matrix")
})

test_that("matrix rows and columns are users and items in matrix order", {
  m3 <- matrix(c(5, NA, 3,
                 NA, NA, NA,
                 NA, 2, NA,
                 4, NA, NA), 4, 3, byrow = TRUE,
               dimnames = list(c("a", "b", "c", "d"), c("x", "y", "z")))
  set.seed(1)
  f <- svdpp(m3, 1, epochs = 2)
  expect_identical(f$users, c("a", "b", "c", "d"))
  expect_identical(f$items, c("x", "y", "z"))
  expect_equal(nobs(f), 4)
  ## User b has no ratings: zero bias and factors, still recommendable.
  expect_equal(unname(f$bu["b"]), 0)
  expect_equal(unname(f$U["b", ]), c(0))
  expect_silent(r <- recommend(f, users = "b", n = 3))
  expect_equal(nrow(r), 3)
  b <- baseline_predictor(m3, c(user = 0, item = 0))
  expect_true(all(is.finite(c(b$bu, b$bi))))
})

test_that("an unnamed implicit matrix takes the names of a matrix `data`", {
  set.seed(1)
  f <- svdpp(m, 2, epochs = 2, implicit = !is.na(unname(m)))
  expect_identical(f$users, users)
  expect_identical(f$items, items)
  imp <- unname(m)[, 1:3]
  expect_error(svdpp(m, 2, implicit = !is.na(imp)), "as many columns")
  ## Row names given, column names taken from `data`.
  imp <- !is.na(m)
  colnames(imp) <- NULL
  set.seed(1)
  g <- svdpp(m, 2, epochs = 2, implicit = imp)
  expect_identical(g$items, items)
})

test_that("fold_in() needs row names on a matrix", {
  set.seed(1)
  f <- asvd(m, 2, epochs = 2)
  expect_error(fold_in(f, unname(m[1:2, ])), "needs row names")
  g <- fold_in(f, m[1:2, , drop = FALSE])
  expect_identical(g$users, f$users)
})

test_that("zero baseline lambda does not give NaN for users or items without ratings", {
  set.seed(1)
  a <- asvd(d, 2, epochs = 2, implicit = data.frame(user = "zz", item = "i1"),
            baseline_lambda = c(user = 0, item = 10))
  expect_true(all(is.finite(a$baseline$bu)))
  a <- asvd(d, 2, epochs = 2, implicit = data.frame(user = "u1", item = "zz"),
            baseline_lambda = c(user = 10, item = 0))
  expect_true(all(is.finite(a$baseline$bi)))
  set.seed(1)
  a <- asvd(d, 2, epochs = 2, baseline_lambda = c(user = 0, item = 10))
  g <- suppressWarnings(fold_in(a, data.frame(user = "new", item = "nope",
                                               rating = 3)))
  expect_true(is.finite(g$bu[["new"]]))
})

test_that("old positional calls get a hint", {
  expect_error(svdpp(d, "user", "item", "rating"), "Since svdpp 0.2.0")
  expect_error(asvd(d, "user"), "Since svdpp 0.2.0")
  expect_error(baseline_predictor(d, "user"), "Since svdpp 0.2.0")
})
