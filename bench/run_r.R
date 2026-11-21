## Benchmark the package's models on a split written by make_splits.R.
## Usage (from the package root): Rscript bench/run_r.R 1m [models...]
## models: baseline, svdpp, asvd (default: all)
args <- commandArgs(trailingOnly = TRUE)
size <- if (length(args)) args[1] else "100k"
models <- if (length(args) > 1) args[-1] else c("baseline", "svdpp", "asvd")
if (requireNamespace("svdpp", quietly = TRUE)) library(svdpp) else pkgload::load_all(".", quiet = TRUE)
dir <- file.path("bench", "data", size)
rd <- function(f) read.csv(file.path(dir, f))
tr <- rd("train.csv"); te <- rd("test.csv"); st <- rd("subtrain.csv"); va <- rd("val.csv")
rmse <- function(fit, x) sqrt(mean((predict(fit, x) - x$rating)^2))
k <- 20
dir.create(file.path("bench", "results"), showWarnings = FALSE)

res <- list()
add <- function(name, fit, settings) {
  row <- data.frame(dataset = size, model = name, settings = settings,
                    fit_s = round(fit$time, 2), test_rmse = round(rmse(fit, te), 4))
  res[[length(res) + 1]] <<- row
  print(row, row.names = FALSE)
}
## Pick the grid row with the lowest validation RMSE, fitting on subtrain.
tune <- function(label, grid, fitter) {
  v <- vapply(seq_len(nrow(grid)), function(j) rmse(do.call(fitter, c(list(st), grid[j, , drop = FALSE])), va), 0)
  cat(label, "validation:\n"); print(cbind(grid, val_rmse = round(v, 4)), row.names = FALSE)
  grid[which.min(v), , drop = FALSE]
}
fmt <- function(p) paste(names(p), unlist(p), sep = "=", collapse = ", ")

if ("baseline" %in% models) {
  add("baseline_predictor (ALS)", baseline_predictor(tr), "lambda user=15, item=10")
}
if ("svdpp" %in% models) {
  set.seed(1)
  add("svdpp sgd", svdpp(tr, factors = k), "Surprise defaults: lr=0.007, lambda=0.02, 20 epochs")
  best <- tune("svdpp lbfgs", data.frame(lambda = c(10, 20, 40, 80)),
               function(x, lambda) { set.seed(1); svdpp(x, factors = k, method = "lbfgs", lambda = lambda) })
  set.seed(1)
  add("svdpp lbfgs", svdpp(tr, factors = k, method = "lbfgs", lambda = best$lambda), fmt(best))
}
if ("asvd" %in% models) {
  set.seed(1)
  add("asvd sgd", asvd(tr, factors = k), "paper defaults: lr=0.002, lambda=0.04, 30 epochs")
  best <- tune("asvd sgd", expand.grid(lr = c(0.002, 0.005, 0.01), lambda = c(0.02, 0.04, 0.08)),
               function(x, lr, lambda) { set.seed(1); asvd(x, factors = k, lr = lr, lambda = lambda) })
  set.seed(1)
  add("asvd sgd", asvd(tr, factors = k, lr = best$lr, lambda = best$lambda), paste(fmt(best), "30 epochs, val-tuned"))
  best <- tune("asvd lbfgs", data.frame(lambda = c(10, 20, 40, 80)),
               function(x, lambda) { set.seed(1); asvd(x, factors = k, method = "lbfgs", lambda = lambda) })
  set.seed(1)
  add("asvd lbfgs", asvd(tr, factors = k, method = "lbfgs", lambda = best$lambda), paste(fmt(best), "val-tuned"))
}
out <- do.call(rbind, res)
write.csv(out, file.path("bench", "results", sprintf("r_%s_%s.csv", size, paste(models, collapse = "-"))), row.names = FALSE)
