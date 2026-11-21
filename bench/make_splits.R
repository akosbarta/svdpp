## Download MovieLens and write identical train/test (+ validation) splits as
## CSV, so R and Python benchmarks see exactly the same data.
## Usage (from the package root): Rscript bench/make_splits.R 1m
args <- commandArgs(trailingOnly = TRUE)
size <- if (length(args)) args[1] else "100k"
if (requireNamespace("svdpp", quietly = TRUE)) library(svdpp) else pkgload::load_all(".", quiet = TRUE)
out <- file.path("bench", "data", size)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
d <- movielens(size, dir = file.path("bench", "data", "raw"))[, c("user", "item", "rating")]
set.seed(1)
te <- sample(nrow(d), round(0.1 * nrow(d)))
tr <- d[-te, ]
va <- sample(nrow(tr), round(0.1 * nrow(tr)))
write.csv(tr, file.path(out, "train.csv"), row.names = FALSE)
write.csv(d[te, ], file.path(out, "test.csv"), row.names = FALSE)
write.csv(tr[-va, ], file.path(out, "subtrain.csv"), row.names = FALSE)
write.csv(tr[va, ], file.path(out, "val.csv"), row.names = FALSE)
cat(sprintf("%s: %d users, %d items, %d train / %d test ratings -> %s\n", size,
            length(unique(d$user)), length(unique(d$item)), nrow(tr), length(te), out))
