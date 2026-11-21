# svdpp

[![R-CMD-check](https://github.com/akosbarta/svdpp/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/akosbarta/svdpp/actions/workflows/R-CMD-check.yaml)

The latent factor recommender models of Koren (2008), *Factorization meets
the neighborhood* (see [References](#references)), written in pure R. The only dependency is **Matrix**,
which ships with R.

| Function | Model | Prediction |
|---|---|---|
| `baseline_predictor()` | Baseline estimates (Sec. 2.1) | $\mu + b_u + b_i$ |
| `asvd()` | Asymmetric-SVD (Sec. 4.1, Eq. 13) | $\mu + b_u + b_i + q_i^\top\big(\lvert R(u)\rvert^{-1/2}\sum_{j\in R(u)}(r_{uj}-b_{uj})x_j + \lvert N(u)\rvert^{-1/2}\sum_{j\in N(u)} y_j\big)$ |
| `svdpp()` | SVD++ (Sec. 4.2, Eq. 15) | $\mu + b_u + b_i + q_i^\top\big(p_u + \lvert N(u)\rvert^{-1/2}\sum_{j\in N(u)} y_j\big)$ |

`R(u)` is the set of items `u` rated. `N(u)` is the set of items with implicit
feedback from `u`; it defaults to `R(u)`. You can add other implicit
feedback, such as views or purchases without a rating, through `implicit`.

Asymmetric-SVD has no free user vector: a user is represented through the
items they rated. As a result, `fold_in()` can score new users from their
ratings without refitting, and predictions can be explained by the user's
past ratings.

## Installation

```r
# install.packages("remotes")
remotes::install_github("akosbarta/svdpp")
```

## Usage

```r
library(svdpp)
set.seed(1)                                       # for reproducible fits
d <- simulate_ratings()                           # or movielens("100k")
fit <- svdpp(d, factors = 20)                     # SGD, as in the paper
fit2 <- svdpp(d, factors = 20, method = "lbfgs")  # full-batch L-BFGS-B
predict(fit, data.frame(user = 1, item = 2))
recommend(fit, users = c(1, 2), n = 10)

a <- asvd(d, factors = 20)
a <- fold_in(a, data.frame(user = "new", item = c(1, 2, 3), rating = c(5, 4, 1)))
recommend(a, users = "new", n = 5)
```

## Estimators

Both latent factor models have two estimators.

- **`"sgd"`** follows the paper's stochastic gradient descent but processes
  ratings user by user. The implicit term is computed once per user, and the
  `y_j` (and, in ASVD, `x_j`) are updated once at the end of that user's
  ratings. An epoch then costs time linear in the number of ratings, while
  recomputing the implicit sum for every rating, as Surprise does, makes the
  cost grow with the square of a user's rating count. ASVD goes further: the
  user's representation is fixed during their ratings, so all of a user's
  updates are vectorised. The only sequential part, the `b_u` update, is a
  linear recurrence computed with `stats::filter()`. The tests check that
  this gives the same result as a plain loop over ratings.
- **`"lbfgs"`** minimises the full regularised squared error with
  `stats::optim()`. The gradient is a few sparse matrix products, so most of
  the work runs in compiled code even though the package has none of its own.

## Memory and robustness

Peak R memory on MovieLens-1M (900k ratings, 20 factors) is 0.3 to 0.5 GB
for every model and estimator, and grows roughly linearly with the number
of ratings. Expect a few GB for 10M ratings. The package has no compiled
code, so it cannot crash R; running out of memory gives an ordinary R error.
Inputs and arguments are validated with informative errors, ids match
whether stored as integer, double, character or factor, and SGD stops with
an error if the learning rate makes it diverge.

## Early benchmark (MovieLens-1M, 90/10 split, 20 factors, one seed)

| Implementation | Fit time | Test RMSE |
|---|---|---|
| Surprise BaselineOnly (ALS) | 0.4 s | 0.9066 |
| `baseline_predictor()` (ALS run to convergence) | 2.7 s | 0.9066 |
| Surprise SVD (Funk, no implicit term) | 0.8 s | 0.8637 |
| Surprise SVD++ (Cython), defaults | 106–160 s | 0.8561 |
| `svdpp(method = "sgd")`, Surprise's defaults | 74.0 s | 0.8480 |
| `svdpp(method = "lbfgs")`, lambda = 20 (validation) | 34.1 s | **0.8399** |
| `asvd(method = "sgd")`, paper's defaults | 20.0 s | 0.8470 |
| `asvd(method = "sgd")`, lr = 0.002, lambda = 0.08 (validation) | 18.5 s | 0.8415 |
| `asvd(method = "lbfgs")`, lambda = 40 (validation) | 38.4 s | 0.8441 |

The ordering matches the paper's Table 1: SVD is worse than Asymmetric-SVD,
which is slightly worse than SVD++. See [`bench/`](bench) to reproduce. These
numbers are preliminary: one split, one seed, unequal tuning. Surprise's
SVD++ time varied between two runs on the same machine.

## Roadmap

- The integrated neighbourhood + factor model (Koren 2008, Sec. 5) and the
  global neighbourhood model (Sec. 3)
- timeSVD++ (Koren 2009)
- recommenderlab integration

## References

Koren, Y. (2008). Factorization meets the neighborhood: a multifaceted
collaborative filtering model. In *Proceedings of the 14th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining (KDD '08)*,
Las Vegas, Nevada, USA, August 24–27, 2008, pp. 426–434. New York: ACM.
<https://doi.org/10.1145/1401890.1401944>

```bibtex
@inproceedings{koren2008factorization,
  author    = {Koren, Yehuda},
  title     = {Factorization Meets the Neighborhood: A Multifaceted
               Collaborative Filtering Model},
  booktitle = {Proceedings of the 14th ACM SIGKDD International Conference
               on Knowledge Discovery and Data Mining},
  series    = {KDD '08},
  year      = {2008},
  pages     = {426--434},
  publisher = {ACM},
  address   = {New York, NY, USA},
  location  = {Las Vegas, Nevada, USA},
  doi       = {10.1145/1401890.1401944}
}
```
