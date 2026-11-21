# svdpp 0.2.0

## Breaking changes

* The arguments of `svdpp()`, `asvd()` and `baseline_predictor()` are
  reordered so the main modelling choices come first, as in `svd()` or
  Surprise's `SVDpp`: `svdpp(data, factors, method, lambda, implicit, ...)`.
  The column-name arguments `user`, `item` and `rating` move to the end.
  Calls that name their arguments are unaffected; positional calls such as
  `svdpp(d, "u", "i", "r")` must name them.

## New features

* `data` can be a users x items matrix: a base `matrix` with `NA` for
  missing ratings, or a sparse `Matrix` whose stored entries are the
  ratings. Row and column names are the user and item ids. `implicit` and
  `fold_in()` accept matrices in the same way.
* `predict()` without `newdata` returns predictions for the training
  ratings, and fitted models have `fitted()`, `residuals()` and `nobs()`
  methods, as for `lm()`.

## Other changes

* `recommend()` returns user and item ids in their original type (integer,
  factor, Date, ...) instead of always as character, so its result can be
  joined back onto the input data.
* A baseline `lambda` of 0 no longer gives `NaN` biases (or an error) for
  users or items without ratings.
* Calls written for 0.1.0 that pass column names by position now fail with
  a hint to name them.
* Requires Matrix >= 1.5-0.

# svdpp 0.1.0

* First release.
* `svdpp()`: SVD++ (Koren 2008) with two estimators, stochastic gradient
  descent with per-user grouping of the implicit-feedback updates, and
  full-batch L-BFGS-B with sparse-matrix gradients.
* `asvd()`: Asymmetric-SVD with fully vectorised per-user SGD and L-BFGS-B,
  and `fold_in()` to score new users without refitting.
* `baseline_predictor()`: regularised baseline estimates solved by
  alternating least squares.
* `predict()` and `recommend()` methods shared by all models.
* `simulate_ratings()` and `movielens()` data helpers.
* Input and argument validation with informative errors; SGD stops if it
  diverges; ids match across integer, double, character and factor types.
