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
