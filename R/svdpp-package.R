#' svdpp: SVD++ and related latent factor recommender models
#'
#' Pure-R implementations of the models from Koren (2008):
#' [baseline_predictor()], [asvd()] (Asymmetric-SVD) and [svdpp()]. All
#' fitted models share [predict.cf_model()] and [recommend()]; Asymmetric-SVD
#' models can also take new users without refitting via [fold_in()].
#'
#' @references
#' Koren, Y. (2008). Factorization meets the neighborhood: a multifaceted
#' collaborative filtering model. *Proceedings of the 14th ACM SIGKDD
#' International Conference on Knowledge Discovery and Data Mining*, 426--434.
#' \doi{10.1145/1401890.1401944}
#'
#' @keywords internal
#' @importFrom Matrix sparseMatrix Diagonal t crossprod rowSums colSums
#' @importFrom methods as
"_PACKAGE"
