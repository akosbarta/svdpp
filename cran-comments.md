## Submission

This is a new submission.

The package implements the SVD++ and Asymmetric-SVD recommender models of
Koren (2008) <doi:10.1145/1401890.1401944> in pure R.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.
* Possibly misspelled words in DESCRIPTION: "Koren" is the author of the
  cited paper, "Broyden", "Fletcher", "Goldfarb" and "Shanno" name the
  L-BFGS-B method, and "Recommender" is the standard term (recommender
  systems); all are spelled correctly.

## Test environments

* local macOS, R 4.5.1
* GitHub Actions: macOS, Windows and Ubuntu (R release), Windows and
  Ubuntu (R devel), Ubuntu (R oldrel)

## Notes for the reviewers

* `movielens()` downloads data from GroupLens. Its example is in
  `\donttest{}` because it needs an internet connection; the function
  returns `NULL` with an informative message if the download fails.
