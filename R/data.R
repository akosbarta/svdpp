## Internal data preparation shared by the model fitters.

## Column-name arguments must be single, distinct strings.
check_colnames <- function(...) {
  args <- list(...)
  for (nm in names(args)) {
    x <- args[[nm]]
    if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
      stop(sprintf("`%s` must be a single column name (a string).", nm),
           call. = FALSE)
    }
  }
  if (anyDuplicated(unlist(args))) {
    stop(sprintf("%s must name different columns.",
                 paste0("`", names(args), "`", collapse = ", ")), call. = FALSE)
  }
}

## `data` must be a long-format data frame (one row per user-item pair) with
## the given columns, each a plain atomic vector.
check_columns <- function(data, cols, arg, allow_na = FALSE) {
  if (!is.data.frame(data)) {
    hint <- if (is.matrix(data) || inherits(data, "Matrix")) {
      paste0(" It is a matrix; the package expects one row per rating with ",
             "user, item and rating columns. A users x items matrix `m` with ",
             "NA for missing ratings can be converted with\n  i <- which(!is.na(m), ",
             "arr.ind = TRUE); data.frame(user = i[, 1], item = i[, 2], ",
             "rating = m[i])")
    } else {
      sprintf(" It is of class '%s'.", class(data)[1])
    }
    stop(sprintf("`%s` must be a data frame.%s", arg, hint), call. = FALSE)
  }
  miss <- setdiff(cols, names(data))
  if (length(miss)) {
    stop(sprintf("Column(s) %s not found in `%s`.",
                 paste0("'", miss, "'", collapse = ", "), arg), call. = FALSE)
  }
  for (col in cols) {
    x <- data[[col]]
    if (!is.atomic(x) || !is.null(dim(x)) || is.complex(x) || is.raw(x)) {
      stop(sprintf(paste0("Column '%s' of `%s` must be a plain vector ",
                          "(character, numeric or factor), not a %s."),
                   col, arg, class(x)[1]), call. = FALSE)
    }
    if (!allow_na && anyNA(x)) {
      stop(sprintf("Column '%s' of `%s` contains missing values.", col, arg),
           call. = FALSE)
    }
  }
}

## Ratings must be finite numbers; say what was found otherwise.
check_rating <- function(r, col) {
  if (!is.numeric(r)) {
    hint <- if (is.factor(r)) {
      " Convert it with as.numeric(as.character(x)), not as.numeric(x)."
    } else if (is.character(r)) {
      " Convert it with as.numeric() and check for values that are not numbers."
    } else ""
    stop(sprintf("The rating column '%s' must be numeric, not %s.%s", col,
                 class(r)[1], hint), call. = FALSE)
  }
  r <- as.numeric(r)
  if (!all(is.finite(r))) {
    stop(sprintf("The rating column '%s' contains infinite values.", col),
         call. = FALSE)
  }
  r
}

## Ids are matched as strings. as.character() writes some whole-number
## doubles in scientific notation (1e5 -> "1e+05") while the same integer
## gives "100000"; write those in fixed notation so both match.
as_id <- function(x) {
  out <- as.character(x)
  if (is.double(x)) {
    sci <- grepl("e", out, fixed = TRUE)
    if (any(sci)) {
      w <- sci & is.finite(x) & x == trunc(x) & abs(x) < 2^53
      out[w] <- formatC(x[w], format = "f", digits = 0)
    }
  }
  out
}

## Map user/item ids to 1-based integer indices and build the implicit-feedback
## matrix N (users x items, 0/1). N(u) is the set of items u rated, plus any
## items listed for u in `implicit`.
prepare_ratings <- function(data, user, item, rating, implicit = NULL) {
  check_colnames(user = user, item = item, rating = rating)
  check_columns(data, c(user, item, rating), "data")
  if (!nrow(data)) stop("`data` has no rows.", call. = FALSE)
  r <- check_rating(data[[rating]], rating)

  du <- as_id(data[[user]])
  di <- as_id(data[[item]])
  ulev <- unique(du)
  ilev <- unique(di)
  if (!is.null(implicit)) {
    check_columns(implicit, c(user, item), "implicit")
    iu <- as_id(implicit[[user]])
    ii <- as_id(implicit[[item]])
    ulev <- union(ulev, iu)
    ilev <- union(ilev, ii)
  }
  nu <- length(ulev)
  ni <- length(ilev)
  u <- match(du, ulev)
  i <- match(di, ilev)
  if (anyDuplicated((u - 1) * ni + i)) {
    stop("`data` contains more than one rating for some user-item pair.",
         call. = FALSE)
  }

  nu_idx <- u
  ni_idx <- i
  if (!is.null(implicit)) {
    nu_idx <- c(nu_idx, match(iu, ulev))
    ni_idx <- c(ni_idx, match(ii, ilev))
  }
  N <- sparseMatrix(i = nu_idx, j = ni_idx, x = 1, dims = c(nu, ni))
  N@x[] <- 1 # duplicates were summed; binarise

  list(u = u, i = i, r = r, nu = nu, ni = ni,
       users = ulev, items = ilev, N = N)
}

## Row-normalised implicit matrix |N(u)|^{-1/2} N.
normalise_implicit <- function(N) {
  n <- rowSums(N)
  Diagonal(x = 1 / sqrt(pmax(n, 1))) %*% N
}

## Row-wise dot products A[ia, ] . B[ib, ], in chunks to bound memory.
rowdot <- function(A, ia, B, ib, chunk = 65536L) {
  n <- length(ia)
  out <- numeric(n)
  start <- 1L
  while (start <= n) {
    idx <- start:min(n, start + chunk - 1L)
    out[idx] <- rowSums(A[ia[idx], , drop = FALSE] * B[ib[idx], , drop = FALSE])
    start <- start + chunk
  }
  out
}
