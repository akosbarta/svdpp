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
    hint <- if (is_rating_matrix(data)) {
      paste0(" It is a matrix; `", arg, "` needs one row per user-item pair ",
             "with the user and item columns.")
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

is_rating_matrix <- function(x) is.matrix(x) || inherits(x, "Matrix")

## Convert a users x items matrix to the long format used internally, with
## columns named by `cols` (c(user = , item = , rating = )).
##
## * Ratings (`value = TRUE`): in a base or dense matrix, every non-NA entry
##   is a rating; in a sparse Matrix, every stored non-NA entry is a rating
##   (entries that are not stored are missing, not zero).
## * Implicit feedback (`value = FALSE`): every non-NA, non-zero (TRUE) entry
##   marks feedback.
##
## Row and column names are the user and item ids; without names the row and
## column numbers are used.
matrix_to_long <- function(m, cols, arg, value = TRUE) {
  if (value && !is.null(colnames(m)) && all(cols %in% colnames(m))) {
    stop(sprintf(paste0("`%s` is a matrix with columns %s, so it looks like ",
                        "ratings in long format (one row per rating). ",
                        "Convert it with as.data.frame() first; a matrix is ",
                        "read as users x items."),
                 arg, paste0("'", cols, "'", collapse = ", ")), call. = FALSE)
  }
  if (inherits(m, "sparseMatrix")) {
    m <- methods::as(methods::as(m, "generalMatrix"), "TsparseMatrix")
    i <- m@i + 1L
    j <- m@j + 1L
    x <- if (methods::.hasSlot(m, "x")) m@x else rep(TRUE, length(i))
  } else {
    m <- as.matrix(m)
    idx <- which(!is.na(m), arr.ind = TRUE)
    i <- idx[, 1]
    j <- idx[, 2]
    x <- m[idx]
  }
  if (value && !(is.numeric(x) || length(x) == 0L)) {
    stop(sprintf("`%s` must be a numeric matrix of ratings, not %s.", arg,
                 typeof(x)), call. = FALSE)
  }
  if (!value && !(is.numeric(x) || is.logical(x) || length(x) == 0L)) {
    stop(sprintf("`%s` must be a logical or numeric matrix, not %s.", arg,
                 typeof(x)), call. = FALSE)
  }
  keep <- !is.na(x)
  if (!value) keep <- keep & as.logical(x)
  ids <- function(nm, n, what) {
    if (is.null(nm)) return(seq_len(n))
    if (anyNA(nm) || anyDuplicated(nm)) {
      stop(sprintf("The %s names of `%s` must be unique and not missing.",
                   what, arg), call. = FALSE)
    }
    nm
  }
  uid <- ids(rownames(m), nrow(m), "row")
  iid <- ids(colnames(m), ncol(m), "column")
  out <- data.frame(uid[i[keep]], iid[j[keep]], stringsAsFactors = FALSE)
  names(out) <- cols[c("user", "item")]
  if (value) out[[cols[["rating"]]]] <- as.numeric(x[keep])
  ## All row and column ids, in matrix order, including rows and columns
  ## without entries: they are users and items of the model too.
  attr(out, "ids") <- list(user = uid, item = iid)
  out
}

## Give an implicit-feedback matrix without row (column) names the row
## (column) names of the ratings matrix `data`, whose shape it must then
## share: otherwise row and column numbers would be matched as ids.
align_implicit <- function(implicit, data) {
  for (k in 1:2) {
    if (is.null(dimnames(implicit)[[k]])) {
      if (dim(implicit)[k] != dim(data)[k]) {
        stop(sprintf(paste0("`implicit` has no %s names, so it must have as ",
                            "many %ss as `data` (%d), not %d."),
                     c("row", "column")[k], c("row", "column")[k],
                     dim(data)[k], dim(implicit)[k]), call. = FALSE)
      }
      if (!is.null(dimnames(data)[[k]])) {
        dn <- dimnames(implicit)
        if (is.null(dn)) dn <- list(NULL, NULL)
        dn[k] <- list(dimnames(data)[[k]])
        dimnames(implicit) <- dn
      }
    }
  }
  implicit
}

## Map user/item ids to 1-based integer indices and build the implicit-feedback
## matrix N (users x items, 0/1). N(u) is the set of items u rated, plus any
## items listed for u in `implicit`. `data` and `implicit` may be data frames
## or users x items matrices (see matrix_to_long()).
prepare_ratings <- function(data, user, item, rating, implicit = NULL) {
  check_colnames(user = user, item = item, rating = rating)
  cols <- c(user = user, item = item, rating = rating)
  if (is_rating_matrix(implicit)) {
    if (is_rating_matrix(data)) implicit <- align_implicit(implicit, data)
    implicit <- matrix_to_long(implicit, cols, "implicit", value = FALSE)
  }
  if (is_rating_matrix(data)) data <- matrix_to_long(data, cols, "data")
  check_columns(data, c(user, item, rating), "data")
  if (!nrow(data)) stop("`data` contains no ratings.", call. = FALSE)
  r <- check_rating(data[[rating]], rating)

  du <- as_id(data[[user]])
  di <- as_id(data[[item]])
  ## The ids in their original type (integer, factor, ...), one per user or
  ## item, so results can be returned in the type they came in. For a
  ## matrix, all its rows and columns in order.
  mat_ids <- attr(data, "ids")
  uval <- if (is.null(mat_ids)) data[[user]][!duplicated(du)] else mat_ids$user
  ival <- if (is.null(mat_ids)) data[[item]][!duplicated(di)] else mat_ids$item
  ulev <- as_id(uval)
  ilev <- as_id(ival)
  if (!is.null(implicit)) {
    check_columns(implicit, c(user, item), "implicit")
    iu <- as_id(implicit[[user]])
    ii <- as_id(implicit[[item]])
    new_u <- setdiff(iu, ulev)
    new_i <- setdiff(ii, ilev)
    uval <- combine_ids(uval, implicit[[user]][match(new_u, iu)])
    ival <- combine_ids(ival, implicit[[item]][match(new_i, ii)])
    ulev <- c(ulev, new_u)
    ilev <- c(ilev, new_i)
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
       users = ulev, items = ilev, user_ids = uval, item_ids = ival, N = N)
}

## Concatenate two id vectors, keeping their type when they agree (both
## numeric, both factors, both Dates, ...) and falling back to the character
## ids used for matching otherwise.
combine_ids <- function(a, b) {
  if (!length(b)) return(a)
  plain_num <- function(x) is.numeric(x) && !is.object(x)
  ## Whole-number doubles (e.g. user = 1000 typed at the console) do not turn
  ## integer ids into doubles.
  if (is.integer(a) && plain_num(b) && all(b == trunc(b)) &&
      all(abs(b) <= .Machine$integer.max)) {
    b <- as.integer(b)
  }
  if (identical(class(a), class(b)) || (plain_num(a) && plain_num(b))) {
    c(a, b)
  } else {
    c(as_id(a), as_id(b))
  }
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
