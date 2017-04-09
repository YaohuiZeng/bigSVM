#' Fit sparse linear SVM with lasso or elasti-net regularization
#' 
#' Fit solution paths for sparse linear SVM regularized by lasso or elastic-net
#' over a grid of values for the regularization parameter lambda.
#' 
#' The sequence of models indexed by the regularization parameter \code{lambda}
#' is fitted using a semismooth Newton coordinate descent algorithm. The
#' objective function is defined to be \deqn{\frac{1}{n} \sum hingeLoss(y_i
#' x_i^T w) + \lambda\textrm{penalty}.}{\sum hingeLoss(y_i x_i^T w) /n +
#' \lambda*penalty.} where \deqn{hingeLoss(t) = max(0, 1-t)} and the intercept
#' is unpenalized.
#' 
#' The program supports different types of preprocessing techniques. They are
#' applied to each column of the input matrix \code{X}. Let x be a column of
#' \code{X}. For \code{preprocess = "standardize"}, the formula is \deqn{x' =
#' \frac{x-mean(x)}{sd(x)};}{x' = (x-mean(x))/sd(x);} for \code{preprocess =
#' "rescale"}, \deqn{x' = \frac{x-min(x)}{max(x)-min(x)}.}{x' =
#' (x-min(x))/(max(x)-min(x)).} The models are fit with preprocessed input,
#' then the coefficients are transformed back to the original scale via some
#' algebra.
#' 
#' @param X Input matrix, without an intercept. It must be a
#' \code{\link[bigmemory]{big.matrix}} object. The function standardizes the
#' data and includes an intercept internally by default during the model
#' fitting.
#' @param y Response vector.
#'@param row.idx The integer vector of row indices of \code{X} that used for
#' fitting the model. \code{1:nrow(X)} by default.
#' @param alpha The elastic-net mixing parameter that controls the relative
#' contribution from the lasso and the ridge penalty. It must be a number
#' between 0 and 1. \code{alpha=1} is the lasso penalty and \code{alpha=0} the
#' ridge penalty.
#' @param gamma The tuning parameter for huberization smoothing of hinge loss.
#' Default is 0.01.
#' @param nlambda The number of lambda values.  Default is 100.
#' @param lambda.min The smallest value for lambda, as a fraction of
#' lambda.max, the data derived entry value.  Default is 0.001 if the number of
#' observations is larger than the number of variables and 0.01 otherwise.
#' @param lambda A user-specified sequence of lambda values. Typical usage is
#' to leave blank and have the program automatically compute a \code{lambda}
#' sequence based on \code{nlambda} and \code{lambda.min}. Specifying
#' \code{lambda} overrides this. This argument should be used with care and
#' supplied with a decreasing sequence instead of a single value. To get
#' coefficients for a single \code{lambda}, use \code{coef} or \code{predict}
#' instead after fitting the solution path with \code{sparseSVM}.  %or
#' performing k-fold CV with \code{cv.sparseSVM}.
#' @param preprocess Preprocessing technique to be applied to the input. Either
#' "standardize" (default), "rescale" or "none" (see \code{Details}). The
#' coefficients are always returned on the original scale.
#' @param screen Screening rule to be applied at each \code{lambda} that
#' discards variables for speed. Either "ASR" (default), "SR" or "none". "SR"
#' stands for the strong rule, and "ASR" for the adaptive strong rule. Using
#' "ASR" typically requires fewer iterations to converge than "SR", but the
#' computing time are generally close. Note that the option "none" is used
#' mainly for debugging, which may lead to much longer computing time.
#' @param max.iter Maximum number of iterations. Default is 1000.
#' @param eps Convergence threshold. The algorithms continue until the maximum
#' change in the objective after any coefficient update is less than \code{eps}
#' times the null deviance.  Default is \code{1E-7}.
#' @param dfmax Upper bound for the number of nonzero coefficients. The
#' algorithm exits and returns a partial path if \code{dfmax} is reached.
#' Useful for very large dimensions.
#' @param penalty.factor A numeric vector of length equal to the number of
#' variables. Each component multiplies \code{lambda} to allow differential
#' penalization. Can be 0 for some variables, in which case the variable is
#' always in the model without penalization.  Default is 1 for all variables.
#' @param message If set to TRUE, sparseSVM will inform the user of its
#' progress. This argument is kept for debugging. Default is FALSE.
#' @return The function returns an object of S3 class \code{"bigSVM"}, which
#' is a list containing: \item{call}{The call that produced this object.}
#' \item{weights}{The fitted matrix of coefficients.  The number of rows is
#' equal to the number of coefficients, and the number of columns is equal to
#' \code{nlambda}. An intercept is included.} \item{iter}{A vector of length
#' \code{nlambda} containing the number of iterations until convergence at each
#' value of \code{lambda}.} \item{saturated}{A logical flag for whether the
#' number of nonzero coefficients has reached \code{dfmax}.} \item{lambda}{The
#' sequence of regularization parameter values in the path.} \item{alpha}{Same
#' as above.} \item{gamma}{Same as above.} \item{penalty.factor}{Same as
#' above.} \item{nv}{The variable screening rules are accompanied with checks
#' of optimality conditions. When violations occur, the program adds in
#' violating variables and re-runs the inner loop until convergence. \code{nv}
#' is the number of violations.}
#' @author Yaohui Zeng and Congrui Yi
#' @keywords models classification machine learning SVM

bigSVM <- function (X, y, row.idx = 1:nrow(X), alpha = 1, gamma = 0.1, nlambda=100, 
                    lambda.min = ifelse(nrow(X)>ncol(X), 0.001, 0.01), lambda, 
                    preprocess = c("standardize", "rescale", "none"),
                    screen = c("ASR", "SR", "none"), max.iter = 1000, 
                    eps = 1e-5, dfmax = ncol(X)+1, penalty.factor=rep(1, ncol(X)), 
                    message = FALSE) {
  
  # Error checking
  preprocess <- match.arg(preprocess)
  screen <- match.arg(screen)
  if (alpha < 0 || alpha > 1) stop("alpha should be between 0 and 1")
  if (gamma < 0 || gamma > 1) stop("gamma should be between 0 and 1")
  if (missing(lambda) && nlambda < 2) stop("nlambda should be at least 2")
  if (length(penalty.factor)!=ncol(X)) stop("the length of penalty.factor should equal the number of columns of X")
  
  call <- match.call()
  # Include a column for intercept
  # n <- nrow(X)
  # p <- ncol(X)
  # XX <- cbind(rep(1,n), X)
  # penalty.factor <- c(0, penalty.factor) # no penalty for intercept term
  # p <- ncol(XX)
  
  if(missing(lambda)) {
    lambda <- double(nlambda)
    user <- 0
  } else {
    nlambda <- length(lambda)
    user <- 1
  }
  
  # Flag for preprocessing and screening
  ppflag = switch(preprocess, standardize = 1, rescale = 2, none = 0)
  scrflag = switch(screen, ASR = 1, SR = 2, none = 0)
  
  # subset of the response vector
  y <- y[row.idx]
  
  # Fitting
  fit <- big_svm(X@address, y, as.integer(row.idx - 1), lambda, as.integer(nlambda),
                 lambda.min, penalty.factor, gamma, alpha, eps, as.integer(ppflag),
                 as.integer(scrflag), as.integer(dfmax), as.integer(max.iter),
                 as.integer(user), as.integer(message));

  shift <- fit[[1]]
  scale <- fit[[2]]
  sx_pos <- fit[[3]]
  sx_neg <- fit[[4]]
  syx <- fit[[5]]
  nonconst <- fit[[6]]

  # Output
  structure(list(call = call,
                 shift = shift,
                 scale = scale,
                 sx_pos = sx_pos,
                 sx_neg = sx_neg,
                 syx = syx,
                 nonconst = nonconst),
            class = c("bigSVM"))
}
