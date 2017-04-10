
#' Solution Paths for Sparse Linear Support Vector Machine with Lasso or
#' ELastic-Net Regularization
#' 
#' Fast algorithm for fitting solution paths of sparse linear SVM with 
#' lasso or elastic-net regularization to generate sparse solutions and extend 
#' model fitting with big data that cannot be fully loaded into main memory.
#' 
#' \tabular{ll}{ Package: \tab bigSVM\cr Type: \tab Package\cr Version: \tab
#' 1.0-1\cr Date: \tab 2017-04-07\cr License: \tab GPL-3\cr } Very simple to
#' use. Accepts \code{X,y} data for binary classification where \code{y}
#' belongs to +1, -1, and produces the regularized solution path over a grid of
#' values for the regularization parameter \code{lambda}.
#' 
#' @name bigSVM-package
#' @docType package
#' 
#' @useDynLib bigSVM, .registration = TRUE
#' @import parallel bigmemory 
#' @importFrom Matrix Matrix crossprod
#' @importFrom graphics plot abline arrows axis mtext par points
#' @importFrom stats coef predict approx lm quantile sd update
#' @importFrom grDevices hcl
#' @importFrom Rcpp evalCpp
#' @export bigSVM 
#'
#' @author Yaohui Zeng and Congrui Yi
#' @keywords models classification machine learning SVM
NULL
