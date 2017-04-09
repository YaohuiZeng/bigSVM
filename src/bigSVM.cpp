
#include "utilities.h"

// Semismooth Newton Coordinate Descent (SNCD) for lasso/elastic-net regularized SVM
template<typename T, typename BMAccessorType>
List big_svm(BigMatrix *xMat, const NumericVector& y, const IntegerVector& row_idx,
             NumericVector& lambda, int nlambda, double lambda_min, NumericVector pf,
             double gamma, double alpha, double thresh, int ppflag, int scrflag, 
             int dfmax, int max_iter, int user, int message, int n, int p, int L) {
  
  BMAccessorType xAcc(*xMat);
  int i, j, k, l, lstart, lp, jn, num_pos, converged, mismatch, nnzero = 0;
  double csum_p, csum_n, csum, pct, lstep, ldiff, lmax, l1, l2, v1, v2, v3, tmp, change, max_update, update, strfactor = 1.0;  
  
  // double *x2 = Calloc(n*p, double); // x^2
  NumericVector shift(p);
  NumericVector scale(p);
  NumericVector sx_pos(p); // column sum of x where y = 1
  NumericVector sx_neg(p); // column sum of x where y = -1
  // double *yx = Calloc(n*p, double); // elementwise products: y[i] * x[i][j]
  NumericVector syx(p); // column sum of yx

  NumericVector w_old(p);
  NumericVector r(p);
  NumericVector s(p);
  NumericVector d1(p);
  NumericVector d2(p);
  NumericVector z(p);
  IntegerVector include(p);
  IntegerVector nonconst(p);
 
  double cutoff;
  
  // standardization
  standardize<T, BMAccessorType>(shift, scale, sx_pos, sx_neg, syx, nonconst, xMat, y, row_idx, n, p);

  return List::create(shift, scale, sx_pos, sx_neg, syx, nonconst);
  
  
}


// Dispatch function for big_svm
// [[Rcpp::export]]
List big_svm(SEXP X_, SEXP y_, SEXP row_idx_, SEXP lambda_, SEXP nlambda_,
             SEXP lambda_min_, SEXP pf_, SEXP gamma_, SEXP alpha_, 
             SEXP thresh_, SEXP ppflag_, SEXP scrflag_, SEXP dfmax_,
             SEXP max_iter_, SEXP user_, SEXP message_) {
  
  XPtr<BigMatrix> xMat(X_);
  NumericVector y(y_);
  IntegerVector row_idx(row_idx_);
  NumericVector lambda(lambda_);
  int nlambda = INTEGER(nlambda_)[0];
  double lambda_min = REAL(lambda_min_)[0];
  NumericVector pf(pf_);
  double gamma = REAL(gamma_)[0];
  double alpha = REAL(alpha_)[0];
  double thresh = REAL(thresh_)[0];
  int ppflag = INTEGER(ppflag_)[0];
  int scrflag = INTEGER(scrflag_)[0];
  int dfmax = INTEGER(dfmax_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  int user= INTEGER(user_)[0];
  int message = INTEGER(message_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = nlambda;
  
  switch(xMat->matrix_type()) {
  case 1:
    return big_svm<char, MatrixAccessor<char>>(
        xMat, y, row_idx, lambda, nlambda, lambda_min, pf, gamma, alpha, thresh,
        ppflag, scrflag, dfmax, max_iter, user, message, n, p, L);
  case 2:
    return big_svm<short, MatrixAccessor<short>>(
        xMat, y, row_idx, lambda, nlambda, lambda_min, pf, gamma, alpha, thresh,
        ppflag, scrflag, dfmax, max_iter, user, message, n, p, L);
  case 4:
    return big_svm<int, MatrixAccessor<int>>(
        xMat, y, row_idx, lambda, nlambda, lambda_min, pf, gamma, alpha, thresh,
        ppflag, scrflag, dfmax, max_iter, user, message, n, p, L);
  case 6:
    return big_svm<float, MatrixAccessor<float>>(
        xMat, y, row_idx, lambda, nlambda, lambda_min, pf, gamma, alpha, thresh,
        ppflag, scrflag, dfmax, max_iter, user, message, n, p, L);
  case 8:
    return big_svm<double, MatrixAccessor<double>>(
        xMat, y, row_idx, lambda, nlambda, lambda_min, pf, gamma, alpha, thresh,
        ppflag, scrflag, dfmax, max_iter, user, message, n, p, L);
  default:
    throw Rcpp::exception("unknown type detected for big.matrix object!"); 
  }
}
