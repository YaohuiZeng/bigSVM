
#include "utilities.h"

double sign(double x) {
  if(x > 0.00000000001) return 1.0;
  else if(x < -0.00000000001) return -1.0;
  else return 0.0;
}

int sum(int *x, int n) {
  int sum = 0;
  for (int j = 0; j < n; j++) {
    sum += x[j];
  }
  return sum;
}


// [[Rcpp::export]] 
NumericVector crossprod_bm(SEXP X_, SEXP y_, SEXP row_idx_) {
  XPtr<BigMatrix> xMat(X_);
  NumericVector y(y_);
  IntegerVector row_idx(row_idx_);
  int p = xMat->ncol();
  int n = xMat->nrow();
  
  NumericVector res(p);
  
  switch (xMat->matrix_type()) {
  case 2:
    for(int j = 0; j < p; j++) {
      res[j] = crossprod_bm_Xj_Xk<short, MatrixAccessor<short>>(xMat, row_idx, n, j, 0);
    }
    break;
  case 4:
    for(int j = 0; j < p; j++) {
      res[j] = crossprod_bm_Xj_Xk<int, MatrixAccessor<int>>(xMat, row_idx, n, j, 0);
    }
    break;
  case 6:
    for(int j = 0; j < p; j++) {
      res[j] = crossprod_bm_Xj_Xk<float, MatrixAccessor<float>>(xMat, row_idx, n, j, 0);
    }
    break;
  case 8:
    for(int j = 0; j < p; j++) {
      res[j] = crossprod_bm_Xj_Xk<double, MatrixAccessor<double>>(xMat, row_idx, n, j, 0);
    }
    break;
  }
  
  return res;
}

