
#include <RcppArmadillo.h>
#include "bigmemory/BigMatrix.h"
#include <time.h>
#include "bigmemory/BigMatrix.h"
#include "bigmemory/MatrixAccessor.hpp"
#include "bigmemory/bigmemoryDefines.h"

#include "bigSVM_omp.h"

#ifndef UTILITIES_H
#define UTILITIES_H

using namespace Rcpp;
using namespace std;

double sign(double x);

double sum(double *x, int n);

// crossprod of columns X_j and X_k
template<typename T, typename BMAccessorType>
double crossprod_bm_Xj_Xk(BigMatrix *xMat, IntegerVector& row_idx, int n, int j, int k) {
  BMAccessorType xAcc(*xMat);
  T *xCol_j = xAcc[j];
  T *xCol_k = xAcc[k];
  double sum_xj_xk = 0.0;
  
  for (int i = 0; i < n; i++) {
    sum_xj_xk += (double)xCol_j[row_idx[i]] * xCol_k[row_idx[i]];
  }
  return sum_xj_xk;
}

#endif
