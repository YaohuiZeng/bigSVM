
#include <RcppArmadillo.h>
#include "bigmemory/BigMatrix.h"
#include <time.h>
#include "bigmemory/MatrixAccessor.hpp"
#include "bigmemory/bigmemoryDefines.h"

#include "bigSVM_omp.h"

#ifndef UTILITIES_H
#define UTILITIES_H

using namespace Rcpp;
using namespace std;

double sign(double x);

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

// standardization of features
template<typename T, typename BMAccessorType>
void standardize(NumericVector &shift, NumericVector &scale, NumericVector &sx_pos, 
                 NumericVector &sx_neg, NumericVector &syx, IntegerVector &nonconst,
                 BigMatrix *xMat, const NumericVector &y, const IntegerVector &row_idx,
                 int n, int p) {
  BMAccessorType xAcc(*xMat);
  int i, j;
  double csum_pos, csum_neg, csum, sum_xy, sum_y;
  T *xCol;
  for (j = 0; j < p; j++) {
    sum_xy = 0.0; sum_y = 0.0; csum_pos = 0.0; csum_neg = 0.0; csum = 0.0; 
    xCol = xAcc[j];
   
    for (i = 0; i < n; i++) {
      shift[j] += (double)xCol[row_idx[i]];
    }
    shift[j] = shift[j] / n; // center
    for (i = 0; i < n; i++) scale[j] += pow((double)xCol[row_idx[i]] - shift[j], 2);
    scale[j] = sqrt(scale[j] / n); // scale
    
    if (scale[j] > 1e-6) {
      nonconst[j] = 1;
      for (i = 0; i < n; i++) {
        if (y[i] > 0) {
          csum_pos = csum_pos + ((double)xCol[row_idx[i]] - shift[j]);
        } else {
          csum_neg = csum_neg + ((double)xCol[row_idx[i]] - shift[j]);
        }
        sum_xy += (double)xCol[row_idx[i]] * y[i];
        sum_y += y[i];
      }
      sx_pos[j] = csum_pos / scale[j];
      sx_neg[j] = csum_neg / scale[j];
      syx[j] = (sum_xy - shift[j] * sum_y) / scale[j];
    }
  }
  
}


#endif
