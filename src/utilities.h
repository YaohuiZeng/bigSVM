
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

// template<typename T>
// T sum(T *x, int n);
template<typename T, typename BMAccessorType>
double crossprod_bm_Xj_Xk(XPtr<BigMatrix> xMat, IntegerVector& row_idx, int n, int j, int k);
#endif
