
#include "utilities.h"

void postprocess(arma::sp_mat &w, NumericVector &w0, NumericVector &shift, 
                 NumericVector &scale, IntegerVector &nonconst, int p, int L) {
  int j, l;
  double prod;
  for (l = 0; l < L; l++) {
    prod = 0.0;
    for (j = 0; j < p; j++) {
      if (nonconst[j]) {
        w(j, l) = w(j, l) / scale[j];
        prod += shift[j] * w(j, l);
      }
    }
    w0[l] -= prod;
  }
}

// Semismooth Newton Coordinate Descent (SNCD) for lasso/elastic-net regularized SVM
template<typename T, typename BMAccessorType>
List big_svm(BigMatrix *xMat, const NumericVector &y, const IntegerVector &row_idx,
             NumericVector &lambda, int nlambda, double lambda_min, NumericVector &pf,
             double gamma, double alpha, double thresh, int ppflag, int scrflag, 
             int dfmax, int max_iter, int user, int message, int n, int p, int L) {
  
  BMAccessorType xAcc(*xMat);  
  T *xCol;
  
  int i, j, k, l, lstart, lp, jn, num_pos = 0, converged, mismatch, nnzero = 0, violations = 0, nv = 0;
  double gi = 1.0 / gamma, cmax, cmin, csum_pos, csum_neg, csum, pct, lstep, ldiff, lmax, l1, l2, v1, v2, v3, tmp, xtmp, change, max_update, update, cutoff, scrfactor = 1.0;  
  
  // double *x2 = Calloc(n*p, double); // x^2
  NumericVector shift(p);
  NumericVector scale(p);
  NumericVector sx_pos(p); // column sum of x where y = 1
  NumericVector sx_neg(p); // column sum of x where y = -1
  NumericVector syx(p); // column sum of yx
  double sx_pos_int, sx_neg_int, syx_int = 0.0; // for intercept
  
  arma::sp_mat w = arma::sp_mat(p, L); // w without intercept
  NumericVector w0(L); // intercept
  NumericVector w_old(p); // previous w without intercept
  double w0_old = 0.0; // previous intercept
  int saturated = 0;
  IntegerVector iter(L);
  NumericVector r(p);
  NumericVector s(p);
  NumericVector d1(p);
  NumericVector d2(p);
  NumericVector z(p);
  IntegerVector include(p);
  IntegerVector nonconst(p);
 
  // standardization
  standardize<T, BMAccessorType>(shift, scale, sx_pos, sx_neg, syx, nonconst, xMat, y, row_idx, n, p);

  // scrflag = 2: sequential strong rule (SSR), scrfactor = 1
  for (j = 0; j < p; j++) {
    if (!pf[j] && nonconst[j]) {
      include[j] = 1;
    }
  }
  
  // Initialization
  for (i = 0; i < n; i++) {
    syx_int += y[i];
    if (y[i] > 0) num_pos++;
  }
  sx_pos_int = num_pos;
  sx_neg_int = n - num_pos;
  
  if (2 * num_pos > n) {
    // initial intercept = 1
    w0 = 1.0;
    w0_old = 1.0;
    for (i = 0; i < n; i++) {
      if (y[i] > 0) {
        r[i] = 0.0;
        d1[i] = 0.0;
        d2[i] = gi;
      } else {
        r[i] = 2.0;
        d1[i] = 1.0;
        d2[i] = 0.0;
      }
    } 
  } else {
    // initial intercept = -1
    w0 = -1.0;
    w0_old = -1.0;
    for (i = 0; i < n; i++) {
      if (y[i] > 0) {
        r[i] = 2.0;
        d1[i] = 1.0;
        d2[i] = 0.0;
      } else {
        r[i] = 0.0;
        d1[i] = 0.0;
        d2[i] = gi;
      }
    }
  }
  
  // set up lambda sequence on log-scale
  if (user == 0) {
    lmax = 0.0;
    for (j = 0; j < p; j++) {
      if (nonconst[j]) {
        if (2*num_pos > n) {
          z[j] = (2 * sx_neg[j] - sx_pos[j]) / (2*n);
        } else {
          z[j] = (2*sx_pos[j] - sx_neg[j]) / (2*n);
        }
        if (pf[j]) {
          tmp = fabs(z[j]) / pf[j];
          if (tmp > lmax) lmax = tmp;
        }
      }
    }
    lmax = lmax / alpha;
    lambda[0] = lmax;
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min) / (nlambda - 1);
    for (l = 1; l < nlambda; l++) lambda[l] = lambda[l-1] * exp(lstep);
    lstart = 1;
  } else {
    lstart = 0;
  }
  
  // solution path
  for (l = lstart; l < nlambda; l++) {
    if (saturated) break;
    if (message) Rprintf("Lambda %d\n", l+1);

    // variable screening    
    l1 = lambda[l] * alpha;
    l2 = lambda[l] * (1.0 - alpha);
    if (l != 0) {
      cutoff = alpha * (2*lambda[l] - lambda[l-1]);
      ldiff = lambda[l-1] - lambda[l];
    } else {
      cutoff = alpha * lambda[0];
      ldiff = 1.0;
    }
    for (j = 0; j < p; j++) {
      if (!include[j] && nonconst[j] && fabs(z[j]) > cutoff * pf[j]) include[j] = 1;
    }
    
    while (iter[l] < max_iter) {
      // check dfmax
      if (nnzero > dfmax) {
        for (int ll = l; ll < nlambda; ll++) iter[ll] = NA_INTEGER;
        saturated = 1;
        break;
      }
      
      // solve KKT equations over eligible set
      while(iter[l] < max_iter) {
        iter[l]++;
        mismatch = 0; max_update = 0.0;
        
        // TODO: need to separte the update for intercept and variables.
        for (j = 0; j < p; j++) {
          if (include[j]) {
            
            xCol = xAcc[j];
            for (k = 0; k < 5; k++) {
              update = 0.0; mismatch = 0; v1 = 0.0; v2 = 0.0; pct = 0.0;
              for (i = 0; i < n; i++) {
                xtmp = ((double)xCol[row_idx[i]] - shift[j]) / scale[j];
                v1 += y[i] * xtmp * d1[i];
                v2 += pow(xtmp, 2) * d2[i];
                pct += d2[i];
              }
            }
            pct *= gamma / n; // percent of residuals with absolute values below gamma
            if (pct < 0.05 || pct < 1.0 / n) {
              // approximate v2 with a continuation technique
              for (i = 0; i < n; i++) {
                tmp = fabs(r[i]);
                xtmp = ((double)xCol[row_idx[i]] - shift[j]) / scale[j];
                if (tmp > gamma) v2 += pow(xtmp, 2) / tmp;
              }
            }
            v1 = (v1 + syx[j]) / (2.0 * n);
            v2 /= 2.0 * n;
            
            // update w_j
            if (pf[j] == 0.0) {
              // unpenalized
              w(j, l) = w_old[j] + v1 / v2;
            } else if (fabs(w_old[j] + s[j]) > 1.0){
              s[j] = sign(fabs(w_old[j] + s[j]));
              w(j, l) = w_old[j] + (v1 - l1 * pf[j] * s[j] - l2 * pf[j] * w_old[j]) / (v2 + l2 * pf[j]);
            } else {
              s[j] = (v1 + v2 * w_old[j]) / (l1 * pf[j]);
              w(j, l) = 0.0;
            }
            
            // mismatch between beta and s
            if (pf[j] > 0) {
              if (fabs(s[j]) > 1 || (w(j, l) != 0.0 && s[j] != sign(w(j, l)))) mismatch = 1;
            }
            
            // Update r, d1, d2 and compute candidate of max_update
            change = w(j, l) - w_old[j];
            if (change > 1e-6) {
              for (i = 0; i < n; i++) {
                xtmp = ((double)xCol[row_idx[i]] - shift[j]) / scale[j];
                r[i] -= y[i] * xtmp * change;
                if (fabs(r[i]) > gamma) {
                  d1[i] = sign(r[i]);
                  d2[i] = 0.0;
                } else {
                  d1[i] = r[i] * gi;
                  d2[i] = gi;
                }
              }
              update = (v2 + l2 * pf[j]) * pow(change, 2);
              if (update > max_update) max_update = update;
              w_old[j] = w(j, l);
            }
            if (!mismatch && update < thresh) break;
          }
        }
      }
      
      // check convergence
      if (max_update < thresh) break;
    }
    
    // scan for violations 
    violations = 0; nnzero = 0;
    for (j = 0; j < p; j++) {
      if (!include[j] && nonconst[j]) {
        v1 = 0.0;
        for (i = 0; i < n; i++) {
          xtmp = ((double)xCol[row_idx[i]] - shift[j]) / scale[j];
          v1 += y[i] * xtmp * d1[i];
        }
        v1 = (v1 + syx[j]) / (2.0 * n);
        // check KKT
        if (fabs(v1) > l1 * pf[j]) {
          include[j] = 1;
          s[j] = v1 / (l1 * pf[j]);
          violations++;
          if (message) Rprintf("+V%d", j);
        } 
        z[j] = v1;
      }
      if (w(j, l) != 0.0) nnzero++;
    }
    
    if (message) Rprintf("# iterations = %d\n", iter[l]);
    if (violations == 0) break;
    nv += violations;
  }
  
  if (scrflag != 0 && message) Rprintf("# violations detected and fixed: %d\n", nv);
  if (ppflag) postprocess(w, w0, shift, scale, nonconst, p, L);
  
  return List::create(w0, w, iter, lambda, saturated, shift, scale, nonconst);
  
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
