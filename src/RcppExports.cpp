// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// crossprod_bm
NumericVector crossprod_bm(SEXP X_, SEXP y_, SEXP row_idx_);
RcppExport SEXP bigSVM_crossprod_bm(SEXP X_SEXP, SEXP y_SEXP, SEXP row_idx_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type X_(X_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type y_(y_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type row_idx_(row_idx_SEXP);
    rcpp_result_gen = Rcpp::wrap(crossprod_bm(X_, y_, row_idx_));
    return rcpp_result_gen;
END_RCPP
}