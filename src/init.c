#include <R.h>
#include <Rinternals.h> // for SEXP
#include <stdlib.h> //
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>  // optional

extern SEXP bigSVM_big_svm(SEXP X_SEXP, SEXP y_SEXP, SEXP row_idx_SEXP, SEXP lambda_SEXP, 
                           SEXP nlambda_SEXP, SEXP lambda_min_SEXP, SEXP pf_SEXP, 
                           SEXP gamma_SEXP, SEXP alpha_SEXP, SEXP thresh_SEXP, 
                           SEXP ppflag_SEXP, SEXP scrflag_SEXP, SEXP dfmax_SEXP, 
                           SEXP max_iter_SEXP, SEXP user_SEXP, SEXP message_SEXP);

static R_CallMethodDef callMethods[] = {
  {"bigSVM_big_svm", (DL_FUNC) &bigSVM_big_svm, 16},
  {NULL, NULL, 0}
};

void R_init_bigSVM(DllInfo *dll) {
  R_registerRoutines(dll,NULL,callMethods,NULL,NULL);
  R_useDynamicSymbols(dll, FALSE);
}
