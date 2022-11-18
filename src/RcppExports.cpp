// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// expand_step
Rcpp::List expand_step(int K, arma::vec old_assign, arma::vec alpha, arma::vec xi, double a_theta, double b_theta);
RcppExport SEXP _ClusterProj_expand_step(SEXP KSEXP, SEXP old_assignSEXP, SEXP alphaSEXP, SEXP xiSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(expand_step(K, old_assign, alpha, xi, a_theta, b_theta));
    return rcpp_result_gen;
END_RCPP
}
// cluster_assign
Rcpp::List cluster_assign(int K, arma::vec old_assign, arma::vec xi, arma::mat y, arma::vec gamma_hyper, arma::vec alpha);
RcppExport SEXP _ClusterProj_cluster_assign(SEXP KSEXP, SEXP old_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP gamma_hyperSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma_hyper(gamma_hyperSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(cluster_assign(K, old_assign, xi, y, gamma_hyper, alpha));
    return rcpp_result_gen;
END_RCPP
}
// split_merge
Rcpp::List split_merge(int K, arma::vec old_assign, arma::vec psi, arma::vec xi, arma::mat y, arma::vec gamma_hyper, double a_theta, double b_theta);
RcppExport SEXP _ClusterProj_split_merge(SEXP KSEXP, SEXP old_assignSEXP, SEXP psiSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP gamma_hyperSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type psi(psiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma_hyper(gamma_hyperSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(split_merge(K, old_assign, psi, xi, y, gamma_hyper, a_theta, b_theta));
    return rcpp_result_gen;
END_RCPP
}
// cluster_func
Rcpp::List cluster_func(int K, arma::vec old_assign, arma::vec psi, arma::vec xi, arma::mat y, arma::vec gamma_hyper, double a_theta, double b_theta, int iter);
RcppExport SEXP _ClusterProj_cluster_func(SEXP KSEXP, SEXP old_assignSEXP, SEXP psiSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP gamma_hyperSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type psi(psiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma_hyper(gamma_hyperSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(cluster_func(K, old_assign, psi, xi, y, gamma_hyper, a_theta, b_theta, iter));
    return rcpp_result_gen;
END_RCPP
}
// test_fn
Rcpp::List test_fn(arma::vec probs);
RcppExport SEXP _ClusterProj_test_fn(SEXP probsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type probs(probsSEXP);
    rcpp_result_gen = Rcpp::wrap(test_fn(probs));
    return rcpp_result_gen;
END_RCPP
}
// foo
Rcpp::NumericVector foo(double t, Rcpp::NumericVector k);
RcppExport SEXP _ClusterProj_foo(SEXP tSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type t(tSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(foo(t, k));
    return rcpp_result_gen;
END_RCPP
}
// test_List
arma::vec test_List(arma::vec test_1);
RcppExport SEXP _ClusterProj_test_List(SEXP test_1SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type test_1(test_1SEXP);
    rcpp_result_gen = Rcpp::wrap(test_List(test_1));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ClusterProj_expand_step", (DL_FUNC) &_ClusterProj_expand_step, 6},
    {"_ClusterProj_cluster_assign", (DL_FUNC) &_ClusterProj_cluster_assign, 6},
    {"_ClusterProj_split_merge", (DL_FUNC) &_ClusterProj_split_merge, 8},
    {"_ClusterProj_cluster_func", (DL_FUNC) &_ClusterProj_cluster_func, 9},
    {"_ClusterProj_test_fn", (DL_FUNC) &_ClusterProj_test_fn, 1},
    {"_ClusterProj_foo", (DL_FUNC) &_ClusterProj_foo, 2},
    {"_ClusterProj_test_List", (DL_FUNC) &_ClusterProj_test_List, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_ClusterProj(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
