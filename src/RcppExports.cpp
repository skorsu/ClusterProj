// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// active_inactive
Rcpp::List active_inactive(int K, arma::vec clus_assign);
RcppExport SEXP _ClusterProj_active_inactive(SEXP KSEXP, SEXP clus_assignSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type clus_assign(clus_assignSEXP);
    rcpp_result_gen = Rcpp::wrap(active_inactive(K, clus_assign));
    return rcpp_result_gen;
END_RCPP
}
// sample_clus
int sample_clus(arma::vec norm_probs, arma::uvec active_clus);
RcppExport SEXP _ClusterProj_sample_clus(SEXP norm_probsSEXP, SEXP active_clusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type norm_probs(norm_probsSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type active_clus(active_clusSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_clus(norm_probs, active_clus));
    return rcpp_result_gen;
END_RCPP
}
// log_density_gamma
double log_density_gamma(arma::rowvec y, arma::rowvec hyper_gamma_k);
RcppExport SEXP _ClusterProj_log_density_gamma(SEXP ySEXP, SEXP hyper_gamma_kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::rowvec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::rowvec >::type hyper_gamma_k(hyper_gamma_kSEXP);
    rcpp_result_gen = Rcpp::wrap(log_density_gamma(y, hyper_gamma_k));
    return rcpp_result_gen;
END_RCPP
}
// log_allocate_prob
arma::vec log_allocate_prob(int i, arma::vec current_assign, arma::vec xi, arma::mat y, arma::mat gamma_hyper_mat, arma::uvec active_clus);
RcppExport SEXP _ClusterProj_log_allocate_prob(SEXP iSEXP, SEXP current_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP gamma_hyper_matSEXP, SEXP active_clusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type current_assign(current_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type gamma_hyper_mat(gamma_hyper_matSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type active_clus(active_clusSEXP);
    rcpp_result_gen = Rcpp::wrap(log_allocate_prob(i, current_assign, xi, y, gamma_hyper_mat, active_clus));
    return rcpp_result_gen;
END_RCPP
}
// log_sum_exp
arma::vec log_sum_exp(arma::vec log_unnorm_prob);
RcppExport SEXP _ClusterProj_log_sum_exp(SEXP log_unnorm_probSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type log_unnorm_prob(log_unnorm_probSEXP);
    rcpp_result_gen = Rcpp::wrap(log_sum_exp(log_unnorm_prob));
    return rcpp_result_gen;
END_RCPP
}
// rdirichlet_cpp
arma::mat rdirichlet_cpp(int num_samples, arma::vec alpha_m);
RcppExport SEXP _ClusterProj_rdirichlet_cpp(SEXP num_samplesSEXP, SEXP alpha_mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type num_samples(num_samplesSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha_m(alpha_mSEXP);
    rcpp_result_gen = Rcpp::wrap(rdirichlet_cpp(num_samples, alpha_m));
    return rcpp_result_gen;
END_RCPP
}
// expand_step
Rcpp::List expand_step(int K, arma::vec old_assign, arma::vec alpha, arma::vec xi, arma::mat y, arma::mat gamma_hyper, double a_theta, double b_theta);
RcppExport SEXP _ClusterProj_expand_step(SEXP KSEXP, SEXP old_assignSEXP, SEXP alphaSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP gamma_hyperSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type gamma_hyper(gamma_hyperSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(expand_step(K, old_assign, alpha, xi, y, gamma_hyper, a_theta, b_theta));
    return rcpp_result_gen;
END_RCPP
}
// cluster_assign
Rcpp::List cluster_assign(int K, arma::vec old_assign, arma::vec xi, arma::mat y, arma::mat gamma_hyper, arma::vec alpha);
RcppExport SEXP _ClusterProj_cluster_assign(SEXP KSEXP, SEXP old_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP gamma_hyperSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type gamma_hyper(gamma_hyperSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(cluster_assign(K, old_assign, xi, y, gamma_hyper, alpha));
    return rcpp_result_gen;
END_RCPP
}
// split_merge
Rcpp::List split_merge(int K, arma::vec old_assign, arma::vec alpha, arma::vec xi, arma::mat y, arma::mat gamma_hyper, double a_theta, double b_theta, int sm_iter);
RcppExport SEXP _ClusterProj_split_merge(SEXP KSEXP, SEXP old_assignSEXP, SEXP alphaSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP gamma_hyperSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP, SEXP sm_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type gamma_hyper(gamma_hyperSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    Rcpp::traits::input_parameter< int >::type sm_iter(sm_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(split_merge(K, old_assign, alpha, xi, y, gamma_hyper, a_theta, b_theta, sm_iter));
    return rcpp_result_gen;
END_RCPP
}
// update_alpha
arma::vec update_alpha(int K, arma::vec alpha, arma::vec xi, arma::vec old_assign);
RcppExport SEXP _ClusterProj_update_alpha(SEXP KSEXP, SEXP alphaSEXP, SEXP xiSEXP, SEXP old_assignSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    rcpp_result_gen = Rcpp::wrap(update_alpha(K, alpha, xi, old_assign));
    return rcpp_result_gen;
END_RCPP
}
// cluster_func
Rcpp::List cluster_func(int K, int K_init, arma::mat y, arma::vec xi, arma::mat gamma_hyper, double a_theta, double b_theta, int sm_iter, int all_iter, bool print_iter, int iter_print);
RcppExport SEXP _ClusterProj_cluster_func(SEXP KSEXP, SEXP K_initSEXP, SEXP ySEXP, SEXP xiSEXP, SEXP gamma_hyperSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP, SEXP sm_iterSEXP, SEXP all_iterSEXP, SEXP print_iterSEXP, SEXP iter_printSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type K_init(K_initSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type gamma_hyper(gamma_hyperSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    Rcpp::traits::input_parameter< int >::type sm_iter(sm_iterSEXP);
    Rcpp::traits::input_parameter< int >::type all_iter(all_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type print_iter(print_iterSEXP);
    Rcpp::traits::input_parameter< int >::type iter_print(iter_printSEXP);
    rcpp_result_gen = Rcpp::wrap(cluster_func(K, K_init, y, xi, gamma_hyper, a_theta, b_theta, sm_iter, all_iter, print_iter, iter_print));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ClusterProj_active_inactive", (DL_FUNC) &_ClusterProj_active_inactive, 2},
    {"_ClusterProj_sample_clus", (DL_FUNC) &_ClusterProj_sample_clus, 2},
    {"_ClusterProj_log_density_gamma", (DL_FUNC) &_ClusterProj_log_density_gamma, 2},
    {"_ClusterProj_log_allocate_prob", (DL_FUNC) &_ClusterProj_log_allocate_prob, 6},
    {"_ClusterProj_log_sum_exp", (DL_FUNC) &_ClusterProj_log_sum_exp, 1},
    {"_ClusterProj_rdirichlet_cpp", (DL_FUNC) &_ClusterProj_rdirichlet_cpp, 2},
    {"_ClusterProj_expand_step", (DL_FUNC) &_ClusterProj_expand_step, 8},
    {"_ClusterProj_cluster_assign", (DL_FUNC) &_ClusterProj_cluster_assign, 6},
    {"_ClusterProj_split_merge", (DL_FUNC) &_ClusterProj_split_merge, 9},
    {"_ClusterProj_update_alpha", (DL_FUNC) &_ClusterProj_update_alpha, 4},
    {"_ClusterProj_cluster_func", (DL_FUNC) &_ClusterProj_cluster_func, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_ClusterProj(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
