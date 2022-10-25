#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// User defined functions: -----------------------------------------------------

// Step 1: Update the cluster space: -------------------------------------------

// Step 2: Allocate the observation to the existing clusters: ------------------
// Assume that all clusters share the same hyperparameters (gamma)

// [[Rcpp::export]]
arma::vec cluster_assign(arma::vec clus_assign, arma::vec clus_hyper, 
                         arma::mat y, arma::vec data_hyper){
  
  // Input: a previous assignment, cluster weight's hyperparameter, 
  //        data, hyperparameter for each data cluster
  
  // Output: a vector consisted of a new assignment
  arma::vec new_assign = clus_assign;
  arma::vec unique_clus = unique(clus_assign);
  int K_pos = unique_clus.size(); 
  
  // Start the process
  for(int i = 0; i < y.n_rows; i++){
    arma::vec assign_prob = -1 * arma::ones(K_pos); // Assignment probability vector
    
    // Split the data into two sets -- (1) observation i, (2) not an observation i
    arma::vec obs_i = arma::conv_to<arma::vec>::from((y.row(i)));
    arma::vec clus_not_i = new_assign; clus_not_i.shed_row(i);
    arma::mat obs_not_i = y; obs_not_i.shed_row(i);

    // Calculate the first term
    Rcpp::NumericVector x = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(obs_i)); // arma::vec -> Rcpp::NumericVector
    arma::vec x_fact = factorial(x);
    Rcpp::NumericVector x_sum = {sum(x)};
    arma::vec first_t = {factorial(x_sum)/arma::prod(x_fact)};
    
    int n_not_i = clus_not_i.size();
    double sum_hyper_clus = sum(clus_hyper);
    // Calculate the second, third, forth term of the allocation probability
    for(int j = 1; j <= K_pos; j++){ // j is the cluster index
      // Cluster Indicator Vector
      arma::uvec clus_index = find(clus_not_i == j);
      // Sum of each column in that cluster
      arma::vec sum_col = arma::conv_to<arma::vec>::from(sum(obs_not_i.rows(clus_index), 0));
      
      // Second Term
      Rcpp::NumericVector col_gam = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(sum_col + data_hyper));
      arma::vec col_gam_gamma = gamma(col_gam);
      Rcpp::NumericVector sum_col_gam = {sum(col_gam)};
      arma::vec second_t = {gamma(sum_col_gam)/arma::prod(col_gam_gamma)};
      
      // Third Term
      Rcpp::NumericVector i_col_gam = x + col_gam;
      arma::vec i_col_gam_gamma = gamma(i_col_gam);
      Rcpp::NumericVector sum_i_col_gam = {sum(i_col_gam)};
      arma::vec third_t = {arma::prod(i_col_gam_gamma)/gamma(sum_i_col_gam)};
      
      // Forth Term
      double fouth_t = (clus_index.size() + clus_hyper[(j-1)])/(n_not_i + sum_hyper_clus);
      assign_prob[(j-1)] = as_scalar(first_t * second_t * third_t * fouth_t);
    }
    
    // Reassign
    arma::vec norm_assign_prob = arma::normalise(assign_prob, 1);
    arma::imat C = arma::imat(K_pos, 1);
    rmultinom(1, norm_assign_prob.begin(), K_pos, C.colptr(0));
    int clus_new = arma::conv_to<int>::from(find(C == 1));
    new_assign.row(i) = clus_new + 1;
  }
  
  return new_assign;
}


// Testing Area: ---------------------------------------------------------------
// Test: Multinomial Distribution
// [[Rcpp::export]]
Rcpp::List test_fn(arma::vec probs){
  Rcpp::List result;
  int n = 1; // First parameter for the Multinomial distribution
  int k = 10; // Number of group
  arma::imat C = arma::imat(k, 1);
  // C++ indices start at 0
  rmultinom(n, probs.begin(), k, C.colptr(0));
  int val = 0;
  for(int i = 0; i < k; i++){
    if(C(i, 0) == 1){
      val = (i + 1);
      break;
    }
  }
  
  result["MU"] = C;
  result["clus"] = val;
  
  return result;
}

// [[Rcpp::export]]
Rcpp::NumericVector foo(double t, Rcpp::NumericVector k) {
  Rcpp::NumericVector x = factorial(k);
  return x;
}













































