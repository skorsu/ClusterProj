#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// Note to self: ---------------------------------------------------------------
// - Not reset the cluster index. (Maybe lower computational cost.)
// - K is fixed. (We should include K as one of our input if necessary.)
// - We should get the index of the existed cluster first in most step.
// - For the contract step, still confused :(
// - For the contract step, should we merge the smaller cluster to the larger or not?
// - Right now, I did not consider the size of the clusters. (Contract Step)
// - psi vector is K dimension.
// - Fix Code: Step 2

// Questions: ------------------------------------------------------------------

// User defined functions: -----------------------------------------------------

// Step 1: Update the cluster space: -------------------------------------------
// [[Rcpp::export]]
Rcpp::List expand_cluster(int K, Rcpp::IntegerVector inactive_clus, 
                          arma::uvec active_clus, arma::vec old_assign, 
                          arma::vec psi, arma::vec xi, double a_theta, 
                          double b_theta){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), inactive clusters, active clusters, 
   *        previous cluster assignment, previous cluster weight (psi), 
   *        hyperparameter for cluster (xi), hyperparameter (a_theta, b_theta)
   * Output: new cluster weight, updated cluster assignment.
   */
  
  // Select the cluster that we will expand
  Rcpp::IntegerVector d_new_clus = Rcpp::sample(inactive_clus, 1);
  int new_clus = d_new_clus[0];
  
  // Convert the cluster weight (psi) to the alpha.
  // Since we assume that alpha_j ~ Gamma(xi_j, 1), alpha+ ~ Gamma(sum_xi, 1).
  arma::vec xi_active = xi.rows(active_clus - 1);
  double alpha_p = R::rgamma(sum(xi_active), 1);
  arma::vec alpha_vec = alpha_p * psi;
  double sum_active_alpha = sum(alpha_vec);
  
  // Sample alpha for new active cluster
  alpha_vec.at(new_clus - 1) = R::rgamma(xi.at(new_clus - 1), 1);
  
  // Calculate the acceptance probability
  arma::vec accept_prob = (alpha_vec.at(new_clus - 1)/alpha_vec) * 
    (sum_active_alpha/(sum_active_alpha + alpha_vec.at(new_clus - 1))) * 
    (a_theta/b_theta);
  arma::vec A = arma::min(accept_prob, arma::ones(alpha_vec.size()));
  
  // Assign a new cluster
  arma::vec new_assign = -1 * arma::ones(old_assign.size());
  for(int i = 0; i < old_assign.size(); i++){
    double u = arma::randu();
    if(u <= A.at(old_assign.at(i) - 1)){
      new_assign.at(i) = new_clus;
    } else{
      new_assign.at(i) = old_assign.at(i);
    }
  }
  
  // Adjust the psi vector
  arma::vec new_alpha = arma::zeros(K);
  arma::vec new_active_clus = arma::unique(new_assign);
  arma::uvec index_active = arma::conv_to<arma::uvec>::from(new_active_clus) - 1;
  new_alpha.elem(index_active) = alpha_vec.elem(index_active);
  arma::vec new_phi = arma::normalise(new_alpha, 1);
  
  // Return the result
  result["new_assign"] = new_assign;
  result["new_phi"] = new_phi;

  return result;
}

// [[Rcpp::export]]
Rcpp::List contract_cluster(int K, arma::uvec active_clus, 
                            arma::vec old_assign, arma::vec psi, arma::vec xi, 
                            double a_theta, double b_theta){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), active clusters, previous cluster assignment, 
   *        previous cluster weight (psi), hyperparameter for cluster (xi), 
   *        hyperparameter (a_theta, b_theta)
   * Output: new cluster weight, updated cluster assignment.
   */
  
  // Select two clusters from the existed cluster space
  Rcpp::IntegerVector candidate_clus = 
    Rcpp::sample(Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(active_clus)), 2);
  
  // Convert the cluster weight (psi) to the alpha. (Same logic as expand func.)
  arma::vec xi_active = xi.rows(active_clus - 1);
  double alpha_p = R::rgamma(sum(xi_active), 1);
  arma::vec alpha_vec = alpha_p * psi;
  
  // Calculate the acceptance probability
  int clus_1 = candidate_clus[0];
  int clus_2 = candidate_clus[1];
  double accept_prob = (alpha_vec[clus_2 - 1]/alpha_vec[clus_1 - 1]) *
    (sum(alpha_vec)/(sum(alpha_vec) - alpha_vec[clus_1 - 1])) *
    (b_theta/a_theta);
  double A = std::min(accept_prob, 1.0);
  
  // Reassign only the element from cluster # 1
  arma::vec new_assign = old_assign;
  arma::uvec reassign_elem = find(old_assign == clus_1);
  for(int j  = 0; j < reassign_elem.size(); j ++){
    double u = R::runif(0.0, 1.0);
    if(u <= A){
      new_assign.at(reassign_elem[j]) = clus_2;
    }
  }
  
  // Adjust the psi vector
  arma::vec new_alpha = arma::zeros(K);
  arma::vec new_active_clus = arma::unique(new_assign);
  arma::uvec index_active = arma::conv_to<arma::uvec>::from(new_active_clus) - 1;
  new_alpha.elem(index_active) = alpha_vec.elem(index_active);
  arma::vec new_phi = arma::normalise(new_alpha, 1);
  
  // Return the result
  result["new_assign"] = new_assign;
  result["new_phi"] = new_phi;
  
  return result;
}

// [[Rcpp::export]]
Rcpp::List expand_contract(int K, arma::vec old_assign, arma::vec psi,
                           arma::vec xi, double a_theta, double b_theta){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (psi), hyperparameter for cluster (xi),
   *        hyperparameter (a_theta, b_theta).
   * Output: new cluster weight, updated cluster assignment.
   */ 
  
  // Indicate the existed clusters and inactive clusters
  Rcpp::IntegerVector all_possible = Rcpp::seq(1, K);
  Rcpp::IntegerVector existed_clus = Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(arma::unique(old_assign)));
  Rcpp::IntegerVector inactive_clus = Rcpp::setdiff(all_possible, existed_clus);
  
  // Decide Expand or Contract. (1 = Expand; 0 = Contract)
  int expand_indi;
  int n_existed = existed_clus.size();
  if(n_existed == 1){
    expand_indi = 1;
  } else if(n_existed == K){
    expand_indi = 0;
  } else {
    expand_indi = R::rbinom(1, 0.5);
  }
  
  // MH algorithm
  if(expand_indi == 1){
    Rprintf("-- Expand -- \n");
    // Convert all component to fit in the function.
    arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(old_assign));
    result = expand_cluster(K, inactive_clus, active_clus, old_assign, psi, 
                            xi, a_theta, b_theta);
  } else {
    Rprintf("-- Contract -- \n");
    // Convert all component to fit in the function.
    arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(old_assign));
    result = contract_cluster(K, active_clus, old_assign, psi, xi, 
                              a_theta, b_theta);
  }
  
  return result;
}

/*
Rcpp::List expand_cluster(arma::vec old_clus, arma::vec psi, arma::vec xi, 
                          double a_theta, double b_theta){
  Rcpp::List result;
  
  // The number of the existed cluster
  arma::vec n_clus_vec = arma::unique(old_clus);
  int n_clus = n_clus_vec.size();
  int new_clus = n_clus + 1;
  
  // Convert the cluster weight to the alpha.
  arma::vec alpha_vec = -1 * arma::ones(new_clus);
  double sum_existed_xi = sum(xi.head_rows(n_clus));
  double sum_existed_ci = R::rgamma(sum_existed_xi, 1.0);
  alpha_vec.rows(0, (n_clus - 1)) = psi * sum_existed_ci;
  
  // sample an alpha for the new cluster
  double new_clus_alpha = R::rgamma(xi.at(n_clus), 1.0);
  alpha_vec.row(new_clus - 1) = new_clus_alpha;
  
  // Calculate the acceptance probability
  arma::vec accept_prob = -1 * arma::ones(psi.size());
  accept_prob = (new_clus_alpha/alpha_vec) * 
    (sum(alpha_vec)/(sum(alpha_vec) + new_clus_alpha)) * (a_theta/b_theta);
  for(int j = 0; j < accept_prob.size(); j++){
    arma::vec a_prob = {accept_prob.at(j), 1};
    accept_prob.at(j) = arma::min(a_prob);
  }
  
  // Assign a new cluster
  arma::vec new_clus_assign = -1 * arma::ones(old_clus.size());
  for(int i = 0; i < old_clus.size(); i++){
    double u = arma::randu();
    if(u <= accept_prob.at(old_clus.at(i) - 1)){
      new_clus_assign.at(i) = new_clus;
    } else{
      new_clus_assign.at(i) = old_clus.at(i);
    }
  }
  
  // Reset the cluster index
  arma::uvec clus_elem;
  int clus_index = 1;
  arma::vec elem_count = -1 * arma::ones(new_clus);
  for(int k = 1; k <= new_clus; k++){
    clus_elem = arma::find(new_clus_assign == k);
    elem_count.at(k-1) = clus_elem.size();
    if(clus_elem.size() != 0){
      new_clus_assign.replace(k, clus_index);
      clus_index = clus_index + 1;
    } else {
      alpha_vec.shed_row(k - 1);
    }
  }
  
  result["elem_count"] = elem_count;
  result["new_alpha"] = alpha_vec;
  result["new_assign"] = new_clus_assign;
  result["new_clus_weight"] = new_clus_alpha;
  result["accept_prob"] = accept_prob;
  return result;
}
*/

/*
 *   // Determine which one is larger.
 int clus_small = d_candidate_clus[0]; //pre-define the smaller and larger
 int clus_large = d_candidate_clus[1];
 arma::vec freq_assign = -1 * arma::ones(2);
 for(int i = 0; i <= 1; i++){
 arma::uvec clus_elem = arma::find(old_assign == d_candidate_clus[i]);
 freq_assign.at(i) = clus_elem.size();
 if(i == 1 && (freq_assign[1] < freq_assign[0])){ // adjusted if incorrect
 clus_small = d_candidate_clus[1];
 clus_large = d_candidate_clus[0];
 }
 }
*/

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













































