#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// Note to self: ---------------------------------------------------------------
// - Not reset the cluster index. (Maybe lower computational cost.)
// - K is fixed. (We should include K as one of our input if necessary.)
// - We should get the index of the existed cluster first in most step.
// - Instead of interested in psi, we will use alpha vector instead.
// - alpha vector is K dimension.
// - For the active cluster that will be an input for `allocate_prob`, 

// Questions: ------------------------------------------------------------------
// - Multinomial(1, p) = sample(x, 1 , p)?

// User-defined function: ------------------------------------------------------
Rcpp::List active_inactive(int K, arma::vec clus_assign){
  Rcpp::List result;
  
  /* Description: This function will return the list consisted of two vectors
   *              (1) active clusters and (2) inactive cluster from the cluster
   *              assignment vector.
   * Input: maximum cluster (K), cluster assignment vector (clus_assign)
   * Output: A list of two vectors (active & inactive clusters.) 
   */
  
  Rcpp::IntegerVector all_possible = Rcpp::seq(1, K);
  Rcpp::IntegerVector active_clus = 
    Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(arma::unique(clus_assign)));
  Rcpp::IntegerVector inactive_clus = Rcpp::setdiff(all_possible, active_clus);
  
  result["active"] = active_clus;
  result["inactive"] = inactive_clus;

  return result;
}

// [[Rcpp::export]]
arma::vec allocate_prob(int i, arma::vec current_assign, arma::vec xi, 
                        arma::mat y, arma::vec gamma_hyper, 
                        arma::uvec active_clus){
  
  arma::vec unnorm_prob = -1 * arma::ones(active_clus.size());
    
  /* Description: Calculate the unnormalized probability for each cluster 
   *              for observation i.
   * Input: current index (i), current cluster assignment, 
   *        hyperparameter for cluster (xi), data matrix (y), 
   *        hyperparameter for the data (gamma), active cluster.
   * Output: unnormalized allocation probability.
   */
  
  // Split the data into two sets: (1) observation i (2) without observation i.
  arma::vec y_i = arma::conv_to<arma::vec>::from((y.row(i)));
  arma::mat y_not_i = y; 
  y_not_i.shed_row(i);
  arma::vec clus_not_i = current_assign; 
  clus_not_i.shed_row(i);
  
  // Calculate the unnormalized allocation probability for each active cluster
  for(int k = 0; k < active_clus.size(); k++){
    int current_c = active_clus[k];
    arma::uvec current_ci = arma::find(clus_not_i == current_c);
    
    // Filter only the observation from cluster i
    arma::mat y_current = y_not_i.rows(current_ci);
    
    // First term: not consider the observation i
    // Sum for each column and gamma_hyper
    arma::rowvec sum_y_gamma_column = gamma_hyper.t() + sum(y_current, 0); 
    Rcpp::NumericVector y_gamma = 
      Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(sum_y_gamma_column));
    Rcpp::NumericVector sum_y_gamma = {sum(y_gamma)};
    arma::vec gfx_sum_y_gamma = gamma(sum_y_gamma);
    arma::vec gfx_y_gamma = gamma(y_gamma);
    
    // Second Term: consider observation i
    arma::rowvec i_sum_y_gamma_column = y_i.t() + sum_y_gamma_column; 
    Rcpp::NumericVector i_y_gamma = 
      Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(i_sum_y_gamma_column));
    Rcpp::NumericVector i_sum_y_gamma = {sum(i_y_gamma)};
    arma::vec gfx_i_sum_y_gamma = gamma(i_sum_y_gamma);
    arma::vec gfx_i_y_gamma = gamma(i_y_gamma);
    
    // Calculate the allocation probability and store it
    arma::vec alloc_prob = (gfx_sum_y_gamma/arma::prod(gfx_y_gamma)) *
      (arma::prod(gfx_i_y_gamma)/gfx_i_sum_y_gamma) *
      (current_ci.size() + xi[(current_c - 1)]);
    unnorm_prob[k] = as_scalar(alloc_prob);
  }

  return unnorm_prob;
}

arma::vec adjust_alpha(int K, arma::vec clus_assign, arma::vec alpha_vec){
  arma::vec a_alpha = arma::zeros(K);
  
  /* Description: To adjust the alpha vector. Keep only the element with at 
   *              least 1 observation is allocated to.
   * Input: maximum cluster (K), cluster assignment, cluster weight (alpha_vec)
   * Output: adjusted alpha vector
   */
  
  arma::vec new_active_clus = arma::unique(clus_assign);
  arma::uvec index_active = arma::conv_to<arma::uvec>::from(new_active_clus) - 1;
  a_alpha.elem(index_active) = alpha_vec.elem(index_active);
  
  return a_alpha;
}

Rcpp::List expand_function(int K, Rcpp::IntegerVector inactive_clus, 
                           arma::uvec active_clus, arma::vec old_assign, 
                           arma::vec alpha, arma::vec xi, double a_theta, 
                           double b_theta){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), inactive clusters, active clusters, 
   *        previous cluster assignment, previous cluster weight (alpha), 
   *        hyperparameter for cluster (xi), hyperparameter (a_theta, b_theta)
   * Output: new cluster weight, updated cluster assignment.
   */
  
  // Prevent: The case when the all clusters are active.
  Rcpp::IntegerVector adjusted_inactive;
  if(inactive_clus.length() == 0){
    adjusted_inactive = active_clus;
  } else {
    adjusted_inactive = inactive_clus;
  }
  
  // Select the cluster that we will expand
  Rcpp::IntegerVector d_new_clus = Rcpp::sample(adjusted_inactive, 1);
  int new_clus = d_new_clus[0];
  
  // Sample alpha for new active cluster
  alpha.at(new_clus - 1) = R::rgamma(xi.at(new_clus - 1), 1);
  // So, the alpha now will have both previous alpha and the new alpha
  
  // Calculate the acceptance probability
  arma::vec accept_prob = (alpha.at(new_clus - 1)/alpha) * 
    ((sum(alpha) - alpha.at(new_clus - 1))/sum(alpha)) * (a_theta/b_theta);
  arma::vec A = arma::min(accept_prob, arma::ones(alpha.size()));
  
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
  
  // Adjust the alpha vector
  arma::vec new_alpha = adjust_alpha(K, new_assign, alpha);

  // Return the result
  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// Multinomial Distribution
// [[Rcpp::export]]
int sample_clus(arma::vec norm_probs, arma::uvec active_clus){

  /* Description: To run a mulyinomial distribution and get the reallocated 
   *              cluster.
   * Input: Normalized probability (norm_prob), active cluster
   * Output: New assigned cluster
   */
  
  int k = active_clus.size();
  arma::imat C = arma::imat(k, 1);
  rmultinom(1, norm_probs.begin(), k, C.colptr(0));
  
  int index = as_scalar(arma::index_max(C));
  int new_clus = active_clus.at(index);
  
  return new_clus;
}

// Step 1: Update the cluster space: -------------------------------------------
// [[Rcpp::export]]
Rcpp::List expand_step(int K, arma::vec old_assign, arma::vec alpha,
                       arma::vec xi, double a_theta, double b_theta){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (alpha), hyperparameter for cluster (xi),
   *        hyperparameter (a_theta, b_theta).
   * Output: new cluster weight, updated cluster assignment.
   */ 
  
  // Indicate the existed clusters and inactive clusters
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  Rcpp::IntegerVector inactive_clus = List_clusters["inactive"];
  
  // Expand (and/or Contract) Cluster Space
  arma::uvec active_clus = 
    arma::conv_to<arma::uvec>::from(arma::unique(old_assign));
  result = expand_function(K, inactive_clus, active_clus, old_assign, alpha, 
                           xi, a_theta, b_theta);
  
  return result;
}

// Step 2: Allocate the observation to the existing clusters: ------------------
// [[Rcpp::export]]
Rcpp::List cluster_assign(int K, arma::vec old_assign, arma::vec xi, 
                          arma::mat y, arma::vec gamma_hyper, arma::vec alpha){
  
  Rcpp::List result;
  arma::vec new_assign = old_assign;
  
  // Assume that all clusters share the same hyperparameters (gamma)
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        hyperparameter for cluster (xi), data matrix (y), 
   *        hyperparameter for the data (gamma)
   * Output: updated cluster assignment.
   */
  
  // Create the vector of the active cluster
  Rcpp::List active_List = active_inactive(K, old_assign);
  arma::uvec active_clus = active_List["active"];

  // Assign a new assignment
  for(int a = 0; a < new_assign.size(); a++){
    // Calculate the unnormalized probability
    arma::vec unnorm_prob = allocate_prob(a, new_assign, xi, 
                                          y, gamma_hyper, active_clus);
    // Calculate the normalized probability
    arma::vec norm_prob = arma::normalise(unnorm_prob, 1);
    
    // Reassign a new cluster
    new_assign.at(a) = sample_clus(norm_prob, active_clus);
  }
  
  // Adjust an alpha vector
  arma::vec new_alpha = adjust_alpha(K, new_assign, alpha);

  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// Step 3: Split-Merge ---------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List split_merge(int K, arma::vec old_assign, arma::vec psi,
                       arma::vec xi, arma::mat y, arma::vec gamma_hyper, 
                       double a_theta, double b_theta){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (psi), hyperparameter for cluster (xi),
   *        data matrix (y), hyperparameter for the data (gamma),
   *        hyperparameter (a_theta, b_theta).
   * Output: new cluster weight, updated cluster assignment.
   */ 
  
  // Vector for a new assignment
  arma::vec new_assign = old_assign;
  
  // Create the set of active and inactive cluster
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec active_clus = List_clusters["active"];
  arma::uvec inactive_clus = List_clusters["inactive"];
  
  // Create an alpha vector
  double alpha_p = R::rgamma(sum(xi.elem(arma::find(psi != 0))), 1);
  arma::vec alpha_vec = alpha_p * psi;
  
  // Sample two observation from the previous assignment.
  Rcpp::IntegerVector obs_index = Rcpp::seq(0, old_assign.size() - 1);
  Rcpp::IntegerVector samp_obs = Rcpp::sample(obs_index, 2); // i and j
  
  // Subset only sample that have the same cluster as ci and cj
  arma::uvec S_index = 
    arma::find(old_assign == old_assign.at(samp_obs[0]) or 
                 old_assign == old_assign.at(samp_obs[1]));
  
  // Launch Step
  if(old_assign.at(samp_obs[0]) == old_assign.at(samp_obs[1])){
    // when ci = cj, ci = K_new and cj = cj.
    // Sample K_new from inactive cluster and put ci back to that cluster.
    Rcpp::IntegerVector d_new_clus = 
      Rcpp::sample(Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(inactive_clus)), 1);
    int K_new = d_new_clus[0];
    new_assign.at(samp_obs[0]) = K_new;
    // Adjust the active and inactive vectors
    active_clus.insert_rows(active_clus.size(), 1);
    active_clus[active_clus.size() - 1] = K_new;
    arma::uvec K_new_index = arma::find(inactive_clus == K_new);
    inactive_clus.shed_row(K_new_index[0]);
  }
  
  // Randomly assigned the observation in S to be either ci or cj
  
  result["active_clus"] = active_clus;
  result["inactive_clus"] = inactive_clus;
  result["samp_obs"] = samp_obs;
  
  return result;
}

// Final Function: -------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List cluster_func(int K, arma::vec old_assign, arma::vec alpha,
                        arma::vec xi, arma::mat y, arma::vec gamma_hyper, 
                        double a_theta, double b_theta, int iter = 100){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (alpha), hyperparameter for cluster (xi),
   *        data matrix (y), hyperparameter for the data (gamma),
   *        hyperparameter (a_theta, b_theta), iteration (default at 100).
   * Output: new cluster weight, updated cluster assignment, 
   *         number of active cluster in each iteration.
   */ 
  
  // Storing the active cluster for each iteration
  arma::vec n_active = -1 * arma::ones(iter + 1);
  arma::vec dum_unique = arma::unique(old_assign);
  n_active.row(0) = dum_unique.size();
  
  // Storing the cluster assignment for each iteration
  arma::mat clus_assign = -1 * arma::ones(old_assign.size(), iter + 1);
  clus_assign.col(0) = old_assign;
  
  // Storing alpha vector
  arma::mat alpha_update = -1 * arma::ones(alpha.size(), iter + 1);
  alpha_update.col(0) = alpha;
  
  for(int i = 0; i < iter; i++){
    // Initial value
    arma::vec current_assign = clus_assign.col(i);
    arma::vec current_alpha = alpha_update.col(i);
    
    // Step 1: Expand Step
    Rcpp::List result_s1 = expand_step(K, current_assign, current_alpha, 
                                       xi, a_theta, b_theta);
    arma::vec expand_assign = result_s1["new_assign"];
    arma::vec expand_alpha = result_s1["new_alpha"];
    
    // Step 2: Reallocation
    Rcpp::List result_s2 = cluster_assign(K, expand_assign, xi, y, 
                                          gamma_hyper, expand_alpha);
    arma::vec reallocate_assign = result_s2["new_assign"];
    arma::vec reallocate_alpha = result_s2["new_alpha"];

    // Record the result
    arma::vec n_unique_expand = arma::unique(expand_assign);
    clus_assign.col(i+1) = expand_assign;
    alpha_update.col(i+1) = expand_alpha;
    n_active.row(i+1) = n_unique_expand.size();
  }
  
  result["n_active"] = n_active;
  result["alpha_update"] = alpha_update;
  result["clus_assign"] = clus_assign;
  
  return result;
}

// Testing Area: ---------------------------------------------------------------












































