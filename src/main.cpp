#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// Note to self: ---------------------------------------------------------------
// - Not reset the cluster index. (Maybe lower computational cost.)
// - K is fixed. (We should include K as one of our input if necessary.)
// - We should get the index of the existed cluster first in most step.
// - Instead of interested in psi, we will use alpha vector instead.
// - alpha vector is K dimension.

// Tasks: ----------------------------------------------------------------------
// * Step 2: Update a code to be suitable with cluster gamma vectors.
// * Follows Matt's Email: sampling until merge for step 3
// * Step 4: Update other parameters. (?)

// Questions: ------------------------------------------------------------------
// - For Step 1, if all clusters are already active, 
//   can we randomly select one of the active cluster?

// User-defined function: ------------------------------------------------------
// [[Rcpp::export]]
int sample_clus(arma::vec norm_probs, arma::uvec active_clus){
  
  /* Description: To run a multinomial distribution and get the reallocated 
   *              cluster.
   * Input: Normalized probability (norm_prob), active cluster
   * Output: New assigned cluster
   */
  
  int k = active_clus.size();
  arma::imat C = arma::imat(k, 1);
  rmultinom(1, norm_probs.begin(), k, C.colptr(0));
  
  arma::mat mat_index = arma::conv_to<arma::mat>::from(arma::index_max(C));
  int new_clus = active_clus.at(mat_index(0, 0));
  
  return new_clus;
}

// [[Rcpp::export]]
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
double density_gamma(arma::rowvec y, arma::rowvec hyper_gamma_k){
  double result;
  
  /* Description: This is the function for calculating p(yi|gamma_k)
   * Input: Data point (y) and hyperparameter for that cluster (hyper_gamma_k)
   * Output: p(yi|gamma_k)
   */
  
  // Convert arma object to Rcpp object
  Rcpp::NumericVector yi = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(y));
  Rcpp::NumericVector gamma_k = 
    Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(hyper_gamma_k));
  
  // Calculate the vector for yi + gamma_k
  Rcpp::NumericVector yi_gamma_k = yi + gamma_k;
  
  arma::vec gamma_gamma_k = gamma(gamma_k);
  Rcpp::NumericVector sum_gamma_k = {sum(gamma_k)};
  arma::vec gamma_sum_gamma_k = gamma(sum_gamma_k);
  
  arma::vec gamma_yi_gamma_k = gamma(yi_gamma_k);
  Rcpp::NumericVector sum_yi_gamma_k = {sum(yi_gamma_k)};
  arma::vec gamma_sum_yi_gamma_k = gamma(sum_yi_gamma_k);
  
  return (gamma_sum_gamma_k/arma::prod(gamma_gamma_k))[0] *
    (arma::prod(gamma_yi_gamma_k)/gamma_sum_yi_gamma_k)[0];
}

// [[Rcpp::export]]
arma::vec allocate_prob(int i, arma::vec current_assign, arma::vec xi, 
                        arma::mat y, arma::mat gamma_hyper_mat, 
                        arma::uvec active_clus){
  
  arma::vec unnorm_prob = -1 * arma::ones(active_clus.size());
    
  /* Description: Calculate the unnormalized probability for each cluster 
   *              for observation i.
   * Input: current index (i), current cluster assignment, 
   *        hyperparameter for cluster (xi), data matrix (y), 
   *        hyperparameter for the data (gamma_hyper_mat), active cluster.
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
    
    // Select the hyperparameter that corresponding to the cluster k
    arma::vec gamma_hyper = arma::conv_to<arma::vec>::
      from(gamma_hyper_mat.row(current_c - 1));
    
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

// Step 1: Update the cluster space: -------------------------------------------
// [[Rcpp::export]]
Rcpp::List expand_step(int K, arma::vec old_assign, arma::vec alpha,
                       arma::vec xi, arma::mat y, arma::mat gamma_hyper,
                       double a_theta, double b_theta){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (alpha), hyperparameter for cluster (xi),
   *        data point (yi), hyperparameter for each cluster (gamma_hyper),
   *        hyperparameter (a_theta, b_theta).
   * Output: new cluster weight, updated cluster assignment.
   */ 
  
  // Indicate the existed clusters and inactive clusters
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec inactive_clus = List_clusters["inactive"];
  arma::uvec active_clus = List_clusters["active"];
  
  // If all clusters are already active, we randomly select the cluster from 
  // those which already active.
  if(inactive_clus.size() == 0){
    inactive_clus = active_clus;
  }
  
  // Select a candidate cluster
  arma::vec samp_prob = arma::ones(inactive_clus.size())/inactive_clus.size();
  int candidate_clus = sample_clus(samp_prob, inactive_clus);
  
  // Sample alpha for new active cluster
  alpha.at(candidate_clus - 1) = R::rgamma(xi.at(candidate_clus - 1), 1);
  
  // Calculate the acceptance probability and assign a new cluster
  arma::vec accept_prob = (alpha.at(candidate_clus - 1)/alpha) * 
    ((sum(alpha) - alpha.at(candidate_clus - 1))/sum(alpha)) * 
    (a_theta/b_theta);
  
  arma::vec new_assign = old_assign;
  for(int i = 0; i < old_assign.size(); i++){
    double prob = accept_prob.at(old_assign.at(i) - 1) *
      density_gamma(y.row(i), gamma_hyper.row(candidate_clus - 1)) /
        density_gamma(y.row(i), gamma_hyper.row(old_assign.at(i) - 1));
    double A = std::min(prob, 1.0);
    double U = arma::randu();
    if(U <= A){
      new_assign.at(i) = candidate_clus;
    }
  }
  
  // Adjust an alpha vector
  arma::vec new_alpha = adjust_alpha(K, new_assign, alpha);
  
  result["new_alpha"] = new_alpha;
  result["new_assign"] = new_assign;

  return result;
}

// Step 2: Allocate the observation to the existing clusters: ------------------
// [[Rcpp::export]]
Rcpp::List cluster_assign(int K, arma::vec old_assign, arma::vec xi, 
                          arma::mat y, arma::mat gamma_hyper, arma::vec alpha){
  
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
  for(int i = 0; i < new_assign.size(); i++){
    // Calculate the unnormalized probability
    arma::vec unnorm_prob = allocate_prob(i, new_assign, xi, 
                                          y, gamma_hyper, active_clus);
    // Calculate the normalized probability
    arma::vec norm_prob = arma::normalise(unnorm_prob, 1);
    
    // Reassign a new cluster
    new_assign.at(i) = sample_clus(norm_prob, active_clus);
  }
  
  // Adjust an alpha vector
  arma::vec new_alpha = adjust_alpha(K, new_assign, alpha);
  
  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// Final Function: -------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List cluster_func(int K, arma::vec old_assign, arma::vec alpha,
                        arma::vec xi, arma::mat y, arma::mat gamma_hyper, 
                        double a_theta, double b_theta, int sm_iter = 10, 
                        int all_iter = 100){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (alpha), hyperparameter for cluster (xi),
   *        data matrix (y), hyperparameter for the data (gamma),
   *        hyperparameter (a_theta, b_theta), 
   *        iteration for the split-merge process (sm_iter; default = 10)
   *        overall iteration (all_iter; default at 100).
   * Output: new cluster weight, updated cluster assignment, 
   *         number of active cluster in each iteration.
   */ 
  
  // Storing the active cluster for each iteration
  arma::vec n_active = -1 * arma::ones(all_iter + 1);
  arma::vec dum_unique = arma::unique(old_assign);
  n_active.row(0) = dum_unique.size();
  
  // Storing the cluster assignment for each iteration
  arma::mat clus_assign = -1 * arma::ones(old_assign.size(), all_iter + 1);
  clus_assign.col(0) = old_assign;
  
  // Storing alpha vector
  arma::mat alpha_update = -1 * arma::ones(alpha.size(), all_iter + 1);
  alpha_update.col(0) = alpha;
  
  for(int i = 0; i < all_iter; i++){
    // Initial value
    arma::vec current_assign = clus_assign.col(i);
    arma::vec current_alpha = alpha_update.col(i);
    
    // Step 1: Expand Step
    Rcpp::List result_s1 = expand_step(K, current_assign, current_alpha, 
                                       xi, y, gamma_hyper, a_theta, b_theta);
    arma::vec expand_assign = result_s1["new_assign"];
    arma::vec expand_alpha = result_s1["new_alpha"];
    
    // Step 2: Reallocation
    Rcpp::List result_s2 = cluster_assign(K, expand_assign, xi, y, 
                                          gamma_hyper, expand_alpha);
    arma::vec reallocate_assign = result_s2["new_assign"];
    arma::vec reallocate_alpha = result_s2["new_alpha"];
    
    // Record the result
    arma::vec n_unique_expand = arma::unique(reallocate_assign);
    clus_assign.col(i+1) = reallocate_assign;
    alpha_update.col(i+1) = reallocate_alpha;
    n_active.row(i+1) = n_unique_expand.size();
  }
  
  result["n_active"] = n_active;
  result["alpha_update"] = alpha_update;
  result["clus_assign"] = clus_assign;
  
  return result;
}










































