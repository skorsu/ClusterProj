#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// Note to self: ---------------------------------------------------------------
// - Not reset the cluster index. (Maybe lower computational cost.)
// - K is fixed. (We should include K as one of our input if necessary.)
// - We should get the index of the existed cluster first in most step.
// - Instead of interested in psi, we will use alpha vector instead.
// - alpha vector is K dimension.
// - gamma_hyper is a matrix with K by J dimension.

// Tasks: ----------------------------------------------------------------------
// * Step 4: Update other parameters. (?)

// Questions: ------------------------------------------------------------------
// - For Step 1, if all clusters are already active, 
//   can we randomly select one of the active cluster?
// - For Step 3, split-merge process when ci = cj and 
//   all clusters are already active.

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

// [[Rcpp::export]]
arma::mat rdirichlet_cpp(int num_samples, arma::vec alpha_m){
  int distribution_size = alpha_m.n_elem;
  // each row will be a draw from a Dirichlet
  arma::mat distribution = arma::zeros(num_samples, distribution_size);
  
  /* Description: Sample from dirichlet distribution.
   * Credit: https://www.mjdenny.com/blog.html
   */
  
  for (int i = 0; i < num_samples; ++i){
    double sum_term = 0;
    // loop through the distribution and draw Gamma variables
    for (int j = 0; j < distribution_size; ++j){
      double cur = R::rgamma(alpha_m[j],1.0);
      distribution(i,j) = cur;
      sum_term += cur;
    }
    // now normalize
    for (int j = 0; j < distribution_size; ++j) {
      distribution(i,j) = distribution(i,j)/sum_term;
    }
  }
  return(distribution);
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

// Step 3: Split-Merge: --------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List split_merge(int K, arma::vec old_assign, arma::vec alpha,
                       arma::vec xi, arma::mat y, arma::mat gamma_hyper, 
                       double a_theta, double b_theta, int sm_iter = 10){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (alpha), hyperparameter for cluster (xi),
   *        data matrix (y), hyperparameter for the data (gamma),
   *        hyperparameter (a_theta, b_theta), 
   *        number of iteration for launch step (sm_iter; default = 10).
   * Output: new cluster weight, updated cluster assignment.
   */ 
  
  // Vector for a new assignment
  arma::vec new_assign = old_assign;
  arma::vec dum_alpha = alpha;
  
  // Create the set of active and inactive cluster
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec active_clus = List_clusters["active"];
  arma::uvec inactive_clus = List_clusters["inactive"];
  
  // Sample two observations from the data.
  Rcpp::IntegerVector obs_index = Rcpp::seq(0, old_assign.size() - 1);
  Rcpp::IntegerVector samp_obs = Rcpp::sample(obs_index, 2);
  
  int obs_i = samp_obs[0];
  int obs_j = samp_obs[1];
  int c_i = old_assign.at(obs_i);
  int c_j = old_assign.at(obs_j);
  
  if(active_clus.size() == K){
    while(c_i == c_j){
      samp_obs = Rcpp::sample(obs_index, 2);
      obs_i = samp_obs[0];
      obs_j = samp_obs[1];
      c_i = old_assign.at(obs_i);
      c_j = old_assign.at(obs_j);
    }
  }
  
  // Subset only sample that have the same cluster as ci and cj
  arma::uvec S_index = arma::find(old_assign == c_i or old_assign == c_j);
  
  // Launch Step
  if(c_i == c_j){ // Split (when ci = cj, ci = K_new and cj = cj)
    // Select K_new
    arma::vec clus_prob = arma::ones(inactive_clus.size())/inactive_clus.size();
    c_i = sample_clus(clus_prob, inactive_clus);
    new_assign.at(obs_i) = c_i;
    dum_alpha.at(c_i - 1) = R::rgamma(xi.at(c_i - 1), 1);
  } 
  
  Rcpp::List List_launch = active_inactive(5, new_assign.elem(S_index));
  arma::uvec active_launch = List_launch["active"];
  
  // Randomly assigned the observation in S to be either ci or cj
  Rcpp::NumericVector sample_cluster = Rcpp::NumericVector(2);
  sample_cluster[0] = c_i;
  sample_cluster[1] = c_j;
  arma::vec launch_init = Rcpp::sample(sample_cluster, S_index.size(), true);
  
  for(int s = 0; s < S_index.size(); s++){
    int current_obs = S_index.at(s);
    new_assign.at(current_obs) = launch_init.at(s);
  }
  
  // Reassign in the Launch step
  for(int t = 0; t < sm_iter; t++){ // Loop in Launch
    for(int i = 0; i < S_index.size(); i++){ // Loop for each elements
      int s = S_index[i];
      arma::vec unnorm_prob = allocate_prob(s, new_assign, xi, y,
                                            gamma_hyper, active_launch);
      arma::vec norm_prob = arma::normalise(unnorm_prob, 1);
      new_assign.at(s) = sample_clus(norm_prob, active_launch);
    }
  }
  
  // Update the list of active and inactive cluster.
  Rcpp::List List_sm = active_inactive(5, new_assign);
  arma::uvec active_sm = List_sm["active"];
  arma::uvec inactive_sm = List_sm["inactive"];
  
  c_i = new_assign.at(obs_i);
  c_j = new_assign.at(obs_j);
  
  if(active_sm.size() == K){
    while(c_i == c_j){
      samp_obs = Rcpp::sample(obs_index, 2);
      obs_i = samp_obs[0];
      obs_j = samp_obs[1];
      c_i = new_assign.at(obs_i);
      c_j = new_assign.at(obs_j);
    }
  }
  
  // Subset only sample that have the same cluster as ci and cj
  arma::uvec SM_index = arma::find(new_assign == c_i or new_assign == c_j);
  
  Rcpp::List sm_launch = active_inactive(5, new_assign.elem(SM_index));
  arma::uvec active_sm_launch = sm_launch["active"];
  
  int split_merge;
  
  // Split-Merge Step
  if(c_i == c_j){ // Split
    split_merge = 1;
    // Select K_new
    arma::vec clus_prob = arma::ones(inactive_sm.size())/inactive_sm.size();
    c_i = sample_clus(clus_prob, inactive_sm);
    new_assign.at(obs_i) = c_i;
    dum_alpha.at(c_i - 1) = R::rgamma(xi.at(c_i - 1), 1);
    
    // Reassign the element in SM_index
    for(int i = 0; i < SM_index.size(); i++){ // Loop for each elements
      int s = SM_index[i];
      arma::vec unnorm_prob = allocate_prob(s, new_assign, xi, y,
                                            gamma_hyper, active_sm_launch);
      arma::vec norm_prob = arma::normalise(unnorm_prob, 1);
      new_assign.at(s) = sample_clus(norm_prob, active_sm_launch);
    }
    
  } else { // Merge (when ci != cj, ci = cj, which is set to cj)
    split_merge = 0;
    new_assign.elem(SM_index).fill(c_j);
  }
  
  // Adjust an alpha vector
  arma::vec new_alpha = adjust_alpha(K, new_assign, dum_alpha);
  
  // MH update (log form)
  double accept_prob = 0.0;
  double log_proposed = (SM_index.size() - 2) * log(0.5);
  
  // Consider alpha
  for(int k = 0; k < alpha.size(); k++){
    if(alpha.at(k) < new_alpha.at(k)){ 
      // Proposed a non-zero alpha (active cluster from inactive)
      accept_prob = accept_prob + 
        R::dgamma(new_alpha.at(k), xi.at(k), 1.0, 1) + 
        log(a_theta) - log(b_theta);
    } else if(alpha.at(k) > new_alpha.at(k)){
      // Proposed a zero alpha (inactive cluster from active)
      accept_prob = accept_prob - 
        R::dgamma(alpha.at(k), xi.at(k), 1.0, 1) +
        log(b_theta) - log(a_theta);
    }
  }
  
  // Consider the proposal distribution
  if(split_merge == 0){ // Merge
    accept_prob = accept_prob - log_proposed;
  } else { // Split
    accept_prob = accept_prob + log_proposed;
  }
  
  // Consider the multinomial distribution
  double old_multi = 0.0;
  double new_multi = 0.0;
  for(int i = 0; i < old_assign.size(); i++){
    old_multi = old_multi + log(alpha.at(old_assign.at(i) - 1));
    new_multi = new_multi + log(new_alpha.at(new_assign.at(i) - 1));
  }
  
  accept_prob = accept_prob + new_multi - old_multi;
  double log_A = std::min(0.0, accept_prob);
  double log_U = log(arma::randu());
  if(log_U > log_A){
    new_alpha = alpha;
    new_assign = old_assign;
  }
  
  result["new_alpha"] = new_alpha;
  result["new_assign"] = new_assign;
  
  return result;
}

// Step 4: Update alpha: -------------------------------------------------------
// [[Rcpp::export]]
arma::vec update_alpha(int K, arma::vec alpha, arma::vec xi, 
                       arma::vec old_assign){

  arma::vec new_alpha = alpha;
  
  /* Input: maximum cluster (K),previous cluster weight (alpha), 
   *        hyperparameter for cluster (xi), 
   *        previous cluster assignment (old_assign).
   * Output: new cluster weight.
   */
  
  Rcpp::List List_active = active_inactive(K, old_assign);
  arma::uvec active_clus = List_active["active"];
  
  arma::vec n_xi_elem = -1.0 * arma::ones(active_clus.size());
  
  for(int k = 0; k < active_clus.size(); k++){
    int clus_current = active_clus.at(k);
    arma::uvec obs_current_index = old_assign == clus_current;
    n_xi_elem.at(k) = sum(obs_current_index) + xi.at(clus_current - 1);
  }
  
  arma::mat psi_new = rdirichlet_cpp(1, n_xi_elem);
  
  for(int k = 0; k < active_clus.size(); k++){
    int clus_current = active_clus.at(k);
    new_alpha.at(clus_current - 1) = sum(alpha) * psi_new(0, k);
  }
  
  return new_alpha;
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
    
    // Step 3: Split-Merge
    Rcpp::List result_s3 = split_merge(K, reallocate_assign, reallocate_alpha,
                                       xi, y, gamma_hyper, a_theta, b_theta, 
                                       sm_iter);
    arma::vec sm_assign = result_s3["new_assign"];
    arma::vec sm_alpha = result_s3["new_alpha"];
    
    // Step 4: Update alpha
    arma::vec alpha_final = update_alpha(K, sm_alpha, xi, sm_assign);
    
    // Record the result
    clus_assign.col(i+1) = sm_assign;
    alpha_update.col(i+1) = alpha_final;
  }
  
  result["alpha_update"] = alpha_update;
  result["clus_assign"] = clus_assign;
  
  return result;
}










































