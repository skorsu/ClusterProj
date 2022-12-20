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

// Step 3: Split-Merge: --------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List split_merge(int K, arma::vec old_assign, arma::vec alpha,
                       arma::vec xi, arma::mat y, arma::vec gamma_hyper, 
                       double a_theta, double b_theta, int T_iter = 10){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (alpha), hyperparameter for cluster (xi),
   *        data matrix (y), hyperparameter for the data (gamma),
   *        hyperparameter (a_theta, b_theta), 
   *        number of iteration for launch step (T_iter; default = 10).
   * Output: new cluster weight, updated cluster assignment.
   */ 
  
  // Vector for a new assignment
  arma::vec new_assign = old_assign;
  
  // Create the set of active and inactive cluster
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec active_clus = List_clusters["active"];
  arma::uvec inactive_clus = List_clusters["inactive"];
  
  // Prevent: The case when the all clusters are active.
  if(inactive_clus.size() == 0){
    inactive_clus = active_clus;
  }
  
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
    int K_new = 
      sample_clus(arma::ones(inactive_clus.size())/inactive_clus.size(), 
                  inactive_clus);
    new_assign.at(samp_obs[0]) = K_new;
    // Adjust the active and inactive vectors
    List_clusters = active_inactive(K, new_assign);
  }
  
  // Randomly assigned the observation in S to be either ci or cj
  Rcpp::NumericVector sample_cluster = Rcpp::NumericVector(2);
  sample_cluster[0] = new_assign.at(samp_obs[0]);
  sample_cluster[1] = new_assign.at(samp_obs[1]);
  arma::vec launch_init = Rcpp::sample(sample_cluster, S_index.size(), true);
  arma::uvec active_launch(2);
  active_launch[0] = new_assign.at(samp_obs[0]);
  active_launch[1] = new_assign.at(samp_obs[1]);
  
  for(int s = 0; s < S_index.size(); s++){
    int current_obs = S_index.at(s);
    new_assign.at(current_obs) = launch_init.at(s);
  }

  // Perform a launch step
  for(int t = 1; t <= T_iter; t++){
    for(int s = 0; s < S_index.size(); s++){
      int current_obs = S_index.at(s);
      arma::vec unnorm_prob = allocate_prob(current_obs, new_assign, xi, 
                                            y, gamma_hyper, active_launch);
      new_assign.at(current_obs) = 
        sample_clus(arma::normalise(unnorm_prob, 1), active_launch);
    }
  }
  
  arma::uvec launch_active = List_clusters["active"];
  arma::uvec launch_inactive = List_clusters["inactive"];
  
  // Prevent: The case when the all clusters are active.
  if(launch_inactive.size() == 0){
    launch_inactive = launch_active;
  }
  
  int sp_indicator = 0; // Split = 1; Merge = 0
  
  // Begin Split-Merge
  if(new_assign.at(samp_obs[0]) != new_assign.at(samp_obs[1])){
    // Merge two clusters into one (= cj)
    for(int s = 0; s < S_index.size(); s++){
      int current_obs = S_index.at(s);
      new_assign.at(current_obs) = new_assign.at(samp_obs[1]);
    }
  } else {
    sp_indicator = 1;
    // Sample K_new_s from inactive cluster and put ci back to that cluster.
    int K_new_s = 
      sample_clus(arma::ones(launch_inactive.size())/launch_inactive.size(), 
                  launch_inactive);
    new_assign.at(samp_obs[0]) = K_new_s;
    
    // Adjust the active and inactive vectors
    arma::uvec active_split(2);
    active_split[0] = new_assign.at(samp_obs[0]);
    active_split[1] = new_assign.at(samp_obs[1]);
    
    // Run another cluster allocation
    for(int s = 0; s < S_index.size(); s++){
      int current_obs = S_index.at(s);
      arma::vec unnorm_prob = allocate_prob(current_obs, new_assign, xi, 
                                            y, gamma_hyper, active_split);
      new_assign.at(current_obs) = 
        sample_clus(arma::normalise(unnorm_prob, 1), active_split);
    }
  }
  
  // Adjust the alpha vector
  List_clusters = active_inactive(K, new_assign);
  arma::vec final_active = List_clusters["active"];
  arma::vec final_inactive = List_clusters["inactive"];
  arma::vec new_alpha = adjust_alpha(K, new_assign, alpha);
  
  for(int i = 0; i < final_active.size(); i++){
    int current_clus = final_active.at(i);
    if(new_alpha.at(current_clus - 1) == 0){
      new_alpha.at(current_clus - 1) = R::rgamma(xi.at(current_clus - 1), 1);
    }
  }

  // MH update (log form)
  // Consider alpha
  double accept_prob = 0.0;
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
  
  // Consider the multinomial distribution
  double old_multi = 1.0;
  double new_multi = 1.0;
  for(int s = 0; s < S_index.size(); s++){
    int current_obs = S_index.at(s);
    old_multi = old_multi + log(alpha.at(old_assign.at(current_obs) - 1));
    new_multi = new_multi + log(new_alpha.at(new_assign.at(current_obs) - 1));
  }
  
  double proposal = log(2) * (S_index.size() - 2);
  
  if(sp_indicator == 0){
    // Merge
    accept_prob = accept_prob + proposal;
  } else {
    // Split
    accept_prob = accept_prob - proposal;
  }
  
  arma::vec A_vec = arma::zeros(2);
  A_vec[0] = accept_prob + new_multi - old_multi;
  double log_u = log(R::runif(0, 1));
  if(log_u > A_vec.min()){
    new_alpha = alpha;
    new_assign = old_assign;
  };

  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// Final Function: -------------------------------------------------------------


// Testing Area: ---------------------------------------------------------------









































