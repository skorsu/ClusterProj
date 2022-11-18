#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// Note to self: ---------------------------------------------------------------
// - Not reset the cluster index. (Maybe lower computational cost.)
// - K is fixed. (We should include K as one of our input if necessary.)
// - We should get the index of the existed cluster first in most step.
// - Instead of interested in psi, we will use alpha vector instead.
// - alpha vector is K dimension.

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

arma::vec allocate_prob(int i, arma::vec current_assign, arma::vec xi, 
                        arma::mat y, arma::vec gamma_hyper){
  
  /* Description: Calculate the unnormalized probability for each cluster 
   *              for observation i.
   * Input: current index (i), current cluster assignment, 
   *        hyperparameter for cluster (xi), data matrix (y), 
   *        hyperparameter for the data (gamma)
   * Output: unnormalized allocation probability.
   */
  
  // Get the active clusters
  arma::uvec active_clus = 
    arma::conv_to<arma::uvec>::from(arma::unique(current_assign));
  int K_pos = active_clus.size(); 
  double n_not_i = current_assign.size() - 1;
  double sum_hyper_clus = sum(xi.elem(active_clus - 1));
  arma::vec unnorm_prob = -1 * arma::ones(K_pos);
  
  // Split the data into two sets: Observation #i and Excluding Observation #i
  arma::vec obs_i = arma::conv_to<arma::vec>::from((y.row(i)));
  arma::mat obs_not_i = y; 
  obs_not_i.shed_row(i);
  arma::vec clus_not_i = current_assign; 
  clus_not_i.shed_row(i);

  Rcpp::NumericVector x = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(obs_i));
    
  // Calculate the unnormailzed allocation probability for each cluster
  for(int j = 0; j < K_pos; j++){
    arma::uvec clus_index = find(clus_not_i == active_clus[j]);
    
    arma::vec sum_col = 
      arma::conv_to<arma::vec>::from(sum(obs_not_i.rows(clus_index), 0));
    Rcpp::NumericVector col_gam = 
      Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(sum_col + gamma_hyper));
    arma::vec col_gam_gamma = gamma(col_gam);
    Rcpp::NumericVector sum_col_gam = {sum(col_gam)};
    arma::vec first_t = {gamma(sum_col_gam)/arma::prod(col_gam_gamma)};
      
    Rcpp::NumericVector i_col_gam = x + col_gam;
    arma::vec i_col_gam_gamma = gamma(i_col_gam);
    Rcpp::NumericVector sum_i_col_gam = {sum(i_col_gam)};
    arma::vec second_t = {arma::prod(i_col_gam_gamma)/gamma(sum_i_col_gam)};
      
    double third_t = (clus_index.size() + xi[active_clus[j] - 1])/
      (n_not_i + sum_hyper_clus);
      
      unnorm_prob[j] = as_scalar(first_t * second_t * third_t);
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
  
  // Select the cluster that we will expand
  Rcpp::IntegerVector d_new_clus = Rcpp::sample(inactive_clus, 1);
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
  Rcpp::IntegerVector active_clus = active_List["active"];
  
  // Assign a new assignment
  for(int a = 0; a < new_assign.size(); a++){
    // Calculate the unnormalized probability
    arma::vec unnorm_prob = allocate_prob(a, new_assign, xi, y, gamma_hyper);
    
    // Reassign the observation a
    Rcpp::NumericVector normalized_prob = Rcpp::as<Rcpp::NumericVector>
      (Rcpp::wrap(arma::normalise(unnorm_prob, 1)));
    new_assign.at(a) = Rcpp::sample(active_clus, 1, false, normalized_prob)[0];
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
Rcpp::List cluster_func(int K, arma::vec old_assign, arma::vec psi,
                        arma::vec xi, arma::mat y, arma::vec gamma_hyper, 
                        double a_theta, double b_theta, int iter = 1000){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (psi), hyperparameter for cluster (xi),
   *        data matrix (y), hyperparameter for the data (gamma),
   *        hyperparameter (a_theta, b_theta), iteration (default at 1000).
   * Output: new cluster weight, updated cluster assignment.
   */ 
  
  // Storing the final result
  arma::vec new_assign = old_assign;
  arma::vec new_psi = psi;
  
  // First Step: Expand-Contract
  Rcpp::List result_s1 = expand_step(K, new_assign, new_psi, 
                                     xi, a_theta, b_theta);
  // new_assign = result_s1["new_assign"];
  // result_s1["new_psi"] = new_psi;
  
  return result;
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

// [[Rcpp::export]]
arma::vec test_List(arma::vec test_1){
  Rcpp::List test_result = active_inactive(12, test_1);
  arma::vec test_active = test_result["active"];
  return test_active;
}
  
  











































