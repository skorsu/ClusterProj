#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// Note to self: ---------------------------------------------------------------
// - Not reset the cluster index. (Maybe lower computational cost.)
// - K is fixed. (We should include K as one of our input if necessary.)
// - We should get the index of the existed cluster first in most step.
// - Instead of interested in psi, we will use alpha vector instead.
// - alpha vector is K dimension.
// - gamma_hyper is a matrix with K by J dimension.
// - For the numerical stability, I will take logarithm function on the 
//   probability term, especially for the allocation probability.
// - For Step 1, if all clusters are already active, I will skip this step for 
//   that iteration..
// - For Step 3, split-merge process when ci = cj and all clusters are already 
//   active. I will always reject that proposed assignment.

// Tasks: ----------------------------------------------------------------------
// - Matt's Email (2022/12/31)
//   * The rows of gamma_init should be unique. Otherwise, you are implying 
//     that the probabilities across clusters are the same.
//   * Instead of related the row of gamma_init to K, link gamma_init to K_init 
//     instead.

// Questions: ------------------------------------------------------------------
// - For the gamma_init matrix, how can we determine that which rows are 
//   duplicated? Consider only maximum only or consider the proportion?

// User-defined function: ------------------------------------------------------
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
Rcpp::List check_unique_gamma(int K, arma::vec old_assign, 
                              arma::mat gamma_init){
  Rcpp::List result;
  
  /* Description: This function will check whether there are any duplicated row 
   *              or not. If there is a duplicate row, this function will merge 
   *              this two clusters.
   * Input: Maximum Potential Clusters (K), 
   *        Previous cluster assignment (old_assign), 
   *        Hyperparameter for each cluster (gamma_init)
   * Output: updated cluster assignment and hyperparameter.
   */
  
  // Create a new matrix with the maximum indicator for each row.
  arma::vec max_vec = arma::max(gamma_init, 1);
  arma::mat max_mat = arma::mat(arma::size(gamma_init));
  for(int r = 0; r < max_mat.n_rows; ++r){
    arma::vec row_r = arma::conv_to<arma::vec>::from(gamma_init.row(r));
    arma::uvec max_ind = arma::find(row_r == max_vec.at(r));
    arma::uvec not_max_ind = arma::find(row_r != max_vec.at(r));
    row_r.elem(max_ind).fill(1);
    row_r.elem(not_max_ind).fill(0);
    max_mat.row(r) = row_r.t();
  }
  
  // If all element in that row are all zeros, we collapse that cluster.
  arma::ucolvec bool_vec = arma::all(gamma_init == 0, 1);
  arma::uvec active_clus = arma::find(bool_vec == 0) + 1;
  arma::uvec inactive_clus = arma::find(bool_vec == 1) + 1;
  for(int i = 0; i < inactive_clus.size(); ++i){
    int inac_clus = inactive_clus.at(i);
    max_mat.row(inac_clus - 1) = arma::zeros(1, max_mat.n_cols);
    arma::uvec inac_elem = arma::find(old_assign == inac_clus);
    arma::vec samp_index = arma::conv_to<arma::vec>::
      from(arma::randi(inac_elem.size(), 
                       arma::distr_param(1, active_clus.size())) - 1);
    for(int j = 0; j < samp_index.size(); ++j){
      old_assign.row(inac_elem.at(j)).fill(active_clus.at(samp_index.at(j)));
    }
  }
  
  // Merge the duplicated row of gamma
  arma::uvec duplicate_row;
  for(int i = 0; i < max_mat.n_rows - 1; ++i){
    for(int j = (i + 1); j < max_mat.n_rows; ++j){
      if(arma::all(max_mat.row(i) == max_mat.row(j))){
        // If the row are same, we merge the cluster j to cluster i.
        int clus_j = j + 1;
        arma::uvec clus_j_elem = arma::find(old_assign == clus_j);
        old_assign.rows(clus_j_elem).fill(i + 1);
        // Record that row
        int last_rec = duplicate_row.size();
        arma::uvec row_j(1);
        row_j.row(0).fill(j);
        duplicate_row.insert_rows(last_rec, row_j);
      }
    }
  }
  
  max_mat.rows(duplicate_row).fill(0.0);
  
  // Add row of gamma_init and norm_max_mat
  if(max_mat.n_rows < K){
    arma::mat inactive_zero = arma::zeros(K - max_mat.n_rows, max_mat.n_cols);
    max_mat.insert_rows(max_mat.n_rows, inactive_zero);
  }
  
  // For the rows with all zeros, we will randomly simulate gamma.
  for(int r = 0; r < max_mat.n_rows; ++r){
    if(all(max_mat.row(r) == 0.0)){
      bool new_row_dup = true;
      
      while(new_row_dup){
        // Proposed a new row of gamma_init
        arma::vec proposed_gamma = arma::zeros(max_mat.n_cols);
        while(all(proposed_gamma == 0.0)){
          proposed_gamma.randu();
          arma::uvec set_one = arma::find(proposed_gamma > 0.5);
          arma::uvec set_zero = arma::find(proposed_gamma <= 0.5);
          proposed_gamma.elem(set_one).fill(1.0);
          proposed_gamma.elem(set_zero).fill(0.0);
        }
        
        int n_distinct = 0;
        for(int k = 0; k < max_mat.n_rows; ++k){
          if(r != k){
            arma::vec one_indicator_k = 
              arma::conv_to<arma::vec>::from(max_mat.row(k));
            one_indicator_k.elem(arma::find(one_indicator_k != 1.0)).fill(0.0);
            if(all(one_indicator_k == proposed_gamma)){
              n_distinct += 1;
            }
          }
        }
        
        if(n_distinct == 0){
          new_row_dup = false;
          max_mat.row(r) = proposed_gamma.t();
        }
        
      }
    }
  }
  
  max_mat *= 0.5;
  max_mat.elem(arma::find(max_mat == 0.0)).fill(0.1);
  
  result["adjusted_gamma"] = max_mat;
  result["adjusted_old_assign"] = old_assign;
  
  return result;
}

// [[Rcpp::export]]
arma::vec xi_adjust(arma::vec previous_xi){
  
  arma::uvec elem_zero = arma::find(previous_xi == 0);
  arma::uvec elem_non_zero = arma::find(previous_xi != 0);
  
  double xi_sub = mean(previous_xi.elem(elem_non_zero)); 
  previous_xi.elem(elem_zero).fill(xi_sub);
  
  return previous_xi;
}

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
double log_density_gamma(arma::rowvec y, arma::rowvec hyper_gamma_k){
  double log_p = 0.0;
  
  /* Description: This is the function for calculating the log of the 
   *              p(yi|gamma_k)
   * Input: Data point (y) and hyperparameter for that cluster (hyper_gamma_k)
   * Output: log(p(yi|gamma_k))
   */
  
  arma::rowvec y_hyper = y + hyper_gamma_k;
  arma::mat lg_hyper = arma::lgamma(hyper_gamma_k);
  arma::mat lg_y_hyper = arma::lgamma(y_hyper);
  
  log_p += lgamma(sum(hyper_gamma_k)) + arma::accu(lg_y_hyper) -
    arma::accu(lg_hyper) - lgamma(sum(y_hyper));
  
  return log_p;
}

// [[Rcpp::export]]
arma::vec log_allocate_prob(int i, arma::vec current_assign, arma::vec xi, 
                            arma::mat y, arma::mat gamma_hyper_mat, 
                            arma::uvec active_clus){
  
  arma::vec log_unnorm_prob = -1 * arma::ones(active_clus.size());
  
  /* Description: Calculate the log unnormalized probability for each cluster 
   *              for observation i.
   * Input: current index (i), current cluster assignment, 
   *        hyperparameter for cluster (xi), data matrix (y), 
   *        hyperparameter for the data (gamma_hyper_mat), active cluster.
   * Output: log unnormalized allocation probability.
   */
  
  // Split the data into two sets: (1) observation i (2) without observation i.
  arma::rowvec y_i = y.row(i);
  arma::mat y_not_i = y; 
  y_not_i.shed_row(i);
  arma::vec clus_not_i = current_assign; 
  clus_not_i.shed_row(i);
  
  // Calculate the unnormalized allocation probability for each active cluster
  for(int k = 0; k < active_clus.size(); ++k){
    int current_c = active_clus[k];
    arma::uvec current_ci = arma::find(clus_not_i == current_c);
    
    // Select the hyperparameter that corresponding to the cluster k
    arma::rowvec gamma_hyper = arma::conv_to<arma::rowvec>::
      from(gamma_hyper_mat.row(current_c - 1));
    double xi_k = xi.at(current_c - 1);
    
    // Filter only the observation from cluster i
    arma::mat y_current = y_not_i.rows(current_ci);
    
    // Calculate required vectors
    arma::rowvec y_hyper = gamma_hyper + arma::sum(y_current, 0);
    arma::rowvec y_hyper_yi = y_hyper + y_i;
    double n_xi = xi_k + current_ci.size();
    
    // Calculate log(gamma) component.
    arma::mat lg_y_hyper = arma::lgamma(y_hyper);
    arma::mat lg_y_hyper_yi = arma::lgamma(y_hyper_yi);
    
    double calculate_lg = lgamma(sum(y_hyper)) + arma::accu(lg_y_hyper_yi) +
      std::log(n_xi) - arma::accu(lg_y_hyper) - lgamma(sum(y_hyper_yi));
    
    log_unnorm_prob.row(k).fill(calculate_lg);
  }
  
  return log_unnorm_prob;
}

// [[Rcpp::export]]
arma::vec log_sum_exp(arma::vec log_unnorm_prob){
  
  /* Description: This function will calculate the normalized probability 
   *              by applying log-sum-exp trick.
   * Credit: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
   * Input: log of the unnormalized probability (log_unnorm_prob)
   * Output: normalized probability
   */
  
  double max_elem = log_unnorm_prob.max();
  double t = log(0.00000000000000000001) - log(log_unnorm_prob.size());          
  
  for(int k = 0; k < log_unnorm_prob.size(); ++k){
    double prob_k = log_unnorm_prob.at(k) - max_elem;
    if(prob_k > t){
      log_unnorm_prob.row(k).fill(std::exp(prob_k));
    } else {
      log_unnorm_prob.row(k).fill(0.0);
    }
  }
  
  // Normalize the vector
  return arma::normalise(log_unnorm_prob, 1);
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
  
  if(active_clus.size() < K){
    // Select a candidate cluster
    arma::vec samp_prob = arma::ones(inactive_clus.size())/inactive_clus.size();
    int candidate_clus = sample_clus(samp_prob, inactive_clus);
    
    // Sample alpha for new active cluster
    alpha.row(candidate_clus - 1).fill(R::rgamma(xi.at(candidate_clus - 1), 1));
    
    // Calculate the log of acceptance probability and assign a new cluster
    arma::vec log_accept_prob = std::log(alpha.at(candidate_clus - 1)) + 
      std::log((sum(alpha) - alpha.at(candidate_clus - 1))) + 
      std::log(a_theta) - arma::log(alpha) - std::log(sum(alpha)) -
      std::log(b_theta);
    
    arma::vec new_assign = old_assign;
    for(int i = 0; i < old_assign.size(); ++i){
      double log_prob = log_accept_prob.at(old_assign.at(i) - 1) +
        log_density_gamma(y.row(i), gamma_hyper.row(candidate_clus - 1)) -
        log_density_gamma(y.row(i), gamma_hyper.row(old_assign.at(i) - 1));
      double log_A = std::min(log_prob, 0.0);
      double log_U = std::log(arma::randu());
      if(log_U <= log_A){
        new_assign.row(i).fill(candidate_clus);
      }
    }
    
    // Adjust an alpha vector
    arma::vec new_alpha = adjust_alpha(K, new_assign, alpha);
    
    result["new_alpha"] = new_alpha;
    result["new_assign"] = new_assign;
  } else {
    result["new_alpha"] = alpha;
    result["new_assign"] = old_assign;
  }
  
  return result;
}

// Step 2: Allocate the observation to the existing clusters: ------------------
// [[Rcpp::export]]
Rcpp::List cluster_assign(int K, arma::vec old_assign, arma::vec xi, 
                          arma::mat y, arma::mat gamma_hyper, arma::vec alpha){
  
  Rcpp::List result;
  arma::vec new_assign = old_assign;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        hyperparameter for cluster (xi), data matrix (y), 
   *        hyperparameter for the data (gamma)
   * Output: updated cluster assignment.
   */
  
  // Create the vector of the active cluster
  Rcpp::List active_List = active_inactive(K, old_assign);
  arma::uvec active_clus = active_List["active"];
  
  // Assign a new assignment
  for(int i = 0; i < new_assign.size(); ++i){
    // Calculate the unnormalized probability
    arma::vec log_unnorm_prob = log_allocate_prob(i, new_assign, xi, 
                                                  y, gamma_hyper, active_clus);
    // Calculate the normalized probability
    arma::vec norm_prob = log_sum_exp(log_unnorm_prob);
    
    // Reassign a new cluster
    int new_clus = sample_clus(norm_prob, active_clus);
    new_assign.row(i).fill(new_clus);
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
                       double a_theta, double b_theta, int sm_iter){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (alpha), hyperparameter for cluster (xi),
   *        data matrix (y), hyperparameter for the data (gamma),
   *        hyperparameter (a_theta, b_theta), 
   *        number of iteration for launch step (sm_iter).
   * Output: new cluster weight, updated cluster assignment.
   */ 
  
  // Initial the alpha vector and assignment vector
  arma::vec launch_assign = old_assign;
  arma::vec launch_alpha = alpha;
  
  // Create the set of active and inactive cluster
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec active_clus = List_clusters["active"];
  arma::uvec inactive_clus = List_clusters["inactive"];
  
  // Sample two observations from the data.
  Rcpp::IntegerVector obs_index = Rcpp::seq(0, old_assign.size() - 1);
  Rcpp::IntegerVector samp_obs = Rcpp::sample(obs_index, 2);
  
  int obs_i = samp_obs[0];
  int obs_j = samp_obs[1];
  int c_i = old_assign.at(obs_i); // ci_launch
  int c_j = old_assign.at(obs_j); // cj_launch
  
  if(active_clus.size() == K){
    while(c_i == c_j){
      samp_obs = Rcpp::sample(obs_index, 2);
      obs_i = samp_obs[0];
      obs_j = samp_obs[1];
      c_i = old_assign.at(obs_i);
      c_j = old_assign.at(obs_j);
    }
  }
  
  // Select only the observations that in the same cluster as obs_i and obs_j
  arma::uvec s_index = find((old_assign == c_i) or (old_assign == c_j));
  
  if(c_i == c_j){
    arma::vec prob_inactive = arma::ones(inactive_clus.size())/
      inactive_clus.size();
    c_i = sample_clus(prob_inactive, inactive_clus);
    launch_assign.row(obs_i).fill(c_i);
    launch_alpha.row(c_i - 1).fill(R::rgamma(xi.at(c_i - 1), 1.0));
  }
  
  // Randomly assign the observation in s_index to either c_i or c_j
  arma::uvec cluster_launch(2);
  cluster_launch.row(0).fill(c_i);
  cluster_launch.row(1).fill(c_j);
  for(int i = 0; i < s_index.size(); ++i){
    int current_obs = s_index.at(i);
    arma::vec random_prob = 0.5 * arma::ones(2);
    if((current_obs != obs_i) and (current_obs != obs_j)){
      launch_assign.row(current_obs).
      fill(sample_clus(random_prob, cluster_launch));
    }
  }
  
  // Perform a Launch Step
  for(int t = 0; t < sm_iter; ++t){
    for(int i = 0; i < s_index.size(); ++i){
      int current_obs = s_index.at(i);
      arma::vec log_unnorm_prob = log_allocate_prob(current_obs, launch_assign, 
                                                    xi, y, gamma_hyper, 
                                                    cluster_launch);
      arma::vec norm_prob = log_sum_exp(log_unnorm_prob);
      launch_assign.row(current_obs).
      fill(sample_clus(norm_prob, cluster_launch));
    }
  }
  
  // Prepare for the split-merge process
  double sm_indicator = 0.0;
  arma::vec new_assign = launch_assign;
  arma::vec launch_alpha_vec = launch_alpha; 
  // We will use launch_assign and launch_alpha_vec for MH algorithm.
  List_clusters = active_inactive(K, launch_assign);
  arma::uvec active_sm = List_clusters["active"];
  arma::uvec inactive_sm = List_clusters["inactive"];
  
  c_i = launch_assign.at(obs_i);
  c_j = launch_assign.at(obs_j);
  arma::uvec cluster_sm(2);
  cluster_sm.row(0).fill(-1);
  cluster_sm.row(1).fill(c_j);
  
  // Split-Merge Process
  if(c_i != c_j){ 
    // merge these two clusters into c_j cluster
    // Rcpp::Rcout << "final: merge" << std::endl;
    sm_indicator = 1.0;
    new_assign.elem(s_index).fill(c_j);
  } else if((c_i == c_j) and (active_sm.size() != K)) { 
    // split in case that at least one cluster is inactive.
    // Rcpp::Rcout << "final: split (some inactive)" << std::endl;
    sm_indicator = -1.0;
    
    // sample a new inactive cluster
    arma::vec prob_inactive = arma::ones(inactive_sm.size())/inactive_sm.size();
    c_i = sample_clus(prob_inactive, inactive_sm);
    new_assign.row(obs_i).fill(c_i);
    launch_alpha.row(c_i - 1).fill(R::rgamma(xi.at(c_i - 1), 1.0));
    cluster_sm.row(0).fill(c_i);
    
    for(int i = 0; i < s_index.size(); ++i){
      int current_obs = s_index.at(i);
      if((current_obs != obs_i) and (current_obs != obs_j)){
        arma::vec log_unnorm_prob = log_allocate_prob(current_obs, new_assign, 
                                                      xi, y, gamma_hyper, 
                                                      cluster_sm);
        arma::vec norm_prob = log_sum_exp(log_unnorm_prob);
        // Rcpp::Rcout << norm_prob << std::endl;
        new_assign.row(current_obs).fill(sample_clus(norm_prob, cluster_sm));
      }
    }
  } else {
    // Rcpp::Rcout << "final: split (none inactive)" << std::endl;
  }
  
  arma::vec new_alpha = adjust_alpha(K, new_assign, launch_alpha);
  
  // MH Update (log form)
  // Elements
  double launch_elem = 0.0;
  double final_elem = 0.0;
  double alpha_log = 0.0;
  double proposal = sm_indicator * std::log(0.5) * s_index.size();
  
  for(int k = 1; k <= K; ++k){
    // Calculate alpha
    if(launch_alpha_vec.at(k - 1) != new_alpha.at(k - 1)){
      if(new_alpha.at(k - 1) != 0){
        alpha_log += R::dgamma(new_alpha.at(k - 1), xi.at(k - 1), 1.0, 1);
        alpha_log += std::log(a_theta);
        alpha_log -= std::log(b_theta);
      } else {
        alpha_log -= R::dgamma(launch_alpha_vec.at(k - 1), 
                               xi.at(k - 1), 1.0, 1);
        alpha_log -= std::log(a_theta);
        alpha_log += std::log(b_theta);
      }
    }
    // Calculate Multinomial
    arma::uvec launch_elem_vec = arma::find(launch_assign == k);
    arma::uvec final_elem_vec = arma::find(new_assign == k);
    if(launch_elem_vec.size() > 0){
      launch_elem += launch_elem_vec.size() * 
        std::log(launch_alpha_vec.at(k - 1));
    }
    if(final_elem_vec.size() > 0){
      final_elem += final_elem_vec.size() * std::log(new_alpha.at(k - 1));
    }
  }
  
  double log_A = std::min(std::log(1), 
                          alpha_log + final_elem - launch_elem + proposal);
  double log_u = std::log(R::runif(0.0, 1.0));
  
  if(log_u >= log_A){
    new_assign = launch_assign;
    new_alpha = launch_alpha_vec;
  }
  
  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  
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
  
  for(int k = 0; k < active_clus.size(); ++k){
    int clus_current = active_clus.at(k);
    arma::uvec obs_current_index = old_assign == clus_current;
    n_xi_elem.at(k) = sum(obs_current_index) + xi.at(clus_current - 1);
  }
  
  arma::mat psi_new = rdirichlet_cpp(1, n_xi_elem);
  
  for(int k = 0; k < active_clus.size(); ++k){
    int clus_current = active_clus.at(k);
    new_alpha.at(clus_current - 1) = sum(alpha) * psi_new(0, k);
  }
  
  return new_alpha;
}

// Final Function: -------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List cluster_func(int K, arma::vec old_assign, arma::vec alpha,
                        arma::vec xi, arma::mat y, arma::mat gamma_hyper, 
                        double a_theta, double b_theta, int sm_iter, 
                        int all_iter, bool print_iter, int iter_print){
  Rcpp::List result;
  
  /* Input: maximum cluster (K), previous cluster assignment, 
   *        previous cluster weight (alpha), hyperparameter for cluster (xi),
   *        data matrix (y), hyperparameter for the data (gamma),
   *        hyperparameter (a_theta, b_theta), 
   *        iteration for the split-merge process (sm_iter)
   *        overall iteration (all_iter), print progress (print_iter),
   *        if printed, preint every iter_print iteration.
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
  
  for(int i = 0; i < all_iter; ++i){
    
    if(print_iter == true){
      if((i+1) % iter_print == 0){
        Rcpp::Rcout << (i + 1) << std::endl;
      }
    }
    
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

// [[Rcpp::export]]
Rcpp::List clus_func_new(int iter, int K, int K_init, arma::mat data_mat, 
                         arma::mat gamma_hyper, arma::vec xi,
                         double a_theta, double b_theta, int sm_iter){
  Rcpp::List result;
  
  // The maximum value of K is 2^(columns of data) - 1
  if(K > (std::pow(2.0, data_mat.n_cols) - 1.0)){
    Rcpp::Rcout << "K cannot be exceed 2^(columns of data) - 1." << std::endl;
    Rcpp::Rcout << "Fixed: Let K equals to 2^(columns of data) - 1." << std::endl;
    int K_adjust = std::pow(2.0, data_mat.n_cols) - 1.0;
    K = K_adjust;
  }
  
  // K_init should less than or equal to K.
  if(K_init > K){
    Rcpp::Rcout << "K must greater than or equal to K_init." << std::endl;
    Rcpp::Rcout << "Fixed: Let K_init equals to K." << std::endl;
    K_init = K;
  }
  
  // Initial the value of ci
  arma::vec old_assign = arma::conv_to<arma::vec>::
    from(arma::randi(data_mat.n_rows, arma::distr_param(1, K_init)));
  
  // Check the gamma_hyper
  // Need to check: (1) Number of row matches K_init. (2) Duplicate
  if(gamma_hyper.n_rows < K_init){
    arma::mat appended_mat = arma::mat(K_init - gamma_hyper.n_rows, 
                                       gamma_hyper.n_cols, arma::fill::randu);
    gamma_hyper.insert_rows(gamma_hyper.n_rows, appended_mat);
  } else if(gamma_hyper.n_rows > K_init){
    gamma_hyper.shed_rows(K_init, gamma_hyper.n_rows - 1);
  }
  
  // Deals with duplicated row and all-zeros.
  Rcpp::List adjust_cluster = check_unique_gamma(K, old_assign, gamma_hyper);
  arma::mat adj_gamma_hyper = adjust_cluster["adjusted_gamma"];
  arma::vec adj_old_assign = adjust_cluster["adjusted_old_assign"];
  
  // Adjust xi vector and initialize the alpha vector
  arma::vec alpha_vec = arma::zeros(K);
  // For xi, need to (1) extend to K dimension.
  if(xi.size() > K_init){
    xi.shed_rows(K_init, xi.size() - 1);
  }
  arma::vec appended_xi = arma::zeros(K - xi.size());
  xi.insert_rows(xi.size(), appended_xi);
  // (2) Set a xi equals to 0 for inactive cluster.
  Rcpp::List clus_pre = active_inactive(K, adj_old_assign);
  arma::uvec pre_inactive = clus_pre["inactive"];
  xi.elem(pre_inactive - 1).fill(0);
  // (3) Initial the alpha for non-zero xi.
  for(int k = 0; k < K; ++k){
    if(xi.at(k) != 0){
      alpha_vec.row(k).fill(R::rgamma(xi.at(k), 1.0));
    }
  }
  
  arma::vec xi_adjusted = xi_adjust(xi);
  
  /*
  result["adj_gamma_hyper"] = adj_gamma_hyper;
  result["adj_old_assign"] = adj_old_assign;
  result["xi_adjusted"] = xi_adjusted;
  result["alpha"] = alpha_vec;
  */
  
  int i = 0;
  arma::mat iter_clus = arma::mat(data_mat.n_rows, iter);
  while(i < iter){
    
    // Step 1: Expand Step
    Rcpp::List result_s1 = expand_step(K, adj_old_assign, alpha_vec, 
                                       xi_adjusted, data_mat, adj_gamma_hyper, 
                                       a_theta, b_theta);
    arma::vec expand_assign = result_s1["new_assign"];
    arma::vec expand_alpha = result_s1["new_alpha"];
    
    // Step 2: Reallocation
    Rcpp::List result_s2 = cluster_assign(K, expand_assign, xi_adjusted, 
                                          data_mat, adj_gamma_hyper, 
                                          expand_alpha);
    arma::vec reallocate_assign = result_s2["new_assign"];
    arma::vec reallocate_alpha = result_s2["new_alpha"];
    
    // Step 3: Split-Merge
    Rcpp::List result_s3 = split_merge(K, reallocate_assign, reallocate_alpha,
                                       xi_adjusted, data_mat, adj_gamma_hyper, 
                                       a_theta, b_theta, sm_iter);
    arma::vec sm_assign = result_s3["new_assign"];
    arma::vec sm_alpha = result_s3["new_alpha"];
    
    // Step 4: Update alpha
    arma::vec alpha_final = update_alpha(K, sm_alpha, xi_adjusted, sm_assign);
    
    // Record the result
    iter_clus.col(i) = sm_assign;
    adj_old_assign = sm_assign;
    alpha_vec = alpha_final;
    i += 1;
  }
  
  result["cluster_assignment"] = iter_clus;
  
  return result;
}








































