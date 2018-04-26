// Based on code from Michael Betancourt
// Defines a function for sampling from the predictive distribution, for visualization.
// gp_pred_rng(x1_predict, x2_predict, y, x1, x2, alpha, rho1, rho2, sigma, 1e-10);
functions {
  vector gp_pred_rng(real[] x1_predict, real[] x2_predict,
                     vector y, real[] x1,real[] x2,
                     real alpha, real rho1, real rho2, real sigma, real delta) {
    int N1 = rows(y);
    int N2 = size(x1_predict);
    vector[N2] f_predict;
    {
      matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho1) .* cov_exp_quad(x2, 1, rho2)
                         + diag_matrix(rep_vector(square(sigma), N1));
      matrix[N1, N1] L_K = cholesky_decompose(K);

      vector[N1] L_K_div_y = mdivide_left_tri_low(L_K, y);
      vector[N1] K_div_y = mdivide_right_tri_low(L_K_div_y', L_K)';
      matrix[N1, N2] k_x_x_predict = cov_exp_quad(x1, x1_predict, alpha, rho1) .* cov_exp_quad(x2, x2_predict, 1, rho2) ;
      vector[N2] f_predict_mu = (k_x_x_predict' * K_div_y);
      matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x_x_predict);
      matrix[N2, N2] cov_f_predict =   cov_exp_quad(x1_predict, alpha, rho1).*cov_exp_quad(x2_predict, 1, rho2) - v_pred' * v_pred
                              + diag_matrix(rep_vector(delta, N2));
      f_predict = multi_normal_rng(f_predict_mu, cov_f_predict);
    }
    return f_predict;
  }
}

data {
  //Data
  int<lower=1> N;
  real x1[N];
  real x2[N];
  vector[N] y;

  //Locations for predictions
  int<lower=1> N_predict;
  real x1_predict[N_predict];
  real x2_predict[N_predict];

}

parameters {
  real<lower=0> rho1;
  real<lower=0> rho2;
  real<lower=0> alpha;
  real<lower=0> sigma;
}

model {
  // sets up squared exponential kernel, plus gaussian noise.
  matrix[N, N] cov =   cov_exp_quad(x1, alpha, rho1) .* cov_exp_quad(x2, 1, rho2)
                     + diag_matrix(rep_vector(square(sigma), N));
  matrix[N, N] L_cov = cholesky_decompose(cov);

  // priors on hyperparameters -- feel free to change!
  //rho ~ inv_gamma(1,1);
  //alpha ~ inv_gamma(1,1);
  //sigma ~ inv_gamma(1,1);
  //likelihood
  y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
}

generated quantities {
  // generates a random function
  vector[N_predict] f_predict = gp_pred_rng(x1_predict, x2_predict, y, x1, x2, alpha, rho1, rho2, sigma, 1e-10);

  //adds on noise
  vector[N_predict] y_predict;
  for (n in 1:N_predict)
    y_predict[n] = normal_rng(f_predict[n], sigma);
}

