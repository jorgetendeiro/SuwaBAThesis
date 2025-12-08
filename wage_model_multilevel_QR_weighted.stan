
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N, K] X;
  vector[N] y;
  array[N] int<lower=1, upper=J> industry;
  vector[N] w_raw;              // frequency weights
}

transformed data {
  matrix[N, K] Q_ast;
  matrix[K, K] R_ast;
  matrix[K, K] R_ast_inverse;

  Q_ast = qr_thin_Q(X) * sqrt(N - 1);
  R_ast = qr_thin_R(X) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
}

parameters {
  real alpha;
  vector[K] theta;
  vector[J] z_industry;
  real<lower=0> sigma0;         // sigma at min(mu) = sigma0
  real<lower=0> gamma;          // sigma at max(mu) = sigma0 + gamma
  real<lower=0> sigma_industry;
}

transformed parameters {
  vector[J] alpha_industry;
  vector[N] mu;
  vector[N] sigma;

  alpha_industry = sigma_industry * z_industry;
  mu = alpha + Q_ast * theta + alpha_industry[industry];
  
  real mu_min = min(mu);
  real mu_max = max(mu);
  for (n in 1:N) {
    sigma[n] = sigma0 + gamma * (mu[n] - mu_min) / (mu_max - mu_min);
  }
}

model {
  alpha  ~ normal(28.8, 10);
  theta  ~ normal(0, 5);
  sigma0 ~ normal(0, 20); 
  gamma  ~ normal(0, 20);  
  sigma_industry ~ normal(0, 3);
  z_industry ~ normal(0, 1);

  for (n in 1:N) {
    target += 1 * normal_lpdf(y[n] | mu[n], sigma[n]);
  }
}

generated quantities {
  vector[K] beta;
  vector[N] y_rep;

  beta = R_ast_inverse * theta;

  for (n in 1:N) {
    y_rep[n] = normal_rng(mu[n], sigma[n]);
  }
}

