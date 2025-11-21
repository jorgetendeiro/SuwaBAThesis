
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N, K] X;
  vector[N] y;
  array[N] int<lower=1, upper=J> industry;
  vector[N] w_raw;
}

transformed data {
  matrix[N, K] Q_ast;
  matrix[K, K] R_ast;
  matrix[K, K] R_ast_inverse;
  vector<lower=0>[N] w_sqrt;

  Q_ast = qr_thin_Q(X) * sqrt(N - 1);
  R_ast = qr_thin_R(X) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);

  for (n in 1:N) {
    if (w_raw[n] > 0) w_sqrt[n] = sqrt(w_raw[n]); 
    else w_sqrt[n] = 0;
  }
}

parameters {
  real alpha;
  vector[K] theta;
  vector[J] z_industry;
  real<lower=0> sigma;
  real<lower=0> sigma_industry;
}

transformed parameters {
  vector[J] alpha_industry;
  vector[N] mu;

  alpha_industry = sigma_industry * z_industry;
  mu = alpha + Q_ast * theta + alpha_industry[industry];
}

model {
  alpha ~ normal(28.8, 10);
  theta ~ normal(0, 5);
  sigma ~ normal(0, 200); 
  sigma_industry ~ normal(0, 3);
  z_industry ~ normal(0, 1);

  for (n in 1:N) {
    if (w_sqrt[n] > 0) {
      real sigma_n = sigma / w_sqrt[n];
      target += normal_lpdf(y[n] | mu[n], sigma_n);
    }
  }
}

generated quantities {
  vector[K] beta;
  vector[N] y_rep;

  beta = R_ast_inverse * theta;

  for (n in 1:N) {
    if (w_sqrt[n] > 0) {
      real sigma_n = sigma / w_sqrt[n];
      y_rep[n] = normal_rng(mu[n], sigma_n);
    } else {
      y_rep[n] = negative_infinity();
    }
  }
}

