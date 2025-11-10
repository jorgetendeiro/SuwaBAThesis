
data {
  int<lower=0> N;
  int<lower=0> K;
  int<lower=0> J;
  matrix[N, K] Q;
  matrix[K, K] R_inv;
  vector[N] y;
  array[N] int<lower=1, upper=J> industry;
}
parameters {
  vector[K] theta;
  real alpha;
  vector[J] alpha_industry;
  real<lower=0> sigma;
  real<lower=0> sigma_industry;
}
model {
  vector[N] mu;
  mu = alpha + Q * theta + alpha_industry[industry];

  alpha ~ normal(0, 1);
  theta ~ normal(0, 1);
  alpha_industry ~ normal(0, sigma_industry);
  sigma ~ exponential(1);
  sigma_industry ~ exponential(1);

  y ~ normal(mu, sigma);
}
generated quantities {
  vector[K] beta;
  beta = R_inv * theta;
}

