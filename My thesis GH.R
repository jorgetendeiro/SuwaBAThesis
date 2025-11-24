# ===== 0. Setup =====
setwd("C:/Users/atomu1107/Downloads/")
library(dplyr)
library(readr)
library(cmdstanr)
library(posterior)
library(bayesplot)

# ===== 1. Load data =====
data <- read_csv("Suwa_CSWS.csv")


data <- na.omit(data)

# Wage to man-yen
data <- data %>%
  mutate(
    Wage_man = Wage_Yen / 10000
  )


# ===== 2. Encoding =====
data <- data %>%
  mutate(
    Gender = ifelse(Gender == "Male", 0, 1),
    Edu = as.numeric(factor(
      Education,
      levels = c("JuniorHigh","HighSchool","VocationalSchool",
                 "TechnicalSchoolJuniorCollege","University")
    )),
    Age = as.numeric(factor(
      AgeGroup,
      levels = c("0-19","20-24","25-29","30-34","35-39","40-44",
                 "45-49","50-54","55-59","60-64","65-69","70+")
    )),
    Industry = as.numeric(factor(Industry))
  )

# ===== 3. Design matrix =====
X <- model.matrix(~ 0 + Gender + Edu + Age, data = data)

# ===== 4. Stan data =====
stan_data <- list(
  N = nrow(data),
  K = ncol(X),
  J = length(unique(data$Industry)),
  X = X,
  y = data$Wage_man,
  industry = data$Industry,
  w_raw = as.vector(data$Employees)   
)

# ===== 5. Write Stan model =====
stan_code <- "
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
"

writeLines(stan_code, "wage_model_multilevel_QR_weighted.stan")

# ===== 6. Sampling =====
mod <- cmdstan_model("wage_model_multilevel_QR_weighted.stan")

fit <- mod$sample(
  data = stan_data,
  iter_warmup = 1000,
  iter_sampling = 2000,
  chains = 4,
  parallel_chains = 4,
  seed = 123
)

# ===== 7. Results =====
fit$summary(c("alpha", "beta", "sigma", "sigma_industry"))
fit$cmdstan_diagnose()

posterior <- as_draws_df(fit)
mcmc_hist(posterior, pars = c("beta[1]", "beta[2]", "beta[3]"))
