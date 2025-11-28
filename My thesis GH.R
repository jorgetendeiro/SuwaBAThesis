# ===== 0. Setup =====
setwd("C:/Users/atomu1107/Downloads/")
library(dplyr)
library(readr)
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)



# ===== 1. Load data =====
data <- read_csv("Suwa_BSWS.csv")

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



# ===== 3. Drop rows with weights zero =====
#JT# I dropped rows with 0 employees, as these do not contribute to the likelihood.
#JT# This simplifies the Stan model below.
data <- data[data$Employees > 0, ]



# ===== 4. Design matrix =====
X <- model.matrix(~ 0 + Gender + Edu + Age, data = data)



# ===== 5. Stan data =====
#JT# About the weights. Some weights are huge in magnitude:
#JT# range(data$Employees) # up to 355710
#JT# That creates problems for Stan to fit the model.
#JT# Also, I realized that modeling the y means as y~N(mu, sigma/sqrt(n)) is a bad idea, 
#JT# because we do not realistically expect that the mean wages shrink with n (in any 
#JT# company, there will always be lower and higher wages regardless of the number of workers).
#JT# I changed the model in three ways:
#JT# - Now, we model the y means directly, so y ~ N(mu, sigma). Observations are mean wages.
#JT# - I model heteroscedasticity. That is, sigma changes with X. I do so because, after fitting
#JT#   the model using a constant sigma, the residuals plot showed a funnel effect. To account 
#JT#   for this, I model sigma as a linear function of the linear predictor mu.
#JT# - First, I included the weights in the model as frequency weights, rescaled to reduce their size.
#JT#   It works, but fit was not great. I then tried to remove the weights (so, all weights = to 1). And 
#JT#   this fit the data much better. So, in the Stan data below, I make all weights equal to 1 (no 
#JT#   weighing).
#JT# The above three changes improved model fit tremendously.
stan_data <- list(
  N        = nrow(data),
  K        = ncol(X),
  J        = length(unique(data$Industry)),
  X        = X,
  y        = data$Wage_man,
  industry = data$Industry,
  w_raw    = rep(1, nrow(data)) # as.vector(data$Employees / mean(data$Employees)) 
)



# ===== 6. Write Stan model =====
stan_code <- "
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
    target += w_raw[n] * normal_lpdf(y[n] | mu[n], sigma[n]);
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
"

writeLines(stan_code, "wage_model_multilevel_QR_weighted.stan")



# ===== 7. Sampling =====
mod <- cmdstan_model("wage_model_multilevel_QR_weighted.stan")

fit <- mod$sample(
  data            = stan_data,
  iter_warmup     = 1000,
  iter_sampling   = 2000,
  chains          = 4,
  parallel_chains = 4,
  seed            = 123
)



# ===== 8. Results =====
fit$summary(c("alpha", "beta", "sigma0", "gamma", "sigma_industry"))
fit$cmdstan_diagnose()

posterior <- as_draws_df(fit)
mcmc_hist(posterior, pars = c("beta[1]", "beta[2]", "beta[3]"))



# ===== 9. Posterior Predictive Check =====
yrep_cols <- grep("^y_rep\\[", names(posterior), value = TRUE)
y_rep <- as.matrix(posterior[, yrep_cols])

# Observed data
y_obs <- stan_data$y

ppc_hist(y_obs, y_rep[1:50, ]) +
  ggtitle("Posterior Predictive Check: Histogram")

ppc_dens_overlay(y_obs, y_rep[1:50, ]) +
  ggtitle("Posterior Predictive Check: Density Overlay")

ppc_stat(y_obs, y_rep, stat = "mean")
ppc_stat(y_obs, y_rep, stat = "sd")

ppc_boxplot(y_obs, y_rep[1:50, ]) +
  ggtitle("Posterior Predictive Check: Boxplot")

ppc_intervals_grouped(y_obs, y_rep, group = data$Industry)
ppc_ecdf_overlay(y_obs, y_rep)



# ===== 10. Residuals =====
# Residuals (standardized, to account for the heteroscedasticity):
mu_draws      <- posterior::as_draws_matrix(fit$draws("mu"))
mu_hat        <- colMeans(mu_draws)
resid         <- y_obs - mu_hat
sigma_n_draws <- posterior::as_draws_matrix(fit$draws("sigma"))
sigma_n_hat   <- colMeans(sigma_n_draws)
resid_std     <- resid / sigma_n_hat

# Funnel effect, residuals increase with the linear predictor (heteroscedasticity):
plot(mu_hat, resid)           # residual vs fitted
# But the standardized residuals are fine:
plot(mu_hat, resid_std)       # residual vs fitted
hist(resid_std, breaks = 20, freq = FALSE)  # ~ N(0,1)
curve(dnorm(x), add=TRUE)
# See standardized residuals per predictor:
plot(stan_data$X[, 1], resid_std) # vs Gender
plot(stan_data$X[, 2], resid_std) # vs Edu
plot(stan_data$X[, 3], resid_std) # vs Age (a bit or a curvilinear relation here, not so great)

