# ===== 0. 準備 =====
setwd("C:/Users/atomu1107/Downloads/")  # 作業ディレクトリ
library(dplyr)
library(readr)
library(cmdstanr)
library(posterior)
library(bayesplot)

# ===== 1. データ読み込み =====
data <- read_csv("Naoto_Suwa_data_v8_age_style_fixed.csv")

# 欠損削除
data <- na.omit(data)

# ===== 2. コーディング =====
data <- data %>%
  mutate(
    Gender_num = ifelse(Gender == "Male", 0, 1),
    Edu_num = as.numeric(factor(Education,
                                levels = c("JuniorHigh", "HighSchool", "VocationalSchool",
                                           "TechnicalSchoolJuniorCollege", "University"))),
    Age_num = as.numeric(factor(AgeGroup,
                                levels = c("0-19","20-24","25-29","30-34","35-39","40-44",
                                           "45-49","50-54","55-59","60-64","65-69","70+"))),
    Industry_num = as.numeric(factor(Industry))
  )

# ===== 3. 標準化（E-BFMI改善のため） =====
data <- data %>%
  mutate(
    Wage_scaled = as.numeric(scale(Wage_Yen)),  # 平均0, 分散1
    Age_scaled = as.numeric(scale(Age_num))
  )

# ===== 4. Stan に渡すデータ作成 =====
X <- model.matrix(~ 0 + Gender_num + Edu_num + Age_scaled, data = data)

X_centered <- scale(X, center = TRUE, scale = FALSE)
Q <- qr.Q(qr(X_centered))
R <- qr.R(qr(X_centered))
R_inv <- solve(R)

stan_data <- list(
  N = nrow(data),
  K = ncol(X_centered),
  J = length(unique(data$Industry_num)),
  Q = Q,
  R_inv = R_inv,
  y = data$Wage_scaled,
  industry = data$Industry_num
)

# ===== 5. Stan モデルコード（QR再パラメータ化 + 標準化対応） =====
stan_code <- "
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
"

# ===== 6. Stan モデルの保存とコンパイル =====
writeLines(stan_code, "wage_model_multilevel_QR.stan")  
mod <- cmdstan_model("wage_model_multilevel_QR.stan")   

# ===== 7. サンプリング =====
fit <- mod$sample(
  data = stan_data,
  iter_warmup = 1000,
  iter_sampling = 2000,
  chains = 4,
  parallel_chains = 4,
  seed = 123,
  adapt_delta = 0.98,
  max_treedepth = 15
)

# ===== 8. 結果確認 =====
fit$summary(c("alpha", "beta", "sigma", "sigma_industry"))

# E-BFMI診断
fit$cmdstan_diagnose()

# 事後分布可視化
posterior <- as_draws_df(fit)
mcmc_hist(posterior, pars = c("beta[1]", "beta[2]", "beta[3]"))
