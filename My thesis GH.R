# ===== 0. Setup =====
# setwd("C:/Users/atomu1107/Downloads/")
library(dplyr)
library(readr)
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(gridExtra)
library(grid)



# ===== 1. Load data =====
data <- read_csv("Suwa_BSWS.csv", show_col_types = FALSE)

data <- na.omit(data)

# Wage to man-yen
data <- data %>%
  mutate(
    Wage_man = Wage_Yen / 10000
  )


# ===== 3. Encoding =====
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
    Industry.labs = factor(Industry), 
    Industry = as.numeric(factor(Industry))
  )



# ===== 4. Drop rows with weights zero =====
# Dropping rows with 0 employees, as these do not contribute to the likelihood.
# This simplifies the Stan model below.
data <- data[data$Employees > 0, ]



# ===== 5. Design matrix =====
X <- model.matrix(~ 0 + Gender + Edu + Age, data = data)



# ===== 6. Write Stan model =====
# Notes:
# - I model the y means directly, so y ~ N(mu, sigma). Observations are mean wages.
# - I model heteroscedasticity. That is, sigma changes with X. I do so because, after fitting
#   the model using a constant sigma, the residuals plot showed a funnel effect. To account 
#   for this, I modeled sigma as a linear function of the linear predictor mu. After adding
#   random slopes for gender I had too many divergences. To avoid the problem, I updated 
#   the model for sigma by using the inverse logistic function:
#   sigma[n] = sigma0 + gamma * inv_logit((mu[n] - c)/s).
#   This model provides a softer way of assigning (about) sigma0 to small mu values and (about)
#   sigma0+gamma to large mu values. We must scale the inverse logistic function to the scale of 
#   y = wage. We decided to center the inverse logistic function at c=median(y)=27.1.
#   As for scaling s, we considered the width
#   w = quantile(y, 0.9) - quantile(y, 0.1) = 19.8.
#   Since inv_logit(-2.20) = 0.1 and inv_logit(2.20) = 0.9, then
#   s = w / (2.20 - (-2.20)) = 4.5.
# - First, I included the weights in the model as frequency weights, rescaled to reduce their size.
#   It works, but fit was not great. I then tried to remove the weights (so, all weights = to 1). And 
#   this fit the data much better. So, in the Stan data below, I make all weights equal to 1 (no 
#   weighing).

stan_code <- "
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N, K] X;
  vector[N] y;
  array[N] int<lower=1, upper=J> industry;
  vector[N] w_raw;              // frequency weights
  int<lower=1, upper=K> gender_col; // X column holding gender
}

transformed data {
  matrix[N, K] Q_ast;
  matrix[K, K] R_ast;
  matrix[K, K] R_ast_inverse;
  vector[N] gender;

  Q_ast = qr_thin_Q(X) * sqrt(N - 1);
  R_ast = qr_thin_R(X) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
  gender = col(X, gender_col);
}

parameters {
  real alpha;
  vector[K] theta;
  vector[J] z_industry;
  real<lower=0> sigma0;         // sigma at min(mu) = sigma0
  real<lower=0> gamma;          // sigma at max(mu) = sigma0 + gamma
  real<lower=0> sigma_industry;
  vector[J] z_gender; 
  real<lower=0> sigma_gender; 
}

transformed parameters {
  vector[J] alpha_industry;
  vector[J] gender_industry;
  vector[N] mu;
  vector[N] sigma;

  alpha_industry = sigma_industry * z_industry;
  gender_industry = sigma_gender   * z_gender;
  mu = alpha + Q_ast * theta + alpha_industry[industry] + gender .* gender_industry[industry];

  real mu_min = min(mu);
  real mu_max = max(mu);
  for (n in 1:N) {
    sigma[n] = sigma0 + gamma * inv_logit((mu[n] - 27.1)/4.5); 
  }
}

model {
  alpha  ~ normal(28.8, 10);
  theta  ~ normal(0, 5);
  sigma0 ~ normal(0, 20); 
  gamma  ~ normal(0, 20);  
  sigma_industry ~ normal(0, 3);
  z_industry ~ normal(0, 1);
  sigma_gender ~ normal(0, 3);
  z_gender ~ normal(0, 1); 

  for (n in 1:N) {
    target += w_raw[n] * normal_lpdf(y[n] | mu[n], sigma[n]);
  }
}

generated quantities {
  vector[K] beta;
  vector[N] y_rep;

  // === posterior R^2 用 ===
  real RSS = 0;
  real TSS = 0;
  real y_mean = mean(y);
  real R2;

  beta = R_ast_inverse * theta;

  // y_rep の生成 + RSS の累積
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu[n], sigma[n]);
    RSS += square(y[n] - mu[n]);
    TSS += square(y[n] - y_mean);
  }

  R2 = 1 - RSS / TSS;
}
"

# write_stan_file() saves the Stan file to a temporary directory, so we don't 
# need to worry about GitHub tracking it (no need for .gitignore too).
stan_file <- write_stan_file(stan_code)



# ===== 7. Stan data =====
stan_data <- list(
  N        = nrow(data),
  K        = ncol(X),
  J        = length(unique(data$Industry)),
  X        = X,
  y        = data$Wage_man,
  industry = data$Industry,
  w_raw    = rep(1, nrow(data)), # as.vector(data$Employees / mean(data$Employees)) 
  gender_col = 1
)



# ===== 8. Sampling =====
# cmdstan_model() will now save the compiled model in the same temp directory.
mod       <- cmdstan_model(stan_file)

fit <- mod$sample(
  data            = stan_data,
  iter_warmup     = 1000,
  iter_sampling   = 2000,
  chains          = 4,
  parallel_chains = 4,
  seed            = 123
  #init = 0#, 
  # adapt_delta     = 0.995, # added to avoid divergences
  # max_treedepth   = 15    # added to avoid divergences
)
sum_df <- fit$summary()
sum_df[order(-sum_df$rhat), c("variable", "rhat")][1:20, ]



# ===== 9. Results =====
fit$summary(c("alpha", "beta", "sigma0", "gamma", "sigma_industry", "sigma_gender"))
fit$cmdstan_diagnose()

posterior <- as_draws_df(fit)
mcmc_hist(posterior, pars = c("beta[1]", "beta[2]", "beta[3]"))

# Industry random effects:
alpha_draws <- posterior::as_draws_matrix(fit$draws("alpha_industry"))
alpha_means <- colMeans(alpha_draws)
print(alpha_means)

industry_levels <- levels(data$Industry.labs)
industry_levels

data.frame(industry = industry_levels, alpha = round(alpha_means, 2))

# ===== 9.1 Posterior R^2 =====
R2_df     <- data.frame(R2 = posterior$R2)
R2_mean   <- mean(R2_df$R2)
R2_median <- median(R2_df$R2)
R2_CI     <- quantile(R2_df$R2, c(0.025, 0.975))

cat("Posterior R^2 (mean)   :", R2_mean, "\n")
cat("Posterior R^2 (median) :", R2_median, "\n")
cat("95% credible interval   :", R2_CI, "\n")

g <- ggplot(R2_df, aes(x = R2)) +
  geom_density(fill = "#d1e1ec", col = "#03396c") +
  labs(
    x = "R squared",
    y = "Density"
  ) +
  theme_minimal(base_size = 25)

g <- g +
  theme(plot.margin = margin(t = 5.5, r = 20, b = 5.5, l = 5.5))

ggsave(
  filename = "Figures/Rsquared.png",
  plot = g,
  width = 6.5,
  height = 4.5,
  units = "in",
  dpi = 300,
  bg = "white"
)

# ===== 9.2 Industry-specific gender slopes (RQ3) =====
# gender slope in industry j: beta[1] + gender_industry[j]

# Draws
beta1_draws <- as.numeric(posterior[["beta[1]"]])

# gender_industry draws: 
gender_ind_draws <- posterior::as_draws_matrix(fit$draws("gender_industry"))

J <- ncol(gender_ind_draws)

# Compute slopes draws:
slope_draws <- sweep(gender_ind_draws, 1, beta1_draws, FUN = "+")

# Summaries per industry:
slope_mean <- apply(slope_draws, 2, mean)
slope_ci   <- apply(slope_draws, 2, quantile, probs = c(0.025, 0.975))

gender_slope_df <- data.frame(
  industry = industry_levels,
  slope_mean = slope_mean,
  slope_low  = slope_ci[1, ],
  slope_high = slope_ci[2, ]
)

# Sort industries by mean slope (most negative means larger gap since male=0, female=1)
gender_slope_df <- gender_slope_df[order(gender_slope_df$slope_mean), ]
gender_slope_df$industry <- factor(
  gender_slope_df$industry,
  levels = gender_slope_df$industry
)

print(gender_slope_df)

# write_csv(gender_slope_df, "Figures/industry_gender_slopes.csv")

# Plot: industry-specific gender slopes with 95% credible intervals
p_gender_slopes <- ggplot(gender_slope_df, aes(x = slope_mean, y = industry)) +
  geom_vline(xintercept = 0, linetype = 2) +
  geom_errorbarh(
    aes(xmin = slope_low, xmax = slope_high),
    height = 0.22,
    color = "#d1e1ec",
    linewidth = 1
  ) +
  geom_point(size = 2.8, color = "#03396c") +
  labs(
    x = "Gender effect on wage (in 10,000 yen)",
    y = NULL
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.title.x = element_text(size = 16),
    axis.text.x  = element_text(size = 13),
    
    # ★ y軸を強化
    axis.text.y  = element_text(size = 14, face = "bold"),
    
    # ★ y軸ラベルが長いので左余白を増やす
    plot.margin  = margin(t = 5.5, r = 5.5, b = 5.5, l = 28),
    
    # 見やすさ
    panel.grid.major.y = element_blank()
  )

ggsave( filename = "Figures/industry_gender_slopes.png", plot = p_gender_slopes, width = 11, height = 5.5, units = "in", dpi = 300, bg = "white" )



# ===== 10. Posterior predictive checks =====
yrep_cols <- grep("^y_rep\\[", names(posterior), value = TRUE)
y_rep <- as.matrix(posterior[, yrep_cols])

# Observed data
y_obs <- stan_data$y

library(bayesplot)

bayesplot_theme_set(
  theme_bw(base_size = 16) +
    theme(
      axis.title = element_text(size = 20),
      axis.text  = element_text(size =16),
      legend.title = element_text(size = 16),
      legend.text  = element_text(size = 16),
      plot.title = element_text(size = 24)
    )
)


# Save plots as objects, to combine later:
p1 <- ppc_hist(y_obs, y_rep[1:8, ]) +
  labs(x = "Wage (man-yen)", y = "Frequency")

p2 <- ppc_dens_overlay(y_obs, y_rep[1:50, ]) +
  labs(x = "Wage (man-yen)", y = "Density")

p4 <- ppc_stat(y_obs, y_rep, stat = "mean") +
  labs(x = "Predicted Mean", y = "Density")

p5 <- ppc_stat(y_obs, y_rep, stat = "sd") +
  labs(x = "Predicted SD", y = "Density")

p6 <- ppc_boxplot(y_obs, y_rep[1:50, ]) +
  labs(x = "Observation Index", y = "Wage (man-yen)")

# This plot won't make it to the thesis:
# ppc_intervals_grouped(y_obs, y_rep, group = data$Industry)

p3 <- ppc_ecdf_overlay(y_obs, y_rep) +
  labs(x = "Wage (man-yen)", y = "ECDF")

# Combine plots p1 through p6:
# After looking at the result, I further tweaked with the row heights.
layout <- matrix(
  c(1, 1,
    2, 3,
    4, 5,
    6, 6),
  ncol = 2,
  byrow = TRUE
)

g <- arrangeGrob(
  p1, p2, p3, p4, p5, p6,
  layout_matrix = layout,
  heights = c(1, 0.8, 0.8, 1.5)
)

g_final <- grobTree(
  g,
  
  # (a) top panel
  textGrob("(a)", x = unit(0.98, "npc"), y = unit(0.98, "npc"),
           just = c("right", "top"),
           gp = gpar(fontsize = 20, fontface = "bold")),
  
  # (b)
  textGrob("(b)", x = unit(0.48, "npc"), y = unit(0.73, "npc"),
           just = c("right", "top"),
           gp = gpar(fontsize = 20, fontface = "bold")),
  
  # (c)
  textGrob("(c)", x = unit(0.98, "npc"), y = unit(0.73, "npc"),
           just = c("right", "top"),
           gp = gpar(fontsize = 20, fontface = "bold")),
  
  # (d)
  textGrob("(d)", x = unit(0.48, "npc"), y = unit(0.6, "npc"),
           just = c("right", "top"),
           gp = gpar(fontsize = 20, fontface = "bold")),
  
  # (e)
  textGrob("(e)", x = unit(0.98, "npc"), y = unit(0.6, "npc"),
           just = c("right", "top"),
           gp = gpar(fontsize = 20, fontface = "bold")),
  
  # (f) bottom panel
  textGrob("(f)", x = unit(0.98, "npc"), y = unit(0.28, "npc"),
           just = c("right", "top"),
           gp = gpar(fontsize = 20, fontface = "bold"))
)

ggsave(
  filename = "Figures/plot_PPCs.png",
  plot = g_final,
  width = 6.5,
  height = 9.35,
  units = "in",
  dpi = 300,
  bg = "white"
)



# ===== 11. Residuals =====
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
hist(resid_std, breaks = 20, freq = FALSE, main = "")  # ~ N(0,1)
curve(dnorm(x), add=TRUE)
# See standardized residuals per predictor:
plot(stan_data$X[, 1], resid_std) # vs Gender
plot(stan_data$X[, 2], resid_std) # vs Edu
plot(stan_data$X[, 3], resid_std) # vs Age (a bit or a curvilinear relation here, not so great)

# Export combined plot for the thesis:
png(filename = "Figures/residuals_plot.png", width = 2050, height = 1800, res = 300)
layout(
  matrix(c(1, 3,
           2, 3),
         nrow = 2,
         byrow = TRUE),
  heights = c(1, 1)
)

# ---- (a) Residuals vs fitted ----
par(mar = c(4, 4, 1, .5), cex = 1.25)
plot(
  mu_hat, resid,
  xlab = "",
  ylab = "",
  main = "", las = 1, pch = 4, lwd = 1.2, bty = "n", col = "#d1e1ec", 
  xlim = c(15, 45), xaxt = "n", 
  ylim = c(-20, 40), yaxt = "n"
)
axis(1, seq(15, 45, 10), las = 1, cex.axis = 1.2)
axis(2, seq(-20, 40, 20), las = 1, cex.axis = 1.2)
abline(h = 0, col = "#03396c", lty = 2)
mtext("Mean predicted values", 1, 2.5, cex = 1.5)
mtext("Residuals", 2, 2.5, cex = 1.5)
mtext("(a)", side = 3, line = -1, adj = 0.95, font = 2, cex = 1.3)

# ---- (b) Standardized residuals vs fitted ----
par(mar = c(4, 4, 1, .5), cex = 1.25)
plot(
  mu_hat, resid_std,
  xlab = "",
  ylab = "",
  main = "", pch = 4, lwd = 1.2, bty = "n", col = "#d1e1ec", 
  xlim = c(15, 45), xaxt = "n", 
  ylim = c(-3, 5), yaxt = "n"
)
axis(1, seq(15, 45, 10), las = 1, cex.axis = 1.2)
axis(2, c(-3, 0, 5), las = 1, cex.axis = 1.2)
abline(h = 0, col = "#03396c", lty = 2)
mtext("Mean predicted values", 1, 2.5, cex = 1.5)
mtext("Standardized ", 2, 2.8, cex = 1.5)
mtext("residuals", 2, 1.7, cex = 1.5)
mtext("(b)", side = 3, line = -1, adj = 0.95, font = 2, cex = 1.3)

# ---- (c) Histogram of standardized residuals ----
par(mar = c(4, 4, 1, .5), cex = 1.25)
hist(
  resid_std,
  breaks = 20,
  freq = FALSE,
  main = "",
  xlab = "",
  ylab = "",
  col = "#c5d9e7",
  border = "white", 
  xlim = c(-3, 6), xaxt = "n", ylim = c(0, .5), las = 1
)
curve(dnorm(x), add = TRUE, lwd = 2, col = "#03396c")
axis(1, seq(-3, 6, 3), cex.axis = 1.2)
mtext("Mean predicted values", 1, 2.5, cex = 1.5)
mtext("Residuals", 2, 2.5, cex = 1.5)
mtext("(c)", side = 3, line = -1, adj = 0.95, font = 2, cex = 1.3)
dev.off()


