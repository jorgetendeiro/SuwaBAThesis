setwd("C:/Users/atomu1107/Downloads/")  # CSVが置いてある場所

# 必要パッケージ
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
#JT# library(dplyr)


# ===== 1. データ読み込み =====
data <- read_csv("Naoto_Suwa_data_v8_age_style_fixed.csv")

glimpse(data)  # データ構造を確認

# ===== 2. 概要統計（Table 1） =====
summary_table <- data %>%
  group_by(Gender) %>%
  summarise(
    mean_wage = mean(Wage_Yen, na.rm = TRUE),
    median_wage = median(Wage_Yen, na.rm = TRUE),
    total_employees = sum(Employees, na.rm = TRUE),
    n_cells = n()
  )

print(summary_table)

# ===== 3. 産業別の男女賃金格差（Figure 1） =====
industry_gap <- data %>%
  group_by(Industry, Gender) %>%
  summarise(mean_wage = mean(Wage_Yen, na.rm = TRUE)) %>%
  pivot_wider(names_from = Gender, values_from = mean_wage) %>%
  mutate(gap_ratio = Female / Male)

ggplot(industry_gap, aes(x = reorder(Industry, gap_ratio), y = gap_ratio)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    x = "Industry",
    y = "Female / Male average wage ratio",
    title = "Gender wage gap by industry (2024 Wage Survey)"
  ) +
  theme_minimal(base_size = 13)

# ===== 4. 学歴別の男女賃金差（Figure 2） =====
data <- data %>%
  mutate(
    Education = factor(
      Education,
      levels = c("JuniorHigh", "HighSchool",   "VocationalSchool","TechnicalSchoolJuniorCollege","University")
    )
  )

education_gap <- data %>%
  group_by(Education, Gender) %>%
  summarise(mean_wage = mean(Wage_Yen, na.rm = TRUE))

ggplot(education_gap, aes(x = Education, y = mean_wage, fill = Gender)) +
  geom_col(position = "dodge") +
  labs(
    title = "Average wage by education and gender",
    y = "Wage (Yen)",
    x = "Education level"
  ) +
  theme_minimal(base_size = 13)


# ===== 5. 年齢階級別の男女賃金差（Figure 3） =====

# 年齢階級を適切な順序で並べ替える（例：20-24 → 25-29 → ... → 70+）
data <- data %>%
  mutate(
    AgeGroup = factor(
      AgeGroup,
      levels = c("0-19", "20-24", "25-29", "30-34", "35-39",
                 "40-44", "45-49", "50-54", "55-59",
                 "60-64", "65-69", "70+")
    )
  )


# 年齢階級 × 性別ごとの平均賃金を計算
age_gap <- data %>%
  group_by(AgeGroup, Gender) %>%
  summarise(mean_wage = mean(Wage_Yen, na.rm = TRUE))

# 棒グラフを作成
ggplot(age_gap, aes(x = AgeGroup, y = mean_wage, fill = Gender)) +
  geom_col(position = "dodge") +
  labs(
    title = "Average wage by age group and gender",
    y = "Wage (Yen)",
    x = "Age group"
  ) +
  theme_minimal(base_size = 13)
