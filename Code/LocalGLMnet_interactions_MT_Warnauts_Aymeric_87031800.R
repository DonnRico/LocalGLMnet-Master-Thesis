library(readr)
library(tidyr)
library(ggplot2)
library(dplyr)

# Your working directory have to only contain the dataframe form the Python section related to "GRADIENT EXTRACTION"

files <- list.files()

for (file_name in files) {
  # read file into data frame
  file_df <- read.csv(file_name)
  
  # assign data frame a name without the ".csv" extension and "grad_compute_" prefix
  assign(gsub(".csv", "", gsub("grad_compute_", "", file_name)), file_df)
}

set.seed(87031800)
train_indices <- sample(seq_len(nrow(X)), size = round(0.8 * nrow(X)), replace = FALSE)
Xtrain <- X[train_indices, ]
Xtrain <- Xtrain[-nrow(Xtrain), ]

rm(doc_gradients.Rproj)
rm(file_df)
rm(X)
rm(beta_x)

verification_status_Not.Verified <- `verification_status_Not Verified`
rm(`verification_status_Not Verified`)
verification_status_Source.Verified <- `verification_status_Source Verified`
rm(`verification_status_Source Verified`)



# iterate over columns of Xtrain
for (col in names(Xtrain)) {
  
  df <- get(col)
  colnames(df) <- colnames(Xtrain)
  new_col_name <- "value"
  df[[new_col_name]] <- Xtrain[[col]]
  assign(col, df)
}

min_max <- data.frame(variable = c("Scaled_annual_inc", "Scaled_delinq_2yrs", "Scaled_dti", "Scaled_inq_last_6mths", "Scaled_int_rate", 
                                   "Scaled_last_pymnt_amnt", "Scaled_loan_amnt", "Scaled_open_acc", "Scaled_recoveries", "Scaled_revol_bal", 
                                   "Scaled_revol_util", "Scaled_total_rec_late_fee"), 
                      min_value = c(4000.0, 0, 0.0, 0, 5.42, 0.0, 500, 2, 0.0, 0, 0.0, 0.0),
                      max_value = c(6000000.0, 11, 29.99, 8, 24.4, 36115.2, 35000, 44, 29623.35, 149588, 99.9, 180.2))
min_max <- min_max[min_max$variable != "Scaled_recoveries", ]

for (i in seq(nrow(min_max))) {
  colname <- min_max[i, "variable"]
  df <- get(colname)
  print(paste("Before scaling:", colname, nrow(df)))
  min_val <- min_max[i, "min_value"]
  max_val <- min_max[i, "max_value"]
  df$value <- (df$value * (max_val - min_val) + min_val)
  print(paste("After scaling:", colname, nrow(df)))
  assign(colname, df)
}

# In this last part we construct the local regression, change the features each time or build a loop if you prefer
long_df <- Scaled_int_rate %>%
  select(starts_with("Scaled"), value) %>%
  pivot_longer(cols = -value, names_to = "column", values_to = "val")

theme_set(theme_bw())
ggplot(long_df, aes(x = value, y = val, col = column)) +
  geom_smooth(method = lm, formula = y ~ splines::bs(x, 10), se = FALSE) +
  labs(x = "int_rate", y = "interaction strengths")


