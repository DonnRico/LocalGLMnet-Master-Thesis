
library(na.tools)
library(readr)
library(Hmisc)
library(tigerstats)
library(text)
library(usethis)
library(reticulate)
library(factoextra)
library(epiDisplay)
library(e1071)
library(forcats)
library(caTools)
library(dplyr)
library(gitcreds)
library(caret)
library(corrplot)
library(lubridate)
reticulate::conda_list()
library(tibble)
library(latex2exp)
library(gridExtra)
library(leaps)

Sys.setenv(RETICULATE_PYTHON = "C:\\Users\\desktop\\anaconda3\\envs\\textrpp_condaenv/python.exe")

loan <- read_csv("loan.csv")

theme_set(theme_bw())

set.seed(87031800)


loan$earliest_cr_line <- my(loan$earliest_cr_line)
loan$last_pymnt_d <- my(loan$last_pymnt_d)
loan$last_credit_pull_d <- my(loan$last_credit_pull_d)
loan$issue_d <- my(loan$issue_d)
loan$earliest_cr_line_year <- as.numeric(year(loan$earliest_cr_line))
loan$earliest_cr_line_month <- as.numeric(month(loan$earliest_cr_line))
loan$last_pymnt_d_year <- as.numeric(year(loan$last_pymnt_d))
loan$last_pymnt_d_month <- as.numeric(month(loan$last_pymnt_d))
loan$last_credit_pull_d_year <- as.numeric(year(loan$last_credit_pull_d))
loan$last_credit_pull_d_month <- as.numeric(month(loan$last_credit_pull_d))
loan$issue_d_year <- as.numeric(year(loan$issue_d))
loan$issue_d_month <- as.numeric(month(loan$issue_d))

loan <- subset(loan, select = -c(earliest_cr_line, last_pymnt_d, last_credit_pull_d, issue_d))

loan$int_rate <- as.numeric(sub("%", "", loan$int_rate))
loan$revol_util <- as.numeric(sub("%", "", loan$revol_util))
loan <- loan %>% filter(loan_status != "Current")

# my_bootstrap <- function(data, index) {
#   boot <- as.vector(unique(data[[index]]))
#   boot <- boot[!is.na(boot)]
#   for (j in 1:nrow(data)){
#     if (is.na(data[[index]])[j]) {
#       data[j,index] <- sample(boot, 1, replace = TRUE)
#     }
#   }
#   return(data)
# }
# 
# 
# clean.data.by_bootstrap <- function(data, threshold){
#   
#   any_na_values <- 0
#   full_na_values <- 0
#   to_remove <- c()
#   column_changed_mean <- c()
#   
#   for (i in 1:length(data)){
#     if(any_na(data[, i])){
#       if(all_na(data[, i])){
#         full_na_values <- full_na_values + 1
#         to_remove <- c(to_remove,i)
#       }
#       else {
#         any_na_values <- any_na_values + 1
#         if (sapply(data[,i], class) == "character") {
#           data[is.na(data[,i]), i] <- "no information"
#         }
#         else if ((sum(is.na(data[,i]))/nrow(data)) > threshold){
#           to_remove <- c(to_remove,i)
#         }
#         else {
#           data <- my_bootstrap(data, i)
#           column_changed_mean <- c(column_changed_mean, names(data)[i])
#         }
#       }
#     }
#     if (length(unique(data[[i]])) == 1) {
#       to_remove <- c(to_remove,i)
#     }
#   }
#   data_clean <- data[, -to_remove]
#   
#   print(c(any_na_values, full_na_values))
#   
#   return(as.data.frame(data_clean))
# }



##################
## interpolation #
##################

# Find columns with the same unique value across all rows
same_vals_cols <- names(loan)[apply(loan, 2, function(x) length(unique(x))) == 1]

# Concatenate column names and insert "\" before each "_"
cols_str <- gsub("_", "\\\\_",
                 paste(same_vals_cols, collapse = ", "))

# Print the names of those columns
cat("Columns with same unique value across all rows: ", cols_str)
loan <- loan %>% select_if(~ length(unique(.)) > 1)


# Identify the dropped columns
dropped_cols <- names(loan)[colMeans(is.na(loan)) >= 0.5]

# Drop the columns from the dataframe
loan <- loan[, colMeans(is.na(loan)) < 0.5]

# Concatenate column names and insert "\" before each "_"
dropped_cols_str <- gsub("_", "\\\\_",
                         paste(dropped_cols, collapse = ", "))

# Print the names of the dropped columns
cat("Columns with more than 50 % of missing values: ", dropped_cols_str)


# Identify the dropped columns
dropped_cols <- names(loan)[apply(loan, 2, function(x) length(unique(na.omit(x)))) <= 1]


# Concatenate column names and insert "\" before each "_"
dropped_cols_str <- gsub("_", "\\\\_",
                         paste(dropped_cols, collapse = ", "))

# Print the names of the dropped columns
cat("Columns with NA and 1 unique value: ", dropped_cols_str)

# Drop the columns from the dataframe
loan <- loan[, apply(loan, 2, function(x) length(unique(na.omit(x)))) > 1]

# Identify the columns with at least one NA
na_cols <- names(loan)[colSums(is.na(loan)) > 0]

# Concatenate column names and insert "\" before each "_"
dropped_cols_str <- gsub("_", "\\\\_",
                         paste(na_cols, collapse = ", "))


# Print the column names with at least one NA
cat("Columns with NA: ", paste(dropped_cols_str, collapse = ", "))

drop <- c("id", "member_id", "grade","sub_grade", "zip_code", "url")
loan <- loan[,!(names(loan) %in% drop)]

loan$term <- as.factor(loan$term)
loan$emp_length <- as.factor(loan$emp_length)
loan$home_ownership <- as.factor(loan$home_ownership)
loan$verification_status <- as.factor(loan$verification_status)
loan$loan_status <- as.factor(loan$loan_status)
loan$addr_state <- as.factor(loan$addr_state)
loan$pub_rec <- as.factor(loan$pub_rec)
loan$total_acc <- as.factor(loan$total_acc)
loan$purpose <- as.factor(loan$purpose)
loan$pub_rec_bankruptcies <- factor(ifelse(loan$pub_rec_bankruptcies > 0, 1, 0))
loan$earliest_cr_line_year <- as.factor(loan$earliest_cr_line_year)
loan$earliest_cr_line_month <- as.factor(loan$earliest_cr_line_month)
loan$last_pymnt_d_year <- as.factor(loan$last_pymnt_d_year)
loan$last_pymnt_d_month <- as.factor(loan$last_pymnt_d_month)
loan$last_credit_pull_d_year <- as.factor(loan$last_credit_pull_d_year)
loan$last_credit_pull_d_month <- as.factor(loan$last_credit_pull_d_month)
loan$issue_d_year <- as.factor(loan$issue_d_year)
loan$issue_d_month <- as.factor(loan$issue_d_month)

cont_cols <- sapply(loan, is.numeric)
fact_cols <- sapply(loan, is.factor)


library(tidyverse)
# Subset the dataframe using only continuous and factor columns
loan_inter <- loan[, cont_cols | fact_cols]

missing_cols <- c("revol_util", "pub_rec_bankruptcies", "last_pymnt_d_year", "last_pymnt_d_month", "last_credit_pull_d_year", "last_credit_pull_d_month")
predictor_cols <- setdiff(colnames(loan_inter), missing_cols)

loan_complete <- loan_inter[complete.cases(loan_inter), ]
loan_missing <- loan_inter[!complete.cases(loan_inter), ]

missing_counts <- colSums(is.na(loan))
print(missing_counts)

pb_variable_level <- c("addr_state", "home_ownership")
predictor_cols <- setdiff(colnames(loan_inter), c(missing_cols, pb_variable_level))

predicted_vals <- c()

for (col in missing_cols) {
  
  y_train <- loan_complete[[col]]
  x_train <- loan_complete[,!(names(loan_complete) %in% missing_cols)]
  x_train <- x_train[,!(names(x_train) %in% pb_variable_level)]
  # Train the regression model
  formula_str <- paste(col, "~", paste(predictor_cols, collapse = " + "))
  model <- lm(formula_str, data = loan_complete)
  
  # Use the trained model to predict missing values in the test set
  na_rows <- is.na(loan_missing[[col]])
  
  x_test <- loan_missing[na_rows,!(names(loan_missing) %in% missing_cols)]
  x_test <- x_test[,!(names(x_test) %in% pb_variable_level)]
  y_pred <- predict(model, newdata = x_test)
  
  
  # Replace the missing values in the original dataframe with the predicted values
  predicted_vals[[col]] <- y_pred
}

predicted_vals$pub_rec_bankruptcies <- round(predicted_vals$pub_rec_bankruptcies)
predicted_vals$last_pymnt_d_year <- round(predicted_vals$last_pymnt_d_year)
predicted_vals$last_pymnt_d_month <- round(predicted_vals$last_pymnt_d_month)
predicted_vals$last_credit_pull_d_year <- round(predicted_vals$last_credit_pull_d_year)
predicted_vals$last_credit_pull_d_month <- round(predicted_vals$last_credit_pull_d_month)


## Replacing for revol

first_list_values <- unlist(predicted_vals[[1]])
first_list_values <- unname(first_list_values)

na_rows_revol <- is.na(loan$revol_util)
loan[na_rows_revol, "revol_util"] <- first_list_values

missing_counts <- colSums(is.na(loan))
print(missing_counts)

## Replacing for pub_rec_bankruptcies

list_values <- unlist(predicted_vals[[2]])
list_values <- unname(list_values)

na_rows_revol <- is.na(loan$pub_rec_bankruptcies)
loan$pub_rec_bankruptcies <- as.numeric(loan$pub_rec_bankruptcies)
loan[na_rows_revol, "pub_rec_bankruptcies"] <- list_values
loan$pub_rec_bankruptcies <- as.factor(loan$pub_rec_bankruptcies)
levels(loan$pub_rec_bankruptcies) <- c("0", "1")

## Replacing for last_pymnt_d_year

current_levels <- levels(loan$last_credit_pull_d_year)
list_values <- unlist(predicted_vals[[3]])
list_values <- unname(list_values)

na_rows_revol <- is.na(loan$last_pymnt_d_year)
loan$last_pymnt_d_year <- as.numeric(loan$last_pymnt_d_year)
loan[na_rows_revol, "last_pymnt_d_year"] <- list_values
loan$last_pymnt_d_year <- as.factor(loan$last_pymnt_d_year)
levels(loan$last_pymnt_d_year) <- current_levels

## Replacing for last_pymnt_d_month

current_levels <- levels(loan$last_pymnt_d_month)
list_values <- unlist(predicted_vals[[4]])
list_values <- unname(list_values)

na_rows_revol <- is.na(loan$last_pymnt_d_month)
loan$last_pymnt_d_month <- as.numeric(loan$last_pymnt_d_month)
loan[na_rows_revol, "last_pymnt_d_month"] <- list_values
loan$last_pymnt_d_month <- as.factor(loan$last_pymnt_d_month)
levels(loan$last_pymnt_d_month) <- current_levels

## Replacing for last_credit_pull_d_year

current_levels <- levels(loan$last_credit_pull_d_year)
list_values <- unlist(predicted_vals[[5]])
list_values <- unname(list_values)

na_rows_revol <- is.na(loan$last_credit_pull_d_year)
loan$last_credit_pull_d_year <- as.numeric(loan$last_credit_pull_d_year)
loan[na_rows_revol, "last_credit_pull_d_year"] <- list_values
loan$last_credit_pull_d_year <- as.factor(loan$last_credit_pull_d_year)
levels(loan$last_credit_pull_d_year) <- current_levels

## Replacing for last_credit_pull_d_month

current_levels <- levels(loan$last_credit_pull_d_month)
list_values <- unlist(predicted_vals[[6]])
list_values <- unname(list_values)

na_rows_revol <- is.na(loan$last_credit_pull_d_month)
loan$last_credit_pull_d_month <- as.numeric(loan$last_credit_pull_d_month)
loan[na_rows_revol, "last_credit_pull_d_month"] <- list_values
loan$last_credit_pull_d_month <- as.factor(loan$last_credit_pull_d_month)
levels(loan$last_credit_pull_d_month) <- current_levels

### replace char with no information

char_cols <- sapply(loan, is.character)
loan[, char_cols][is.na(loan[, char_cols])] <- "no information"

missing_counts <- colSums(is.na(loan))
print(missing_counts)


### replace the name of level for better interpretation 


##################
## Preprocessing #
##################

new_data <- loan

# new_data <- clean.data.by_bootstrap(loan, 0.5)



num_cols <- unlist(lapply(new_data, is.numeric))

data_correlations <- new_data[ , c(num_cols)]

corrplot(round(cor(data_correlations),1), method = "color",
         addCoef.col = "black",tl.col="black", tl.srt=45, insig = "blank")
drop <- c("funded_amnt", "funded_amnt_inv", "total_pymnt", "total_pymnt_inv",
          "total_rec_prncp", "total_rec_int", "installment", "collection_recovery_fee", "pub_rec", "total_acc")
data_correlations <- data_correlations[,!(names(data_correlations) %in% drop)]
corrplot(round(cor(data_correlations),1), method = "color",
         addCoef.col = "black",tl.col="black", tl.srt=45, insig = "blank")
new_data <- new_data[,!(names(new_data) %in% drop)]


levels(new_data$emp_length)[levels(new_data$emp_length)=="n/a"] <- "no information"

categ_cols <- unlist(lapply(new_data, is.factor))
data_categorical <- new_data[ , c(categ_cols)]

for(i in names(data_categorical)){
  data_categorical[i] <- sapply(data_categorical[i], as.numeric)
}

corrplot(round(cor(data_categorical),1), method = "color",
         addCoef.col = "black",tl.col="black", tl.srt=45, insig = "blank")

corrplot(round(cor(cbind(data_categorical, data_correlations)),1), method = "color",
         addCoef.col = "black",tl.col="black", tl.srt=45, insig = "blank")

##################
## Outliers anal #
##################

missing_counts <- colSums(is.na(new_data))
print(missing_counts)



###############
# Description #
###############


describe(new_data)

html(describe(new_data), size=85, tabular=TRUE,
     greek=TRUE, scroll=FALSE)

tab1(new_data$loan_status, sort.group = "decreasing", cum.percent = TRUE)

mean1 <- new_data %>% group_by(loan_status) %>% summarise(mean_loan=mean(int_rate))

ggplot(new_data, aes(x=loan_amnt, y=int_rate)) +
  geom_point(aes(color=loan_status, alpha = 0.2)) +
  geom_hline(data=mean1, aes(yintercept=mean_loan, col=loan_status), linewidth = 1.5)

mean1 <- new_data %>% group_by(loan_status) %>% summarise(mean_inc=mean(log(annual_inc)))

ggplot(new_data, aes(x=loan_amnt, y=log(annual_inc))) +
  geom_point(aes(color=loan_status)) +
  geom_hline(data=mean1, aes(yintercept=mean_inc, col=loan_status), linewidth = 1.5)

ggplot(new_data, aes(x = purpose, fill = loan_status)) + 
  geom_bar(position = 'fill') +
  geom_text(aes(x = purpose, 
                label = scales::percent(after_stat(count / tapply(count, x, sum)[x])), 
                group = loan_status), position = "fill", stat = "count") + labs(y = "proportion")+
  scale_fill_brewer(palette="Paired")

  
######################
# k-means emp_length #
######################


xtabs(~emp_length+loan_status, data = new_data)
tab <- as.data.frame(rowPerc(xtabs(~emp_length+loan_status, data = new_data)))
tab <- cbind(tab[tab$loan_status == "Charged Off",], tab[tab$loan_status == "Fully Paid",])
tab <- tab[, c(1,3,6)]
colnames(tab) <- c("emp_length", "Charged_Off", "Fully_Paid")


ggplot(tab, aes(x = Charged_Off, y = Fully_Paid)) + geom_point(aes(color = emp_length))

rownames(tab) <- tab$emp_length
tab <- tab[, -1]

fviz_nbclust(tab, kmeans, method = "silhouette", k.max = 8)
fviz_nbclust(tab, kmeans, method = "wss", k.max = 8)

km_emp_length <- kmeans(tab, 4, nstart = 25)

levels(new_data$emp_length) <- (as.data.frame(km_emp_length$cluster))[,1]

fviz_cluster(km_emp_length, tab, ggtheme = theme_bw(), labelsize = 15) + ggtitle(label="emp_length levels clustering") 



####################
## k-means purpose #
####################


tab <- as.data.frame(rowPerc(xtabs(~purpose+loan_status, data = new_data)))
tab <- cbind(tab[tab$loan_status == "Charged Off",], tab[tab$loan_status == "Fully Paid",])
tab <- tab[, c(1,3,6)]
colnames(tab) <- c("purpose", "Charged_Off", "Fully_Paid")


ggplot(tab, aes(x = Charged_Off, y = Fully_Paid)) + geom_point(aes(color = purpose))

rownames(tab) <- tab$purpose
tab <- tab[, -1]

fviz_nbclust(tab, kmeans, method = "silhouette", k.max = 8)
fviz_nbclust(tab, kmeans, method = "wss", k.max = 8)

km_purpose <- kmeans(tab, 3, nstart = 25)

km_purpose$size
km_purpose$cluster

levels(new_data$purpose) <- (as.data.frame(km_purpose$cluster))[,1]

fviz_cluster(km_purpose, tab, ggtheme = theme_bw()) + ggtitle(label="purpose levels clustering")


#################################
# k-means earliest_cr_line_year #
#################################


xtabs(~earliest_cr_line_year+loan_status, data = new_data)

tab <- as.data.frame(rowPerc(xtabs(~earliest_cr_line_year+loan_status, data = new_data)))
tab <- cbind(tab[tab$loan_status == "Charged Off",], tab[tab$loan_status == "Fully Paid",])
tab <- tab[, c(1,3,6)]
colnames(tab) <- c("earliest_cr_line_year", "Charged_Off", "Fully_Paid")


ggplot(tab, aes(x = Charged_Off, y = Fully_Paid)) + geom_point(aes(color = earliest_cr_line_year))

rownames(tab) <- tab$earliest_cr_line_year
tab <- tab[, -1]

fviz_nbclust(tab, kmeans, method = "silhouette", k.max = 8)
fviz_nbclust(tab, kmeans, method = "wss", k.max = 8)

km_earliest_cr_line_year <- kmeans(tab, 6, nstart = 25)

levels(new_data$earliest_cr_line_year) <- (as.data.frame(km_earliest_cr_line_year$cluster))[,1]
fviz_cluster(km_earliest_cr_line_year, tab, ggtheme = theme_bw())  + ggtitle(label="earliest_cr_line_year levels clustering")

######################
# k-means addr_state #
######################


xtabs(~addr_state+loan_status, data = new_data)

tab <- as.data.frame(rowPerc(xtabs(~addr_state+loan_status, data = new_data)))
tab
tab <- cbind(tab[tab$loan_status == "Charged Off",], tab[tab$loan_status == "Fully Paid",])
tab <- tab[, c(1,3,6)]
colnames(tab) <- c("addr_state", "Charged_Off", "Fully_Paid")


ggplot(tab, aes(x = Charged_Off, y = Fully_Paid)) + geom_point(aes(color = addr_state))

rownames(tab) <- tab$addr_state
tab <- tab[, -1]

fviz_nbclust(tab, kmeans, method = "silhouette", k.max = 8)
fviz_nbclust(tab, kmeans, method = "wss", k.max = 8)

km_addr_state<- kmeans(tab, 3, nstart = 25)
km_addr_state$size
km_addr_state$cluster

levels(new_data$addr_state) <- (as.data.frame(km_addr_state$cluster))[,1]

fviz_cluster(km_addr_state, tab, ggtheme = theme_bw()) + ggtitle(label="addr_state levels clustering")


write.csv(new_data, 'data_preprocessed.csv')


#######################
# Reload step to fast #
#######################

new_data <- read_csv("data_preprocessed.csv")
new_data <- new_data[, -c(1)]


new_data$term <- as.factor(new_data$term)
new_data$emp_length <- as.factor(new_data$emp_length)
new_data$home_ownership <- as.factor(new_data$home_ownership)
new_data$verification_status <- as.factor(new_data$verification_status)
new_data$loan_status <- as.factor(new_data$loan_status)
new_data$addr_state <- as.factor(new_data$addr_state)
new_data$purpose <- as.factor(new_data$purpose)
new_data$pub_rec_bankruptcies <- factor(ifelse(new_data$pub_rec_bankruptcies > 0, 1, 0))
new_data$earliest_cr_line_year <- as.factor(new_data$earliest_cr_line_year)
new_data$earliest_cr_line_month <- as.factor(new_data$earliest_cr_line_month)
new_data$last_pymnt_d_year <- as.factor(new_data$last_pymnt_d_year)
new_data$last_pymnt_d_month <- as.factor(new_data$last_pymnt_d_month)
new_data$last_credit_pull_d_year <- as.factor(new_data$last_credit_pull_d_year)
new_data$last_credit_pull_d_month <- as.factor(new_data$last_credit_pull_d_month)
new_data$issue_d_year <- as.factor(new_data$issue_d_year)
new_data$issue_d_month <- as.factor(new_data$issue_d_month)




#######################
# Description part II #
#######################


blank_theme <- theme_minimal()+
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.border = element_blank(),
    panel.grid=element_blank(),
    axis.ticks = element_blank(),
    plot.title=element_text(size=14, face="bold")
  )


for (i in c(1:length(levels(new_data$addr_state)))) {
  
  data_adr_first <- new_data[new_data$addr_state == levels(new_data$addr_state)[i], c("loan_status", "addr_state") ]
  
  data_adr_first <- as.data.frame(table(data_adr_first$loan_status))
  
  Sub <- sum(data_adr_first$Freq)
  
  if (i < length(levels(new_data$addr_state))) {
    
    plot <- ggplot(data_adr_first, aes(x="", y = Freq, fill = Var1)) + geom_bar(width = 1, stat = "identity") + 
      coord_polar("y", start=0) + scale_fill_brewer("loan_status") + blank_theme +
      theme(axis.text.x=element_blank(), legend.position = "none", plot.title = element_text(hjust = 0.5, size = 30), plot.subtitle = element_text(hjust = 0.5, size = 25) )+
      geom_text(aes(label = round(Freq/sum(Freq),2)), position = position_stack(vjust = 0.5), size = 15) + 
      ggtitle(paste0("level: ", levels(new_data$addr_state)[i]), subtitle = paste0("Number of occurences:  ", Sub))
    
    print(plot)
    
  }
  
  else {
    
    plot <- ggplot(data_adr_first, aes(x="", y = Freq, fill = Var1)) + geom_bar(width = 1, stat = "identity") + 
      coord_polar("y", start=0) + scale_fill_brewer("loan_status") + blank_theme + 
      theme(axis.text.x=element_blank(),legend.title = element_blank(),legend.text=element_text(size=25), plot.title = element_text(hjust = 0.5, size = 30), plot.subtitle = element_text(hjust = 0.5, size = 25) )+
      geom_text(aes(label = round(Freq/sum(Freq),2)), position = position_stack(vjust = 0.5), size = 15) + 
      ggtitle(paste0("level: ", levels(new_data$addr_state)[i]), subtitle = paste0("Number of occurences: ", Sub)) + labs(color=NULL) 
    
    print(plot)
    
  }
  
}

for (i in c(1:length(levels(new_data$pub_rec_bankruptcies)))) {
  
  data_adr_first <- new_data[new_data$pub_rec_bankruptcies == levels(new_data$pub_rec_bankruptcies)[i], c("loan_status", "pub_rec_bankruptcies") ]
  
  data_adr_first <- as.data.frame(table(data_adr_first$loan_status))
  
  Sub <- sum(data_adr_first$Freq)
  
  if (i < length(levels(new_data$pub_rec_bankruptcies))) {
    
    plot <- ggplot(data_adr_first, aes(x="", y = Freq, fill = Var1)) + geom_bar(width = 1, stat = "identity") + 
      coord_polar("y", start=0) + scale_fill_brewer("loan_status") + blank_theme +
      theme(axis.text.x=element_blank(), legend.position = "none", plot.title = element_text(hjust = 0.5, size = 30), plot.subtitle = element_text(hjust = 0.5, size = 25) )+
      geom_text(aes(label = round(Freq/sum(Freq),2)), position = position_stack(vjust = 0.5), size = 15) + 
      ggtitle(paste0("level: ", levels(new_data$pub_rec_bankruptcies)[i]), subtitle = paste0("Number of occurences:  ", Sub))
    
    print(plot)
    
  }
  
  else {
    
    plot <- ggplot(data_adr_first, aes(x="", y = Freq, fill = Var1)) + geom_bar(width = 1, stat = "identity") + 
      coord_polar("y", start=0) + scale_fill_brewer("loan_status") + blank_theme + 
      theme(axis.text.x=element_blank(),legend.title = element_blank(),legend.text=element_text(size=25), plot.title = element_text(hjust = 0.5, size = 30), plot.subtitle = element_text(hjust = 0.5, size = 25) )+
      geom_text(aes(label = round(Freq/sum(Freq),2)), position = position_stack(vjust = 0.5), size = 15) + 
      ggtitle(paste0("level: ", levels(new_data$pub_rec_bankruptcies)[i]), subtitle = paste0("Number of occurences: ", Sub)) + labs(color=NULL) 
    
    print(plot)
    
  }
  
}


for (i in c(1:length(levels(new_data$term)))) {
  
  data_adr_first <- new_data[new_data$term == levels(new_data$term)[i], c("loan_status", "term") ]
  
  data_adr_first <- as.data.frame(table(data_adr_first$loan_status))
  
  Sub <- sum(data_adr_first$Freq)
  
  if (i < length(levels(new_data$term))) {
    
    plot <- ggplot(data_adr_first, aes(x="", y = Freq, fill = Var1)) + geom_bar(width = 1, stat = "identity") + 
      coord_polar("y", start=0) + scale_fill_brewer("loan_status") + blank_theme +
      theme(axis.text.x=element_blank(), legend.position = "none", plot.title = element_text(hjust = 0.5, size = 30), plot.subtitle = element_text(hjust = 0.5, size = 25) )+
      geom_text(aes(label = round(Freq/sum(Freq),2)), position = position_stack(vjust = 0.5), size = 15) + 
      ggtitle(paste0("level: ", levels(new_data$term)[i]), subtitle = paste0("Number of occurences:  ", Sub))
    
    print(plot)
    
  }
  
  else {
    
    plot <- ggplot(data_adr_first, aes(x="", y = Freq, fill = Var1)) + geom_bar(width = 1, stat = "identity") + 
      coord_polar("y", start=0) + scale_fill_brewer("loan_status") + blank_theme + 
      theme(axis.text.x=element_blank(),legend.title = element_blank(),legend.text=element_text(size=25), plot.title = element_text(hjust = 0.5, size = 30), plot.subtitle = element_text(hjust = 0.5, size = 25) )+
      geom_text(aes(label = round(Freq/sum(Freq),2)), position = position_stack(vjust = 0.5), size = 15) + 
      ggtitle(paste0("level: ", levels(new_data$term)[i]), subtitle = paste0("Number of occurences: ", Sub)) + labs(color=NULL) 
    
    print(plot)
    
  }
  
}

cdplot(loan_status ~ int_rate, data=new_data, col = c("blue", "lightblue"), title("Conditional Densities loan status~int rate"))

ggplot(new_data, aes(x = loan_status, y = int_rate, colour = term, fill = term)) + geom_violin(col = "blue", fill = "lightblue", alpha = 0.6, bw = 0.9) +
  geom_boxplot(alpha = 0.6) + ggtitle("boxplot of loan status ~ int_rate") + 
  theme(axis.title = element_text(size = 20), axis.text= element_text(size = 20),legend.title = element_blank(),legend.text=element_text(size=25), plot.title = element_text(hjust = 0.5, size = 25))




#######################
# logistic regression #
#######################


Dev.Ber = function(y,p){ 
  eps=1e-16
  -2*sum(y*log((p+eps)/(1-p+eps)) + log(1-p+eps))
}

odds_to_be_preproc <- new_data[, c("delinq_2yrs", "total_rec_late_fee")]

drop <- c("delinq_2yrs", "recoveries", "total_rec_late_fee", "desc", "title", "emp_title", "recoveries", "inq_last_6mths", "")
new_data <- new_data[,!(names(new_data) %in% drop)]

num_vars <- names(new_data)[sapply(new_data, is.numeric)]


# Cut numeric features into quartiles
for (var in num_vars) {
  breaks <- unique(quantile(new_data[[var]], probs = seq(0, 1, 0.25)))
  new_data[[var]] <- as.factor(cut(new_data[[var]], breaks = breaks, include.lowest = TRUE, labels = c("_first_q", "_second_q", "_third_q", "_fourth_q")))
}

new_data <- cbind(new_data, odds_to_be_preproc)


cut_points1 <- c(-1,0,3,4,11)
cut_point2 <- c(-1,0,15,30,45,60,181)

# Cut the continuous feature using the specified cut points
new_data$delinq_2yrs <- cut(new_data$delinq_2yrs, breaks = cut_points1)
new_data$total_rec_late_fee <- cut(new_data$total_rec_late_fee, breaks = cut_point2)

missing_counts <- colSums(is.na(new_data))
print(missing_counts)



Categ.to.Quant<-function(data,factor,removeLast=TRUE)
{
  y = paste("data$",sep = "",factor)
  x = eval(parse(text=y))
  ndata = length(x)          #number of lines in the dataset
  nlgen = length(levels(x))  #number of levels
  if (!removeLast)
  {nlgen = nlgen+1}  #number of levels
  lev   = levels(x)
  z     = matrix(0,ndata,nlgen-1)
  nv    = vector("character",nlgen-1)
  for (ct in 1:nlgen-1)
  {
    z[,ct] = ifelse(x==lev[ct],1,0)
    nv[ct] = paste(factor,sep="",lev[ct])
  }
  colnames(z)=nv
  #remove the column
  data <- data[, ! names(data) %in% factor, drop = F]
  data <- data.frame(data,z)
  return(data)
}

cols_with_periodicity <- grep("month|year", names(new_data))
df_glm <- subset(new_data, select = -cols_with_periodicity)

for (col in names(df_glm)) {
  if (col != "loan_status") {
    df_glm <- Categ.to.Quant(df_glm, col)
  }
}

df_glm$loan_status <- ifelse(df_glm$loan_status == "Charged Off", 1, 0)

set.seed(87031800)

train_idx <- sample(nrow(df_glm), nrow(df_glm) * 0.8)
train <- df_glm[train_idx, ]
test <- df_glm[-train_idx, ]


# Build a logistic regression model using all predictors
model <- glm(loan_status ~ ., family = binomial(link="logit"), data = train)
summary(model)

step.model <- stepAIC(model, direction = "both", trace = TRUE)
summary(step.model)

predict.train <- predict.glm(model, train, type = "response")
predict.test <- predict.glm(model, test, type = "response")

# Make predictions on the test set
predict.train.restr <- predict.glm(step.model, train, type = "response")
predict.test.rest <- predict.glm(step.model, test, type = "response")

# Evaluate the performance of the model

Dev.Ber(train$loan_status, predict.train)
Dev.Ber(test$loan_status, predict.test)

Dev.Ber(train$loan_status, predict.train.restr)
Dev.Ber(test$loan_status, predict.test.rest)

AIC(model)
AIC(step.model)

########################
# Sensivity analysis s #
########################

library(leaps)
y <- df_glm$loan_status
X <- as.matrix(df_glm[, -1])
set.seed(87031800)
# Define the number of repetitions and subsets
nreps <- 10
nsubsets <- 5

# Create a matrix to store the results
results <- matrix("", nsubsets, nreps)

# Perform stepwise selection on each subset of the data
for (i in 1:nreps) {
  # Randomly split the data into subsets
  subsets <- split(sample(nrow(X)), rep(1:nsubsets, length.out = nrow(X)))
  
  # Perform stepwise selection on each subset
  for (j in 1:nsubsets) {
    # Select the subset
    Xsubset <- X[subsets[[j]], ]
    ysubset <- y[subsets[[j]]]
    
    # Perform stepwise selection using AIC as the criterion
    model <- regsubsets(Xsubset, ysubset, method = "backward", nvmax = ncol(Xsubset), 
                        really.big = TRUE, intercept = TRUE, criterion = "aic")
    
    # Store the selected features in the results matrix
    results[j, i] <- paste0(colnames(Xsubset)[summary(model)$which], collapse = ",")
  }
}

# Print the results
print(results)


########################
# Calibration logistic #
########################

# Define named function
my_fun <- function(x) {
  return(c(mean = mean(x), n = length(x)))
}

# Pass named function to aggregate
to_comp <- aggregate(x = train$loan_status, 
                     FUN = my_fun, 
                     by = list(Low_Emp = train$emp_length1, 
                               Mid_emp = train$emp_length2, 
                               High_emp = train$emp_length3,
                               Low_term = train$term36.months, 
                               int1 = train$int_rate_first_q, 
                               int2 = train$int_rate_second_q,
                               int3 = train$int_rate_third_q))

to_comp2 <- aggregate(x = predict.train, 
                     FUN = my_fun, 
                     by = list(Low_Emp = train$emp_length1, 
                               Mid_emp = train$emp_length2, 
                               High_emp = train$emp_length3,
                               Low_term = train$term36.months, 
                               int1 = train$int_rate_first_q, 
                               int2 = train$int_rate_second_q,
                               int3 = train$int_rate_third_q))
O <- to_comp$`my_fun.train$loan_status`[,1]*to_comp$`my_fun.train$loan_status`[,2]
E <- to_comp2$my_fun.predict.train[,1] * to_comp2$my_fun.predict.train[,2]
n = to_comp$`my_fun.train$loan_status`[,2]

HLT = sum((O - E)^2/(E*(1-to_comp2$my_fun.predict.train[,1])))
pval = pchisq(HLT, df=(nrow(to_comp)-2), lower.tail = F)
cat("C =", HLT,"\n")
cat("p-value =", pval)

png(filename = "calibration_glm_train.png", width = 7, height = 6, units = 'in', res=600)
plot(to_comp2$my_fun.predict.train[,1], to_comp$`my_fun.train$loan_status`[,1], pch=16, col = "blue", xlim=c(0,1),ylim=c(0,1),
     main = "Group Calibration (logistic regression)", xlab = "Predicted Frequency", ylab="True Frequency")
abline(a=0,b=1,lty=3)
grid()
legend("topleft", legend = c(TeX(sprintf("$\\hat{C}$ = %.3f", round(HLT, 3))), 
                             paste("p-value =", round(pval,3))),
       lty=c(0,0))
dev.off()

to_comp3 <- aggregate(x = test$loan_status, 
                     FUN = my_fun, 
                     by = list(Low_Emp = test$emp_length1, 
                               Mid_emp = test$emp_length2, 
                               High_emp = test$emp_length3,
                               Low_term = test$term36.months, 
                               int1 = test$int_rate_first_q, 
                               int2 = test$int_rate_second_q,
                               int3 = test$int_rate_third_q))

to_comp4 <- aggregate(x = predict.test, 
                      FUN = my_fun, 
                      by = list(Low_Emp = test$emp_length1, 
                                Mid_emp = test$emp_length2, 
                                High_emp = test$emp_length3,
                                Low_term = test$term36.months, 
                                int1 = test$int_rate_first_q, 
                                int2 = test$int_rate_second_q,
                                int3 = test$int_rate_third_q))

O <- to_comp3$`my_fun.test$loan_status`[,1]*to_comp3$`my_fun.test$loan_status`[,2]
E <- to_comp4$my_fun.predict.test[,1] * to_comp4$my_fun.predict.test[,2]
n = to_comp3$`my_fun.test$loan_status`[,2]

HLT = sum((O - E)^2/(E*(1-to_comp4$my_fun.predict.test[,1])))
pval = pchisq(HLT, df=(nrow(to_comp3)-2), lower.tail = F)
cat("C =", HLT,"\n")
cat("p-value =", pval)

png(filename = "calibration_glm_test.png", width = 7, height = 6, units = 'in', res=600)
plot(to_comp4$my_fun.predict.test[,1], to_comp3$`my_fun.test$loan_status`[,1], pch=16, col = "blue", xlim=c(0,1),ylim=c(0,1),
     main = "Group Calibration (logistic regression)", xlab = "Predicted Frequency", ylab="True Frequency")
abline(a=0,b=1,lty=3)
grid()
legend("topleft", legend = c(TeX(sprintf("$\\hat{C}$ = %.3f", round(HLT, 3))), 
                             paste("p-value =", round(pval,3))),
       lty=c(0,0))
dev.off()

########################
# Calibration stepwise #
########################

# Pass named function to aggregate
to_comp <- aggregate(x = train$loan_status, 
                     FUN = my_fun, 
                     by = list(Low_Emp = train$emp_length1, 
                               Mid_emp = train$emp_length2, 
                               High_emp = train$emp_length3,
                               Low_term = train$term36.months, 
                               int1 = train$int_rate_first_q, 
                               int2 = train$int_rate_second_q,
                               int3 = train$int_rate_third_q))

to_comp2 <- aggregate(x = predict.train.restr, 
                      FUN = my_fun, 
                      by = list(Low_Emp = train$emp_length1, 
                                Mid_emp = train$emp_length2, 
                                High_emp = train$emp_length3,
                                Low_term = train$term36.months, 
                                int1 = train$int_rate_first_q, 
                                int2 = train$int_rate_second_q,
                                int3 = train$int_rate_third_q))
O <- to_comp$`my_fun.train$loan_status`[,1]*to_comp$`my_fun.train$loan_status`[,2]
E <- to_comp2$my_fun.predict.train.restr[,1] * to_comp2$my_fun.predict.train.restr[,2]
n = to_comp$`my_fun.train$loan_status`[,2]

HLT = sum((O - E)^2/(E*(1-to_comp2$my_fun.predict.train.restr[,1])))
pval = pchisq(HLT, df=(nrow(to_comp)-2), lower.tail = F)
cat("C =", HLT,"\n")
cat("p-value =", pval)

png(filename = "calibration_glm_rest_train.png", width = 7, height = 6, units = 'in', res=600)
plot(to_comp2$my_fun.predict.train.restr[,1], to_comp$`my_fun.train$loan_status`[,1], pch=16, col = "blue", xlim=c(0,1),ylim=c(0,1),
     main = "Group Calibration (stepwise log regression)", xlab = "Predicted Frequency", ylab="True Frequency")
abline(a=0,b=1,lty=3)
grid()
legend("topleft", legend = c(TeX(sprintf("$\\hat{C}$ = %.3f", round(HLT, 3))), 
                             paste("p-value =", round(pval,3))),
       lty=c(0,0))
dev.off()

to_comp3 <- aggregate(x = test$loan_status, 
                      FUN = my_fun, 
                      by = list(Low_Emp = test$emp_length1, 
                                Mid_emp = test$emp_length2, 
                                High_emp = test$emp_length3,
                                Low_term = test$term36.months, 
                                int1 = test$int_rate_first_q, 
                                int2 = test$int_rate_second_q,
                                int3 = test$int_rate_third_q))

to_comp4 <- aggregate(x = predict.test.rest, 
                      FUN = my_fun, 
                      by = list(Low_Emp = test$emp_length1, 
                                Mid_emp = test$emp_length2, 
                                High_emp = test$emp_length3,
                                Low_term = test$term36.months, 
                                int1 = test$int_rate_first_q, 
                                int2 = test$int_rate_second_q,
                                int3 = test$int_rate_third_q))

O <- to_comp3$`my_fun.test$loan_status`[,1]*to_comp3$`my_fun.test$loan_status`[,2]
E <- to_comp4$my_fun.predict.test.rest[,1] * to_comp4$my_fun.predict.test.rest[,2]
n = to_comp3$`my_fun.test$loan_status`[,2]

HLT = sum((O - E)^2/(E*(1-to_comp4$my_fun.predict.test.rest[,1])))
pval = pchisq(HLT, df=(nrow(to_comp3)-2), lower.tail = F)
cat("C =", HLT,"\n")
cat("p-value =", pval)

png(filename = "calibration_glm_rest_test.png", width = 7, height = 6, units = 'in', res=600)
plot(to_comp4$my_fun.predict.test.rest[,1], to_comp3$`my_fun.test$loan_status`[,1], pch=16, col = "blue", xlim=c(0,1),ylim=c(0,1),
     main = "Group Calibration (stepwise log regression)", xlab = "Predicted Frequency", ylab="True Frequency")
abline(a=0,b=1,lty=3)
grid()
legend("topleft", legend = c(TeX(sprintf("$\\hat{C}$ = %.3f", round(HLT, 3))), 
                             paste("p-value =", round(pval,3))),
       lty=c(0,0))
dev.off()



######################
#   regularization   #
######################

library(glmnet)

# Create matrices for the predictor variables and the response variable
x <- model.matrix(loan_status ~ ., data = train)
y <- train$loan_status

# Fit a Lasso regression model with different values of lambda
lasso.model <- glmnet(x, y, family = "binomial", alpha = 1)
ridge.model <- glmnet(x, y, family = "binomial", alpha = 0)
# View the convergence of coefficients for different values of lambda
png(filename = "lasso.png", width = 8, height = 5, units = 'in', res=600)
plot(lasso.model, xvar = "lambda", label = TRUE)
dev.off()
png(filename = "ridge.png", width = 8, height = 5, units = 'in', res=600)
plot(ridge.model, xvar = "lambda", label = TRUE)
dev.off()
# Compute the predictions on the test set for each value of lambda
test.x <- model.matrix(loan_status ~ ., data = test)
test.y <- test$loan_status
lasso.predictions <- predict(lasso.model, newx = test.x, s = seq(3e-04, 3e-01, by = 5e-04))

# Compute the classification accuracy for each value of lambda
for (i in 1:length(lasso.model$lambda)) {
  lasso.coef <- coef(lasso.model, s = lasso.model$lambda[i])
  lasso.predictions <- predict(lasso.model, newx = test.x, s = lasso.model$lambda[i])
  lasso.predictions <- ifelse(lasso.predictions > 0.5, 1, 0)
  acc <- sum(lasso.predictions == test.y) / length(test.y)
  cat("Lambda =", lasso.model$lambda[i], "Accuracy =", acc, "\n")
}


x <- model.matrix(loan_status ~ ., data = train)[,-1]
y <- train$loan_status

x.test <- model.matrix(loan_status ~ ., data = test)[,-1]
y.test <- test$loan_status

lambda_seq <- seq(0.0001, 5e-02, by = 5e-04)
nlambda <- length(lambda_seq)
error_train <- matrix(0, nrow=nlambda, ncol=1)
error_test <- matrix(0, nrow=nlambda, ncol=1)

for (i in 1:nlambda) {
  
  model <- glmnet(x, y, family = "binomial", lambda = lambda_seq[i], alpha = 1)
  
  beta <- coef(model)
  
  y_hat_train <- predict(model, newx = x, s = lambda_seq[i], type = "response")
  y_hat_test <- predict(model, newx = x.test, s = lambda_seq[i], type = "response")
  
  error_train[i] <- Dev.Ber(y, y_hat_train)
  error_test[i] <- Dev.Ber(y.test, y_hat_test)
  
  print(i)
  
}

to_plot <- as.data.frame(cbind(lambda_seq,error_train, error_test))

to_plot

# Set x-axis limits
xlims <- c(exp(-10), exp(-3))
png(filename = "lasso_perf.png", width = 15, height = 4, units = 'in', res=600)
# Create two separate plots with symmetric x-axis
plot1 <- ggplot(to_plot, aes(x=lambda_seq, y=V2)) +
  geom_line(color="red", linewidth = 1) +
  labs(x="Lambda Sequence", y="Deviance Train") +
  theme_minimal() +
  scale_x_log10(limits = xlims)

plot2 <- ggplot(to_plot, aes(x=lambda_seq, y=V3)) +
  geom_line(color="blue", linewidth = 1) +
  labs(x="Lambda Sequence", y="Deviance Test") +
  theme_minimal() +
  scale_x_log10(limits = xlims)

# Arrange the plots side by side
grid.arrange(plot1, plot2, ncol=2)
dev.off()


x <- model.matrix(loan_status ~ ., data = train)[,-1]
y <- train$loan_status

x.test <- model.matrix(loan_status ~ ., data = test)[,-1]
y.test <- test$loan_status

lambda_seq <- seq(0.0001, 5e-02, by = 5e-04)
nlambda <- length(lambda_seq)
error_train <- matrix(0, nrow=nlambda, ncol=1)
error_test <- matrix(0, nrow=nlambda, ncol=1)

for (i in 1:nlambda) {
  
  model <- glmnet(x, y, family = "binomial", lambda = lambda_seq[i], alpha = 0)
  
  beta <- coef(model)
  
  y_hat_train <- predict(model, newx = x, s = lambda_seq[i], type = "response")
  y_hat_test <- predict(model, newx = x.test, s = lambda_seq[i], type = "response")
  
  error_train[i] <- Dev.Ber(y, y_hat_train)
  error_test[i] <- Dev.Ber(y.test, y_hat_test)
  
  print(i)
  
}

to_plot <- as.data.frame(cbind(lambda_seq,error_train, error_test))

to_plot


# Set x-axis limits
xlims <- c(exp(-10), exp(-3))
png(filename = "ridge_perf2.png", width = 15, height = 4, units = 'in', res=600)
# Create two separate plots with symmetric x-axis
plot1 <- ggplot(to_plot, aes(x=lambda_seq, y=V2)) +
  geom_line(color="red", linewidth = 1) +
  labs(x="Lambda Sequence", y="Deviance Train") +
  theme_minimal() +
  scale_x_log10(limits = xlims)

plot2 <- ggplot(to_plot, aes(x=lambda_seq, y=V3)) +
  geom_line(color="blue", linewidth = 1) +
  labs(x="Lambda Sequence", y="Deviance Test") +
  theme_minimal() +
  scale_x_log10(limits = xlims)

# Arrange the plots side by side
grid.arrange(plot1, plot2, ncol=2)
dev.off()


####################
#  ELASTIC NET cv  #
####################

set.seed(87031800)

# Train the elastic net model with your custom loss function

my_loss <- function(data, lev = NULL, model = NULL) {
  mae <- mean(abs(data$obs - data$pred))
  return(mae)
}

Dev.Ber = function(y,p){ 
  eps=1e-16
  -2*sum(y*log((p+eps)/(1-p+eps)) + log(1-p+eps))
}

summary_fun <- function(data, lev, win) {
  Dev.Ber(data[, "obs"], data[, "pred"])
}

lambda = seq(0.0001,0.2,length=5)
nlambda <- length(lambda_seq)
alpha=seq(0,1,length=10)
nalpha = length(alpha)

error_train <- matrix(0, nrow=nlambda, ncol=1)
error_test <- matrix(0, nrow=nlambda, ncol=1)


alpha_vals <- seq(0.001, 0.2, by = 0.01)
lambda_vals <- seq(0.001, 0.2, by = 0.01)


grid <- expand.grid(alpha = alpha_vals, lambda = lambda_vals)
ctrl <- trainControl(method = "cv", number = 5, savePredictions = "final", summaryFunction = summary_fun, verboseIter = TRUE, returnResamp = "all")
model <- train(loan_status ~ ., data = train, method = "glmnet", trControl = ctrl, tuneGrid = grid, family = "binomial")


x <- model.matrix(loan_status ~ ., data = train)[,-1]
y <- train$loan_status

x.test <- model.matrix(loan_status ~ ., data = test)[,-1]
y.test <- test$loan_status

lambda_seq <- seq(0.0001, 5e-02, by = 5e-04)
nlambda <- length(lambda_seq)
error_train <- matrix(0, nrow=nlambda, ncol=1)
error_test <- matrix(0, nrow=nlambda, ncol=1)

library(glmnet)
library(caret)

# Split data into 5 folds

folds <- createFolds(train$loan_status, k = 5)

# Define lambda and alpha values to search over
lambda_vals <- c(0.1,0.05, 0.01,0.005, 0.001,0.0005, 0.0001, 0.00005, 0.00005)
alpha_vals <- seq(0, 1, length = 11)[-1]

# Initialize matrix to store cross-validation results
cv_results <- matrix(NA, nrow = length(lambda_vals) * length(alpha_vals), ncol = 5)
colnames(cv_results) <- c("lambda", "alpha", "number of folds", "dev.bern.train", "dev.bern.test")
row_idx <- 1

# Loop over lambda and alpha values
for (i in 1:length(lambda_vals)) {
  for (j in 1:length(alpha_vals)) {
    # Initialize vector to store Dev.Bern results for this combination of lambda and alpha
    dev_bern_results <- rep(NA, 5)
    dev_bern_results_train <- rep(NA, 5)
    # Loop over folds
    nfolds <- 0
    for (fold in 1:5) {
      set.seed(87031800)
      # Split data into training and test sets for this fold
      train_idx <- setdiff(seq_len(nrow(df_glm)), folds[[fold]])
      test_idx <- folds[[fold]]
      x_train <- as.matrix(df_glm[train_idx, -1])
      y_train <- df_glm$loan_status[train_idx]
      x_test <- as.matrix(df_glm[test_idx, -1])
      y_test <- df_glm$loan_status[test_idx]
      
      # Fit model with elastic net regularization
      model <- glmnet(x_train, y_train, family = "binomial", lambda = lambda_vals[i], alpha = alpha_vals[j])
      
      # Compute predicted probabilities for test set
      y_pred <- predict(model, newx = x_test, type = "response")
      y_pred_train <- predict(model, newx = x_train, type = "response")
      # Compute Dev.Bern loss for test set
      dev_bern <- Dev.Ber(y_test, y_pred)
      dev_bern_train <- Dev.Ber(y_train, y_pred_train)
      
      # Store results for this fold
      dev_bern_results[fold] <- dev_bern
      dev_bern_results_train[fold] <- dev_bern_train
      nfolds <- nfolds + 1
    }
    
    # Compute mean Dev.Bern loss across folds for this combination of lambda and alpha
    mean_dev_bern <- mean(dev_bern_results)
    mean_dev_bern_train <- mean(dev_bern_results_train)
    # Store results in cv_results matrix
    cv_results[row_idx, 1] <- lambda_vals[i]
    cv_results[row_idx, 2] <- alpha_vals[j]
    cv_results[row_idx, 3:5] <- c(nfolds, mean_dev_bern_train, mean_dev_bern)
    
    # Increment row index
    row_idx <- row_idx + 1
  }
}

# Print cross-validation results
print(cv_results)
print(dev_bern_results)

#####################
#  NN 1 layer only  #
#####################

new_data <- read_csv("data_preprocessed.csv")
new_data <- new_data[, -c(1)]

new_data$term <- as.factor(new_data$term)
new_data$emp_length <- as.factor(new_data$emp_length)
new_data$home_ownership <- as.factor(new_data$home_ownership)
new_data$verification_status <- as.factor(new_data$verification_status)
new_data$loan_status <- as.factor(new_data$loan_status)
new_data$addr_state <- as.factor(new_data$addr_state)
new_data$purpose <- as.factor(new_data$purpose)
new_data$pub_rec_bankruptcies <- factor(ifelse(new_data$pub_rec_bankruptcies > 0, 1, 0))
new_data$earliest_cr_line_year <- as.factor(new_data$earliest_cr_line_year)
new_data$earliest_cr_line_month <- as.factor(new_data$earliest_cr_line_month)
new_data$last_pymnt_d_year <- as.factor(new_data$last_pymnt_d_year)
new_data$last_pymnt_d_month <- as.factor(new_data$last_pymnt_d_month)
new_data$last_credit_pull_d_year <- as.factor(new_data$last_credit_pull_d_year)
new_data$last_credit_pull_d_month <- as.factor(new_data$last_credit_pull_d_month)
new_data$issue_d_year <- as.factor(new_data$issue_d_year)
new_data$issue_d_month <- as.factor(new_data$issue_d_month)

Categ.to.Quant<-function(data,factor,removeLast=TRUE)
{
  y = paste("data$",sep = "",factor)
  x = eval(parse(text=y))
  ndata = length(x)          #number of lines in the dataset
  nlgen = length(levels(x))  #number of levels
  if (!removeLast)
  {nlgen = nlgen+1}  #number of levels
  lev   = levels(x)
  z     = matrix(0,ndata,nlgen-1)
  nv    = vector("character",nlgen-1)
  for (ct in 1:nlgen-1)
  {
    z[,ct] = ifelse(x==lev[ct],1,0)
    nv[ct] = paste(factor,sep="",lev[ct])
  }
  colnames(z)=nv
  #remove the column
  data <- data[, ! names(data) %in% factor, drop = F]
  data <- data.frame(data,z)
  return(data)
}


var_standardization <- function(data,variable) {
  y <- paste("data$",sep = "",variable)
  x <- eval(parse(text=y))
  x <- (x-min(x))/(max(x)-min(x))
  
  name <- paste("Scaled_",sep = "", variable)
  
  data <- data[, ! names(data) %in% variable, drop = F]
  data <- data.frame(data,x)
  names(data)[names(data) == 'x'] <- name
  return(data)
}

cols_with_periodicity <- grep("month|year", names(new_data))
df_NN <- subset(new_data, select = -cols_with_periodicity)

drop <- c("recoveries", "desc", "title", "emp_title")
df_NN <- df_NN[,!(names(df_NN) %in% drop)]


for (col in names(df_NN)) {
  if (is.numeric(df_NN[[col]])) {
    df_NN <- var_standardization(df_NN, col)
  }
}

for (col in names(df_NN)) {
  if (is.factor(df_NN[[col]]) && col != "loan_status") {
    df_NN <- Categ.to.Quant(df_NN, col)
  }
}
df_NN$loan_status <- ifelse(df_NN$loan_status == "Charged Off", 1, 0)
namesnoregress <- c("loan_status")
namesregressor<-! names(df_NN) %in% namesnoregress

x_train   <- df_NN[,namesregressor]
x_train   <- data.matrix(x_train)
y_train   <- df_NN$loan_status
nb.inputs <- ncol(x_train)

######################################################################"
#   JUSTE POUR AFFICHAGE

library(neuralnet)
listcovariates<- colnames(x_train)
equation      <- paste("loan_status~",listcovariates[1])
for (j in c(2:length(listcovariates))){	
  equation=paste(equation,listcovariates[j],sep="+")
}
tmp       <- c()
namestmp  <-! names(df_NN) %in% tmp
modelplot <- neuralnet(formula= equation,
                       data=df_NN[,namestmp],hidden=c(20), threshold=2500000,stepmax = 1,
                       lifesign = "full",lifesign.step = 100, rep = FALSE)


png(filename = "NN1neuron.png", width = 7, height = 6, units = 'in', res = 600)
invisible(plot(modelplot, information = FALSE, radius = 0.1, show.weights = FALSE, fontsize = 10,
               intercept = TRUE, dimension = c(20), col.hidden = "red", col.entry = "green", col.out = "blue",
               col.hidden.synapse = "cadetblue", col.entry.synapse = "cadetblue", col.out.synapse = "cadetblue",
               arrow.length = 0.12))
dev.off()
dev.off()




