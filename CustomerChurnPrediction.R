#Sana : xgboost - Rf - karot model - lm 

# Required Libraries
library(tidyverse)
library(caret)
library(xgboost)
library(pROC)
library(lubridate)

# Function to Find Outliers
find_outliers <- function(x) {
  qnt <- quantile(x, probs = c(.25, .75), na.rm = TRUE)
  iqr <- IQR(x, na.rm = TRUE)
  lower <- qnt[1] - 1.5 * iqr
  upper <- qnt[2] + 1.5 * iqr
  sum(x < lower | x > upper, na.rm = TRUE)
}

# Load Datasets
train <- read.csv('train.csv')
test <- read.csv('test.csv')

# Initial Dataset Dimensions
dimensions <- dim(train)
cat("The dataset has", dimensions[1], "rows and", dimensions[2], "columns.\n")

# Summary Statistics
summary(train)

# Check for Missing Values
total_missing_values <- sum(is.na(train))
cat("Total missing values in the dataset:", total_missing_values, "\n")

# Outliers Count by Column
outliers_count <- sapply(train[, sapply(train, is.numeric)], find_outliers)
cat("Outliers count by column:\n")
print(outliers_count)

# Target Variable Distribution
target_column <- "churn" # Adjust based on your dataset
class_distribution <- table(train[[target_column]])
cat("Class Distribution:\n")
print(class_distribution)

# Handling Missing Values
train <- na.omit(train)
test <- na.omit(test)

#scaling 
data_subset <- train[, -ncol(train), drop = FALSE]
scaled_data <- scale(data_subset)
scaled_data_with_last_column <- cbind(scaled_data, train[, ncol(train), drop = FALSE])
colnames(scaled_data_with_last_column) <- c(colnames(scaled_data), colnames(train)[ncol(train)])
data <- scaled_data_with_last_column

#Balancing 
install.packages("ROSE")
library(ROSE)
train_data <- ROSE(churn ~ ., data = train, seed = 123)$data

# Splitting Data into Training and Validation Sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(train$churn, p = 0.8, list = FALSE)
train_set <- train[train_index, ]
val_set <- train[-train_index, ]

# Preparing Matrices for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_set[, -ncol(train_set)]), label = train_set$churn)
dval <- xgb.DMatrix(data = as.matrix(val_set[, -ncol(val_set)]), label = val_set$churn)

# Hyperparameter Tuning for XGBoost (simplified example)
best_auc <- 0
best_params <- NULL
best_nround <- NULL

params_grid <- expand.grid(
  eta = c(0.01, 0.1),
  max_depth = c(4, 6, 8),
  subsample = c(0.7, 0.8),
  colsample_bytree = c(0.7, 0.8),
  min_child_weight = c(1, 5),
  gamma = c(0, 0.1)
)

for (i in 1:nrow(params_grid)) {
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eta = params_grid$eta[i],
    gamma = params_grid$gamma[i],
    max_depth = params_grid$max_depth[i],
    subsample = params_grid$subsample[i],
    colsample_bytree = params_grid$colsample_bytree[i],
    min_child_weight = params_grid$min_child_weight[i],
    eval_metric = "auc"
  )
  
  cv.model <- xgb.cv(
    params = params, 
    data = dtrain, 
    nrounds = 1000,  # Increase the number of iterations
    nfold = 5, 
    showsd = TRUE, 
    stratified = TRUE, 
    print.every.n = 10, 
    early_stopping_rounds = 10, 
    maximize = TRUE
  )
  
  if (cv.model$evaluation_log$test_auc_mean[cv.model$best_iteration] > best_auc) {
    best_auc <- cv.model$evaluation_log$test_auc_mean[cv.model$best_iteration]
    best_params <- params
    best_nround <- cv.model$best_iteration
  }
}

# Training the XGBoost Model with the Best Parameters
model_fit_xgb <- xgb.train(params = best_params, data = dtrain, nrounds = best_nround)

# Making Predictions on the Validation Set for XGBoost
val_gbm_predictions <- as.numeric(predict(model_fit_xgb, newdata = dval))  # Ensure predictions are numeric
val_labels <- val_set$churn
gbm_auc_val <- roc(response = val_labels, predictor = val_gbm_predictions)$auc
cat("Validation AUC for XGBoost:", gbm_auc_val, "\n")

# Training Logistic Regression Model
model_fit_logit <- glm(churn ~ ., data = train_set, family = "binomial")

# Making Predictions on the Validation Set for Logistic Regression
val_logit_predictions <- predict(model_fit_logit, newdata = val_set, type = "response")
logit_auc_val <- roc(response = val_labels, predictor = val_logit_predictions)$auc
cat("Validation AUC for Logistic Regression:", logit_auc_val, "\n")

# Model Comparison
cat("Model Comparison:\n")
cat("XGBoost AUC:", gbm_auc_val, "\n")
cat("Logistic Regression AUC:", logit_auc_val, "\n")

# Preparing the Test Set for Predictions using model.matrix
test_matrix <- model.matrix(~ . -1, data = test)
dtest <- xgb.DMatrix(data = test_matrix)

# Generating Predictions for Test Set using XGBoost
test_preds <- predict(model_fit_xgb, dtest)

# Prepare Submission Data Frame
submission <- data.frame(id = test$id, churn = test_preds)
write.csv(submission, 'xgb_submission_7.csv', row.names = FALSE)