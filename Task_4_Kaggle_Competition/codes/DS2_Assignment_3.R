######################################################
#                                                    #
#   Attila Serfozo - Data Science 2 - Assignment 3   #
#                                                    #
######################################################


# Set up environment ------------------------------------------------------

path <- "D:/Egyetem/CEU/Winter_Term/Data_Science_2/Assignments/Assignment3"
setwd(path)

# Load packages and data

library(tidyverse)
library(caret)
library(glmnet)
library(pROC)
library(caTools)
library(keras)


my_seed <- 20210408

data <- read_csv("train.csv")
data_test <- read_csv("test.csv")

skimr::skim(data)

data <- mutate(data, is_popular = factor(is_popular, levels = 0:1, labels = c("NoPopular", "Popular")) )


# EDA ---------------------------------------------------------------------

# Quick check on all numeric features
p <- data %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~key, scales = "free") +
  geom_histogram(fill='cyan4', col="black", alpha=0.7)+
  theme_bw()+
  labs(x="Variable", y="Absolute frequency", title="Distribution of variables")
p


# Transformations ---------------------------------------------------------

# There are several skewed variables -> try log transformations to improve accuracy?

# Keyword variables
# data <- data %>% mutate(
#     logkw_avg_avg = ifelse(kw_avg_avg < 0 , 1, log(kw_avg_avg+1)),
#     logkw_avg_max = ifelse(kw_avg_max < 0 , 1, log(kw_avg_max+1)),
#     logkw_avg_min = ifelse(kw_avg_min < 0 , 1, log(kw_avg_min+1)),
#     logkw_max_avg = ifelse(kw_max_avg < 0 , 1, log(kw_max_avg+1)),
#     logkw_max_max = ifelse(kw_max_max < 0 , 1, log(kw_max_max+1)),
#     logkw_max_min = ifelse(kw_max_min < 0 , 1, log(kw_max_min+1)),
#     logkw_min_avg = ifelse(kw_min_avg < 0 , 1, log(kw_min_avg+1)),
#     logkw_min_max = ifelse(kw_min_max < 0 , 1, log(kw_min_max+1)),
#     logkw_min_min = ifelse(kw_min_min < 0 , 1, log(kw_min_min+1)),
#     )
# data <- data %>% dplyr::select(-c(kw_avg_avg,kw_avg_max,kw_avg_min,kw_max_avg,kw_max_max,kw_max_min,kw_min_avg,kw_min_max,kw_min_min))
# 
# Counting variables
# data <- data %>% mutate(
#     logn_non_stop_unique_tokens = ifelse(n_non_stop_unique_tokens < 0 , 1, log(n_non_stop_unique_tokens+1)),
#     logn_non_stop_words = ifelse(n_non_stop_words < 0 , 1, log(n_non_stop_words)),
#     logn_tokens_content = ifelse(n_tokens_content < 0 , 1, log(n_tokens_content+1)),
#     logn_unique_tokens = ifelse(n_unique_tokens < 0 , 1, log(n_unique_tokens+1)),
#     lognum_hrefs = ifelse(num_hrefs < 0 , 1, log(num_hrefs+1)),
#     lognum_imgs = ifelse(num_imgs < 0 , 1, log(num_imgs+1)),
#     lognum_self_hrefs = ifelse(num_self_hrefs < 0 , 1, log(num_self_hrefs+1)),
#     lognum_videos = ifelse(num_videos < 0 , 1, log(num_videos+1)),
#     logself_reference_avg_sharess = ifelse(self_reference_avg_sharess < 0 , 1, log(self_reference_avg_sharess+1)),
#     logself_reference_min_shares = ifelse(self_reference_min_shares < 0 , 1, log(self_reference_min_shares+1)),
#     logself_reference_max_shares = ifelse(self_reference_max_shares < 0 , 1, log(self_reference_max_shares+1)),
#     )
# data <- data %>% dplyr::select(-c(n_non_stop_unique_tokens,n_non_stop_words,n_tokens_content,n_unique_tokens,num_hrefs,num_imgs,num_self_hrefs,num_videos,self_reference_avg_sharess,self_reference_min_shares,self_reference_max_shares))



# 5-fold CV ---------------------------------------------------------------

set.seed(my_seed)
# Separate holdout set
train_indices <- as.integer(createDataPartition(data$is_popular, p = 0.75, list = FALSE))
data_train <- data[train_indices, ]
data_valid <- data[-train_indices, ]

# 5-fold CV
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE,
  verboseIter = FALSE
)


# Models ------------------------------------------------------------------

# Base model - linear

# set.seed(my_seed)
# linear_model <- train(
#   is_popular ~ . -is_popular,
#   data = data_train,
#   method = "glm",
#   trControl = train_control
# )
# saveRDS(linear_model, paste0(path,"/output/linear_model.rds"))
linear_model <- readRDS(paste0(path,"/output/linear_model.rds"))
linear_model

lin_roc <- roc(
  predictor=predict(linear_model, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
lin_roc

# Ridge

ridge_tune_grid <- expand.grid(
  "alpha" = c(0),
  "lambda" = seq(0.05, 0.5, by = 0.025)
)

# set.seed(my_seed)
# ridge_fit <- train(
#   is_popular ~ . -is_popular,
#   data = data_train,
#   method = "glmnet",
#   metric = "ROC",
#   preProcess = c("center", "scale"),
#   tuneGrid = ridge_tune_grid,
#   trControl = train_control
# )
# saveRDS(ridge_fit, paste0(path,"/output/ridge_fit.rds"))
ridge_fit <- readRDS(paste0(path,"/output/ridge_fit.rds"))
ridge_fit

ridge_roc <- roc(
  predictor=predict(ridge_fit, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
ridge_roc

# LASSO

tenpowers <- 10^seq(-1, -5, by = -1)

lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = c(tenpowers, tenpowers / 2) 
)

# set.seed(my_seed)
# lasso_fit <- train(
#   is_popular ~ . -is_popular,
#   data = data_train,
#   method = "glmnet",
#   metric = "ROC",
#   preProcess = c("center", "scale"),
#   tuneGrid = lasso_tune_grid,
#   trControl = train_control
# )
# saveRDS(lasso_fit, paste0(path,"/output/lasso_fit.rds"))
lasso_fit <- readRDS(paste0(path,"/output/lasso_fit.rds"))
lasso_fit

lasso_roc <- roc(
  predictor=predict(lasso_fit, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
lasso_roc

# Elastic Net

enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = union(lasso_tune_grid[["lambda"]], ridge_tune_grid[["lambda"]])
)

# set.seed(my_seed)
# enet_fit <- train(
#   is_popular ~ . -is_popular,
#   data = data_train,
#   method = "glmnet",
#   metric = "ROC",
#   preProcess = c("center", "scale"),
#   tuneGrid = enet_tune_grid,
#   trControl = train_control
# )
# saveRDS(enet_fit, paste0(path,"/output/enet_fit.rds"))
enet_fit <- readRDS(paste0(path,"/output/enet_fit.rds"))
enet_fit

enet_roc <- roc(
  predictor=predict(enet_fit, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
enet_roc

# Results
models <-
  list("Linear"= round(max(lin_roc$auc),4),
       "Ridge" = round(max(ridge_roc$auc),4),
       "LASSO" = round(max(lasso_roc$auc),4),
       "Elastic Net" = round(max(enet_roc$auc),4))

result_1 <- unlist(models) %>% as.data.frame() %>%
  rename("CV AUC" = ".")
result_1


# Random Forest

rf_tune_grid <- expand.grid(.mtry = c(2,4,6,8,10),
                            .splitrule = "gini",
                            .min.node.size = c(5,25,40,50,75,100) )

# set.seed(my_seed)
# rf_model <- train(is_popular ~ .,
#                   data = data_train,
#                   method = "ranger",
#                   metric = "ROC",
#                   trControl = train_control,
#                   tuneGrid= rf_tune_grid,
#                   importance = "impurity")
#saveRDS(rf_model, paste0(path,"/output/rf_model.rds"))
rf_model <- readRDS(paste0(path,"/output/rf_model.rds"))
rf_model

# Best tune parameter is mtry 2 and min node size 50
rf_model$bestTune
plot(rf_model)

rf_roc <- roc(
  predictor=predict(rf_model, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
rf_roc

# Gradient Boosting Machine

gbm_tune_grid <- expand.grid(interaction.depth=c(3,5,7),      
                             n.trees=c(100,300,500),             
                             shrinkage = c(0.01,0.05,0.1),
                             n.minobsinnode = c(10,30,50,75) )

# set.seed(my_seed)
# gbm_model <- train(is_popular ~ .,
#                    data = data_train, 
#                    method = "gbm",
#                    verbose = FALSE,
#                    metric = "ROC",
#                    trControl=train_control,
#                    tuneGrid = gbm_tune_grid)
#saveRDS(gbm_model, paste0(path,"/output/gbm_model.rds"))
gbm_model <- readRDS(paste0(path,"/output/gbm_model.rds"))
gbm_model

# Best tune parameter is 300 trees, 5 depth, 0.05 shrinkage and 50 min node size - 0.7179140
gbm_model$bestTune
plot(gbm_model)

gbm_roc <- roc(
  predictor=predict(gbm_model, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
gbm_roc

# XGBoost

xgb_grid <- expand.grid(nrounds = c(100, 350, 500,750),   
                        max_depth = c(3,5,7),      
                        eta = c(0.03,0.05, 0.06),      
                        gamma = c(0.01),    
                        colsample_bytree = c(0.5,0.75), 
                        subsample = c(0.75), 
                        min_child_weight = c(0,0.2,0.5))   

# set.seed(my_seed)
# xgb_model <- train(
#   is_popular ~ .,
#   data = data_train,
#   method = "xgbTree",
#   metric = "ROC",
#   tuneGrid = xgb_grid,
#   trControl = train_control
# )
#saveRDS(xgb_model, paste0(path,"/output/xgb_model.rds"))
xgb_model <- readRDS(paste0(path,"/output/xgb_model.rds"))
xgb_model

# Best tune parameter is nr of trained trees 350, max depth of 5, learning rate 0.03, min child weight 0.2
xgb_model$bestTune
plot(xgb_model)

xgb_roc <- roc(
  predictor=predict(xgb_model, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
xgb_roc


# Results -----------------------------------------------------------------

models_2 <-
  list("Linear"= round(max(lin_roc$auc),4),
       "Random Forest"= round(max(rf_roc$auc),4),
       "GBM" = round(max(gbm_roc$auc),4),
       "XGBoost" = round(max(xgb_roc$auc),4))

result_2 <- unlist(models_2) %>% as.data.frame() %>%
  rename("CV AUC" = ".")
result_2


# Create test prediction --------------------------------------------------

xgb_test_pred <- predict(xgb_model, data_test, type="prob")

xgb_test_output <- data.frame(article_id = data_test$article_id, score = predict(xgb_model, data_test, type="prob")[2])
xgb_test_output <- rename(xgb_test_output, score = Popular)
#write_csv(xgb_test_output,paste0(path,"/output/xgb_test_prediction.csv"))                              


# Train best model on the full train data ---------------------------------

xgb_grid_full_data <- expand.grid(nrounds = c(350),   
                        max_depth = c(5),      
                        eta = c(0.03),      
                        gamma = c(0.01),    
                        colsample_bytree = c(0.75), 
                        subsample = c(0.75), 
                        min_child_weight = c(0.2))   

# set.seed(my_seed)
# xgb_model_full_data <- train(
#   is_popular ~ .,
#   data = data,
#   method = "xgbTree",
#   metric = "ROC",
#   tuneGrid = xgb_grid_full_data,
#   trControl = train_control
# )
#saveRDS(xgb_model_full_data, paste0(path,"/output/xgb_model_full_data.rds"))
xgb_model_full_data <- readRDS(paste0(path,"/output/xgb_model_full_data.rds"))
xgb_model_full_data

xgb_full_test_output <- data.frame(article_id = data_test$article_id, score = predict(xgb_model_full_data, data_test, type="prob")[2])
xgb_full_test_output <- rename(xgb_full_test_output, score = Popular)
#write_csv(xgb_full_test_output,paste0(path,"/output/xgb_full_test_prediction.csv"))


# Neural network ----------------------------------------------------------

# Re-import data

data <- read_csv("train.csv")
data_test <- read_csv("test.csv")

set.seed(my_seed)
# Separate holdout set
train_indices <- as.integer(createDataPartition(data$is_popular, p = 0.75, list = FALSE))
data_train <- data[train_indices, ]
data_valid <- data[-train_indices, ]

# Normalize the data

x_train <- select(data_train, -is_popular)
x_valid <- select(data_valid, -is_popular)

y_train <- to_categorical(data_train$is_popular)
y_valid <- to_categorical(data_valid$is_popular)

for (i in colnames(x_train)) {
    col <-x_train[,c(i)]
    col1 <- col + abs(min(col))
    col2 <- col1/max(col1)
    x_train[,c(i)] <- col2
}

for (i in colnames(x_valid)) {
  col <-x_valid[,c(i)]
  col1 <- col + abs(min(col))
  col2 <- col1/max(col1)
  x_valid[,c(i)] <- col2
}
x_test <- data_test
for (i in colnames(x_test)) {
  col <-x_test[,c(i)]
  col1 <- col + abs(min(col))
  col2 <- col1/max(col1)
  x_test[,c(i)] <- col2
}
x_train <- as.matrix(x_train) 
x_valid <- as.matrix(x_valid)
x_test <- as.matrix(x_test)


# Model 1 base model

nn_model_1 <- keras_model_sequential()
nn_model_1 %>%
  layer_dense(units = 16, activation = 'relu', input_shape = ncol(x_train)) %>%  
  layer_dropout(rate = 0.3) %>%   
  layer_dense(units = 2, activation = 'softmax')                           

compile(
  nn_model_1,
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# history_1 <- fit(
#   nn_model_1, x_train, y_train,
#   epochs = 30, batch_size = 128,
#   validation_data = list(x_valid, y_valid)
# )

# save_model_hdf5(nn_model_1, paste0(path,"/output/nn_model_1.h5"))
nn_model_1 <- load_model_hdf5(paste0(path,"/output/nn_model_1.h5"))
# saveRDS(history_1, paste0(path,"/output/nn_model_1.rds"))
history_1 <- readRDS(paste0(path,"/output/nn_model_1.rds"))
# history_1

nn_model_output <- predict_proba(nn_model_1,x_valid)[,2]
nn_roc_1 <- pROC::roc(y_valid[,2],nn_model_output)
# nn_roc_1

plot(history_1) + labs(title = "Model 1 - base model with 16 nods in hidden layer")

# NN model with increased node number in layer

nn_model_2 <- keras_model_sequential()
nn_model_2 %>%
  layer_dense(units = 32, activation = 'relu', input_shape = ncol(x_train)) %>%  
  layer_dropout(rate = 0.3) %>%   
  layer_dense(units = 2, activation = 'softmax')                           

compile(
  nn_model_2,
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# history_2 <- fit(
#   nn_model_2, x_train, y_train,
#   epochs = 30, batch_size = 128,
#   validation_data = list(x_valid, y_valid)
# )

# save_model_hdf5(nn_model_2, paste0(path,"/output/nn_model_2.h5"))
nn_model_2 <- load_model_hdf5(paste0(path,"/output/nn_model_2.h5"))
# saveRDS(history_2, paste0(path,"/output/nn_model_2.rds"))
history_2 <- readRDS(paste0(path,"/output/nn_model_2.rds"))
# history_2

nn_model_output <- predict_proba(nn_model_2,x_valid)[,2]
nn_roc_2 <- pROC::roc(y_valid[,2],nn_model_output)
# nn_roc_2

plot(history_2) + labs(title = "Model 2 - 32 nods in hidden layer")

# NN model with one more layer

nn_model_3 <- keras_model_sequential()
nn_model_3 %>%
  layer_dense(units = 32, activation = 'relu', input_shape = ncol(x_train)) %>%  
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 16, activation = 'relu', input_shape = ncol(x_train)) %>%  
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 2, activation = 'softmax')                           

compile(
  nn_model_3,
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# history_3 <- fit(
#   nn_model_3, x_train, y_train,
#   epochs = 30, batch_size = 128,
#   validation_data = list(x_valid, y_valid)
# )

# save_model_hdf5(nn_model_3, paste0(path,"/output/nn_model_3.h5"))
nn_model_3 <- load_model_hdf5(paste0(path,"/output/nn_model_3.h5"))
# saveRDS(history_3, paste0(path,"/output/nn_model_3.rds"))
history_3 <- readRDS(paste0(path,"/output/nn_model_3.rds"))
# history_3

nn_model_output <- predict_proba(nn_model_3,x_valid)[,2]
nn_roc_3 <- pROC::roc(y_valid[,2],nn_model_output)
# nn_roc_3

plot(history_3) + labs(title = "Model 3 - add another layer")

# Prediction on test set
nn_model_output <- data.frame(article_id = data_test$article_id,score = predict_proba(nn_model_2,x_test)[,1])
# write_csv(nn_model_output,paste0(path,"/output/nn_model_prediction.csv"))

# Neural net model results
models_3 <-
  list("NN model 1"= round(max(nn_roc_1$auc),4),
       "NN model 2"= round(max(nn_roc_2$auc),4),
       "NN model 3" = round(max(nn_roc_3$auc),4))

result_3 <- unlist(models_3) %>% as.data.frame() %>%
  rename("AUC" = ".")
result_3


# Final model results -----------------------------------------------------

models_4 <-
  list("Linear"= round(max(lin_roc$auc),4),
       "Random Forest"= round(max(rf_roc$auc),4),
       "GBM" = round(max(gbm_roc$auc),4),
       "XGBoost" = round(max(xgb_roc$auc),4),
       "Neural Net" = round(max(nn_roc_2$auc),4))

result_4 <- unlist(models_4) %>% as.data.frame() %>%
  rename("AUC" = ".")
# result_4


