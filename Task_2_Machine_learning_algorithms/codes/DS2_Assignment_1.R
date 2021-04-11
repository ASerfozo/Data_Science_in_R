######################################################
#                                                    #
#   Attila Serfozo - Data Science 2 - Assignment 1   #
#                                                    #
######################################################

# Import packages
library(tidyverse)
library(caret)

# 1. Tree ensemble models ----------------------------------------------------

data <- as_tibble(ISLR::OJ)
str(data)
skimr::skim(data)

# a. Create a training data of 75% and keep 25% of the data as a test set. 
# Train a decision tree as a benchmark model. Plot the final model and interpret 
# the result (using rpart and rpart.plot is an easier option).

set.seed(20210318)
# Separate holdout set
train_indices <- as.integer(createDataPartition(data$Purchase, p = 0.75, list = FALSE))
data_train <- data[train_indices, ]
data_holdout <- data[-train_indices, ]

# 5-fold CV
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE,
  verboseIter = FALSE
)

library(rpart)
library(rpart.plot)

# First model - CART
set.seed(20210318)
benchmark_model <- train(
  Purchase ~ .,
  data = data_train,
  method = "rpart",
  metric = "ROC",
  trControl = train_control,
  tuneGrid= expand.grid(cp = 0.0005)
  )
benchmark_model$results
rpart.plot(benchmark_model$finalModel, tweak = 1.5)

# b. Investigate tree ensemble models: random forest, gradient boosting machine, XGBoost. 
# Try various tuning parameter combinations and select the best model using cross-validation.

# Random Forest
rf_tune_grid <- expand.grid(
  .mtry = 2:10,
  .splitrule = "gini",
  .min.node.size = seq(5,50,by = 5) 
)

set.seed(20210318)
rf_model <- train(
  Purchase ~ .,
  data = data_train,
  method = "ranger",
  metric = "ROC",
  trControl = train_control,
  tuneGrid= rf_tune_grid,
  importance = "impurity"
  )

# Best tune parameter is mtry 8 and min node size 40
rf_model$bestTune
plot(rf_model)


# Gradient Boosting Machine
gbm_tune_grid <- expand.grid(
  interaction.depth=c(1:5),                    # depth of trees
  n.trees=c(100, 200,300),                     # number of trees
  shrinkage = c(.0005, .001, 0.01, 0.05, 0.1), # learning rate
  n.minobsinnode = seq(10,20,by = 5)           # minimum node size
)

set.seed(20210318)
gbm_model <- train(Purchase ~ .,
                   data = data_train, 
                   method = "gbm",
                   verbose = FALSE,
                   metric = "ROC",
                   trControl=train_control,
                   tuneGrid = gbm_tune_grid)
# Best tune parameter is 100 trees, 5 depth, 0.05 shrinkage and 10 min node size
gbm_model$bestTune
plot(gbm_model)


# XGBoost

xgb_grid <-  expand.grid(
  nrounds=c(350, 500),                   # Number of trees, default: 100
  max_depth = c(2,3,4),                  # Maximum tree depth, default: 6
  eta = c(0.03,0.05, 0.06),              # Learning rate, default: 0.3
  gamma = c(0.01),                       # Used for tuning of Regularization, default: 0
  colsample_bytree = seq(0.3, 0.5, 0.6), # Column sampling, default: 1
  subsample = c(0.75),                   # Row sampling, default: 1
  min_child_weight = c(0))               # Minimum leaf weight, default: 1

set.seed(20210318)
xgb_model <- train(
  Purchase ~ .,
  data = data_train,
  method = "xgbTree",
  metric = "ROC",
  tuneGrid = xgb_grid,
  trControl = train_control
)

xgb_model$bestTune
plot(xgb_model)

# c. Compare the performance of the different models (if you use caret you should 
# consider using the resamples function). Make sure to set the same seed before model 
# training for all 3 models so that your cross validation samples are the same. Is any 
# of these giving significantly different predictive power than the others?

final_models <-
  list("CART" = benchmark_model,
       "Random_Forest" = rf_model,
       "GBM" = gbm_model,
       "XGBoost" = xgb_model)

results <- resamples(final_models) %>% summary()
results

results <- imap(final_models, ~{
  mean(results$values[[paste0(.y,"~ROC")]])
}) %>% unlist() %>% as.data.frame() %>%
  rename("CV ROC" = ".")
results

# d. Choose the best model and plot ROC curve for the best model on the test set. 
# Calculate and interpret AUC.

# ROC on test set

library(pROC)
gbm_pred <- predict(gbm_model, data_holdout, type="prob")
# library(caTools)
# colAUC(gbm_pred, data_holdout$Purchase, plotROC = TRUE)


data_holdout[,"best_model_pred"] <- gbm_pred[,"CH"]

roc_obj_holdout <- roc(data_holdout$Purchase, data_holdout$best_model_pred)

# ROC plot creator made by Gábor Békés 
library(viridis)
createRocPlot <- function(r, plot_name) {
  all_coords <- coords(r, x="all", ret="all", transpose = FALSE)
  
  roc_plot <- ggplot(data = all_coords, aes(x = fpr, y = tpr)) +
    geom_line(color='blue', size = 0.7) +
    geom_area(aes(fill = 'red', alpha=0.4), alpha = 0.3, position = 'identity', color = 'blue') +
    scale_fill_viridis(discrete = TRUE, begin=0.6, alpha=0.5, guide = FALSE) +
    xlab("False Positive Rate (1-Specifity)") +
    ylab("True Positive Rate (Sensitivity)") +
    geom_abline(intercept = 0, slope = 1,  linetype = "dotted", col = "black") +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .1), expand = c(0, 0.01)) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1), expand = c(0.01, 0)) + 
    theme_bw()
  
  roc_plot
}

createRocPlot(roc_obj_holdout, "ROC curve for best model (GBM)")


# Calculate models AUC
CV_AUC_folds <- list()

for (model_name in names(final_models)) {
  
  auc <- list()
  model <- final_models[[model_name]]
  for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
    cv_fold <-
      model$pred %>%
      filter(Resample == fold)
    
    roc_obj <- roc(cv_fold$obs, cv_fold$CH)
    auc[[fold]] <- as.numeric(roc_obj$auc)
  }
  
  CV_AUC_folds[[model_name]] <- data.frame("Resample" = names(auc),
                                           "AUC" = unlist(auc))
}

CV_AUC <- list()
for (model_name in names(final_models)) {
  CV_AUC[[model_name]] <- mean(CV_AUC_folds[[model_name]]$AUC)
}

models_AUC <- CV_AUC %>% rbind() 


# e. Inspect variable importance plots for the 3 models. Are similar variables found to 
# be the most important for the 3 models?
library(gbm)

plot(varImp(rf_model))

plot(varImp(gbm_model))

plot(varImp(xgb_model))


# 2. Variable importance profiles --------------------------------------------

data <- as_tibble(ISLR::Hitters) %>%
  drop_na(Salary) %>%
  mutate(log_salary = log(Salary), Salary = NULL)

# a. Train two random forest models: one with sampling 2 variables randomly for 
# each split and one with 10 (use the whole dataset and don’t do cross-validation). 
# Inspect variable importance profiles. What do you see in terms of how important the 
# first few variables are relative to each other?

# Random Forest
tune_grid <- expand.grid(
  .mtry = 2,
  .splitrule = "variance",
  .min.node.size = c(5,10) 
)

set.seed(20210318)
rf_model_1 <- train(
  log_salary ~ .,
  data = data,
  method = "ranger",
  tuneGrid= tune_grid,
  importance = "impurity"   # need to be added to see varimp
)
rf_model_1


tune_grid <- expand.grid(
  .mtry = 10,
  .splitrule = "variance",
  .min.node.size = c(5,10) 
)

set.seed(20210318)
rf_model_2 <- train(
  log_salary ~ .,
  data = data,
  method = "ranger",
  tuneGrid= tune_grid,
  importance = "impurity"
)
rf_model_2

plot(varImp(rf_model_1), main = "Random Forest sampling 2 variables - Variable Importance")
plot(varImp(rf_model_2), main = "Random Forest sampling 10 variables - Variable Importance")

# b. One of them is more extreme in terms of how the most important and the next 
# ones relate to each other. Give an intuitive explanation how mtry/mtries relates 
# to relative importance of variables in random forest models.

# c. In the same vein, estimate two gbm models with varying rate of sampling for 
# each tree (use 0.1 and 1 for the parameter bag.fraction/sample_rate). Hold all 
# the other parameters fixed: grow 500 trees with maximum depths of 5, applying a 
# learning rate of 0.1. Compare variable importance plots for the two models. 
# Could you explain why is one variable importance profile more extreme than the other?

tune_grid <- expand.grid(
  interaction.depth=5,
  n.trees=500,
  shrinkage = 0.1,
  n.minobsinnode = 5
)

set.seed(20210318)
gbm_model_1 <- train(log_salary ~ .,
                     data = data, 
                     method = "gbm",
                     bag.fraction = 0.1,
                     verbose = FALSE,
                     tuneGrid = tune_grid)

set.seed(20210318)
gbm_model_2 <- train(log_salary ~ .,
                     data = data, 
                     method = "gbm",
                     bag.fraction = 1,
                     verbose = FALSE,
                     tuneGrid = tune_grid)

plot(varImp(gbm_model_1), main = "GBM with 0.1 rate of sampling - Variable Importance")
plot(varImp(gbm_model_2), main = "GBM with 1 rate of sampling - Variable Importance")


# 3. Stacking ----------------------------------------------------------------

library(tidyverse)

setwd("D:/Egyetem/CEU/Winter_Term/Data_Science_2/Assignments/Assignment1")
# Get Data
data <- read_csv("KaggleV2-May-2016.csv")

# some data cleaning
data <- select(data, -one_of(c("PatientId", "AppointmentID", "Neighbourhood"))) %>%
  janitor::clean_names()

# for binary prediction, the target variable must be a factor + generate new variables
data <- mutate(
  data,
  no_show = factor(no_show, levels = c("Yes", "No")),
  handcap = ifelse(handcap > 0, 1, 0),
  across(c(gender, scholarship, hipertension, alcoholism, handcap), factor),
  hours_since_scheduled = as.numeric(appointment_day - scheduled_day)
)

# clean up a little bit
data <- filter(data, between(age, 0, 95), hours_since_scheduled >= 0) %>%
  select(-one_of(c("scheduled_day", "appointment_day", "sms_received")))

# a. Create train / validation / test sets, cutting the data into 5% - 45% - 50% parts.
h2o.shutdown()
library(h2o)
h2o.init()
h2o.no_progress()
h2o.init(max_mem_size = "4g")

my_seed <- 20210318

data <- as.h2o(data)

data_split <- h2o.splitFrame(data, ratios = c(0.05, 0.5), seed = my_seed)

data_train <- data_split[[1]]
data_validation <- data_split[[2]]
data_test <- data_split[[3]]

# b. Train a benchmark model of your choice (such as random forest, gbm or glm) 
# and evaluate it on the validation set.

y <- "no_show"
X <- setdiff(names(data), y)

# benchmark model

simple_lm <- h2o.glm(
  X, y,
  training_frame = data_train,
  model_id = "logit",
  lambda = 0,
  nfolds = 5,
  seed = my_seed
)
simple_lm

# AUC (train / cv)
h2o.auc(simple_lm, train = TRUE, xval = TRUE)
# AUC on Validation set
h2o.auc(h2o.performance(simple_lm, data_validation))


# c. Build at least 3 models of different families using cross validation, keeping 
# cross validated predictions. You might also try deeplearning.

# RANDOM FOREST

rf_params <- list(
  ntrees = c(100, 300, 500),    # Number of trees
  mtries = c(2, 4),             # Number of variables
  max_depth = c(5, 10)          # Depth of tree
)

rf_grid <- h2o.grid("randomForest", x = X, y = y,
                    training_frame = data_train,
                    grid_id = "rf",
                    nfolds = 5,
                    seed = my_seed,
                    hyper_params = rf_params,
                    keep_cross_validation_predictions = TRUE
)

# Select the best model based on AUC
rf_model <- h2o.getModel(rf_grid@model_ids[[1]])
h2o.auc(rf_model)

# GLM

glm_params <- list(alpha = c(0, .25, .5, .75, 1))

# build grid search with previously selected hyperparameters
glm_grid <- h2o.grid("glm", x = X, y = y,
                     grid_id = "glm",
                     training_frame = data_train,
                     lambda_search = TRUE,
                     nfolds = 5,
                     seed = my_seed,
                     hyper_params = glm_params, 
                     keep_cross_validation_predictions = TRUE)

# Select the best model based on AUC
glm_model <- h2o.getModel(glm_grid@model_ids[[1]])
h2o.auc(glm_model)

# GBM

gbm_params <- list(
  learn_rate = c(0.01, 0.05, 0.1, 0.3),
  ntrees = c(100, 300, 500),
  max_depth = c(2, 5)
)

gbm_grid <- h2o.grid("gbm", x = X, y = y,
                      grid_id = "gbm",
                      training_frame = data_train,
                      nfolds = 5,
                      seed = my_seed,
                      hyper_params = gbm_params, 
                     keep_cross_validation_predictions = TRUE
)

# Select the best model based on AUC
gbm_model <- h2o.getModel(gbm_grid@model_ids[[1]])
h2o.auc(gbm_model)


# DEEPLEARNING DOES NOT WORK WITH H2O ON MY COMPUTER
# deeplearning_model <- h2o.deeplearning(
#   X, y,
#   training_frame = data_train,
#   seed = my_seed,
#   nfolds = 5,
#   keep_cross_validation_predictions = TRUE
# )


results3c <-
  list("Random_Forest" = h2o.auc(rf_model),
       "GLM" = h2o.auc(glm_model),
       "GBM" = h2o.auc(gbm_model))

results3c <- results3c %>% unlist() %>% as.data.frame() %>% rename("CV AUC" = ".")
results3c

# d. Evaluate validation set performance of each model.

rf_valid <- h2o.performance(rf_model, data_validation)
glm_valid <- h2o.performance(glm_model, data_validation)
gbm_valid <- h2o.performance(gbm_model, data_validation)

results3d <-
  list("Random_Forest" = h2o.auc(rf_valid),
       "GLM" = h2o.auc(glm_valid),
       "GBM" = h2o.auc(gbm_valid))

results3d <- results3d %>% unlist() %>% as.data.frame() %>% rename("CV AUC" = ".")
results3d




# e. How large are the correlations of predicted scores of the validation set 
# produced by the base learners?
 
models <- list(rf_model, glm_model, gbm_model)

h2o.model_correlation_heatmap(models, data_validation)
h2o.varimp_heatmap(models)

# f. Create a stacked ensemble model from the base learners.
 
ensemble_model <- h2o.stackedEnsemble(
  X, y,
  training_frame = data_train,
  base_models = models,
  keep_levelone_frame = TRUE
)
ensemble_model

# g. Evaluate ensembles on validation set. Did it improve prediction?
   
map_df(
  c(models, ensemble_model),
  ~{tibble(model = .@model_id, auc = h2o.auc(h2o.performance(., newdata = data_validation)))}
)

# h. Evaluate the best performing model on the test set. How does performance 
# compare to that of the validation set?

map_df(
  c(ensemble_model),
  ~{tibble(model = .@model_id, auc = h2o.auc(h2o.performance(., newdata = data_test)))}
)
