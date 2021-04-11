################################
##        Serfőző Attila      ##
##        Data Science 1      ##
##          Assignment        ##
################################

# Set up environment ------------------------------------------------------

rm(list=ls())


# Import libraries
library(tidyverse)
library(datasets)
library(MASS)
library(ISLR)
library(caret)
library(Hmisc)
library(GGally)
library(dplyr)

data <- readRDS(url('http://www.jaredlander.com/data/manhattan_Train.rds')) %>%
  mutate(logTotalValue = log(TotalValue)) %>%
  drop_na()


# 1. Supervised learning with penalized models and PCA --------------------

### a. Do a short exploration of data and find possible predictors of the target variable.

# Zone 2-3-4 variables have too many missing values and therefore have zero variances
describe(data$ZoneDist1)
describe(data$ZoneDist2)
describe(data$ZoneDist3)
describe(data$ZoneDist4)

data <- data %>% dplyr::select(-c(ZoneDist2, ZoneDist3, ZoneDist4))

describe(data$TotalValue)

# Quick check on all numeric features
data %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~key, scales = "free") +
  geom_histogram(fill='cyan4', col="black", alpha=0.7)+
  theme_bw()+
  labs(x="Variable", y="Absolute frequency", title="Distribution of variables")


# Check level property values
# Skewed with a long right tail
plot_value <- ggplot(data = data, aes (x = TotalValue/1000000)) +
  geom_histogram(color = "black", fill = "cyan4")+
  labs(title="Distribution of property values", x = "Property value (in million USD)", y = "Frequency") +
  theme_bw() 
plot_value

# Check log property values
# It takes a closer to normal distribution
plot_lnvalue <- ggplot(data = data, aes (x = logTotalValue)) +
  geom_histogram(color = "black", fill = "cyan4")+
  labs(title="Distribution of log property values", x = "Log Property value (in million USD)", y = "Frequency") +
  theme_bw() 
plot_lnvalue


# Create log of skewed variables: 
data <- data %>% mutate(logBldgArea = ifelse(BldgArea == 0 , 0, log(BldgArea)), 
                        logBldgDepth = ifelse(BldgDepth == 0 , 0, log(BldgDepth)), 
                        logBldgFront = ifelse(BldgFront == 0 , 0, log(BldgFront)), 
                        logBuiltFAR = ifelse(BuiltFAR == 0 , 0, log(BuiltFAR)), 
                        logComArea = ifelse(ComArea == 0 , 0, log(ComArea)), 
                        logFactryArea = ifelse(FactryArea == 0 , 0, log(FactryArea)), 
                        logGarageArea = ifelse(GarageArea == 0 , 0, log(GarageArea)), 
                        logLotArea = ifelse(LotArea == 0 , 0, log(LotArea)), 
                        logLotDepth = ifelse(LotDepth == 0 , 0, log(LotDepth)), 
                        logLotFront = ifelse(LotFront == 0 , 0, log(LotFront)), 
                        logNumBldgs = ifelse(NumBldgs == 0 , 0, log(NumBldgs)), 
                        logNumFloors = ifelse(NumFloors == 0 , 0, log(NumFloors)), 
                        logOfficeArea = ifelse(OfficeArea == 0 , 0, log(OfficeArea)), 
                        logOtherArea = ifelse(OtherArea == 0 , 0, log(OtherArea)),
                        logResArea = ifelse(ResArea == 0 , 0, log(ResArea)),
                        logRetailArea = ifelse(RetailArea == 0 , 0, log(RetailArea)),
                        logStrgeArea = ifelse(StrgeArea == 0 , 0, log(StrgeArea)),
                        logUnitsRes = ifelse(UnitsRes == 0 , 0, log(UnitsRes)),
                        logUnitsTotal = ifelse(UnitsTotal == 0 , 0, log(UnitsTotal)),)

# indicate with a flag is log value is replaced with 0 in case of -Inf 
data <- data %>% mutate(flag_logBldgArea = factor(ifelse(BldgArea == 0 , 1, 0)),
                        flag_logBldgDepth = factor(ifelse(BldgDepth == 0 , 1, 0)),
                        flag_logBldgFront = factor(ifelse(BldgFront == 0 , 1, 0)),
                        flag_logBuiltFAR = factor(ifelse(BuiltFAR == 0 , 1, 0)),
                        flag_logComArea = factor(ifelse(ComArea == 0 , 1, 0)),
                        flag_logFactryArea = factor(ifelse(FactryArea == 0 , 1, 0)),
                        flag_logGarageArea = factor(ifelse(GarageArea == 0 , 1, 0)),
                        flag_logLotArea = factor(ifelse(LotArea == 0 , 1, 0)),
                        flag_logLotDepth = factor(ifelse(LotDepth == 0 , 1, 0)),
                        flag_logLotFront = factor(ifelse(LotFront == 0 , 1, 0)),
                        flag_logNumBldgs = factor(ifelse(NumBldgs == 0 , 1, 0)),
                        flag_logNumFloors = factor(ifelse(NumFloors == 0 , 1, 0)),
                        flag_logOfficeArea = factor(ifelse(OfficeArea == 0 , 1, 0)),
                        flag_logOtherArea = factor(ifelse(OtherArea == 0 , 1, 0)),
                        flag_logResArea = factor(ifelse(ResArea == 0 , 1, 0)),
                        flag_logRetailArea = factor(ifelse(RetailArea == 0 , 1, 0)),
                        flag_logStrgeArea = factor(ifelse(StrgeArea == 0 , 1, 0)),
                        flag_logUnitsRes = factor(ifelse(UnitsRes == 0 , 1, 0)),
                        flag_logUnitsTotal = factor(ifelse(UnitsTotal == 0 , 1, 0)))


data <- data %>% dplyr::select(-c(BldgArea, BldgDepth, BldgFront, BuiltFAR, ComArea, FactryArea, GarageArea, LotArea, LotDepth, LotFront, NumBldgs, NumFloors, OfficeArea, OtherArea, ResArea, RetailArea, StrgeArea, UnitsRes, UnitsTotal))

data <- data %>% dplyr::select(-logTotalValue, logTotalValue)

ggcorr(data)

key_predictors <- c("logTotalValue", "logUnitsTotal", "logNumFloors", "logLotFront", "logLotArea", "logComArea", "logBuiltFAR", "logBldgArea")

ggpairs(data, columns = c("logTotalValue", "logUnitsTotal", "logNumFloors", "logLotFront", "logLotArea"))


### b. Create a training and a test set, assigning 30% of observations to the training set.

set.seed(1234)
training_ratio <- 0.3
train_indices <- createDataPartition(
  y = data[["logTotalValue"]],
  times = 1,
  p = training_ratio,
  list = FALSE
) %>% as.vector()
data_train <- data[train_indices, ]
data_test <- data[-train_indices, ]
fit_control <- trainControl(method = "cv", number = 10) #, selectionFunction = "oneSE")

### c. Use a linear regression to predict logTotalValue and use 10-fold cross validation to assess the predictive power.

set.seed(1234)
linear_reg <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "lm",
  trControl = fit_control
)
linear_reg
# RMSE 0.551, R-squared 0.876, MAE 0.419

ggplot(linear_reg) +
  geom_line(color="cyan4") +
  geom_point(color="cyan4") +
  theme_bw()

### d. Use penalized linear models for the same task. Make sure to try LASSO, Ridge and Elastic Net models. Does the best model improve on the simple linear model?
  
# Ridge

ridge_tune_grid <- expand.grid(
  "alpha" = c(0),
  "lambda" = seq(0.05, 0.5, by = 0.025)
)

set.seed(1234)
ridge_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = ridge_tune_grid,
  trControl = fit_control
)
ridge_fit

ggplot(ridge_fit) +
  geom_line(color="cyan4") +
  geom_point(color="cyan4") +
  theme_bw()
# The optimal value based on RMSE is lambda 0.1

# LASSO

tenpowers <- 10^seq(-1, -5, by = -1)

lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = c(tenpowers, tenpowers / 2) 
)

set.seed(1234)
lasso_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = lasso_tune_grid,
  trControl = fit_control
)

lasso_fit

ggplot(lasso_fit) + scale_x_log10() +
  geom_line(color="cyan4") +
  geom_point(color="cyan4") +
  theme_bw()
# The optimal value based on RMSE is lambda 0.0001

# Elastic Net

enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = union(lasso_tune_grid[["lambda"]], ridge_tune_grid[["lambda"]])
)

set.seed(1234)
enet_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = enet_tune_grid,
  trControl = fit_control
)

enet_fit
# The optimal value based on RMSE is lambda 0.0001

ggplot(enet_fit) + scale_x_log10() +
  theme_bw()


models <-
  list("Linear"= linear_reg,
       "Ridge" = ridge_fit,
       "LASSO" = lasso_fit,
       "Elastic Net" = enet_fit)

results <- resamples(models) %>% summary()

result_2 <- imap(models, ~{
  mean(results$values[[paste0(.y,"~RMSE")]])
}) %>% unlist() %>% as.data.frame() %>%
  rename("CV RMSE" = ".")
result_2

### e. Which of the models you’ve trained is the “simplest one that is still good enough”? 
#(Hint: explore adding selectionFunction = "oneSE" to the trainControl in caret’s train. What is its effect?).

# still the simple linear is the best, what is more it even better than previously

### f. Now try to improve the linear model by using PCA for dimensionality reduction. 
#Center and scale your variables and use pcr to conduct a search for the optimal number of principal components. 
#Does PCA improve the fit over the simple linear model? (Hint: there are many factor variables. 
# Make sure to include large number of principal components such as 60 - 90 to your search as well.)

tune_grid <- data.frame(ncomp = 80:130)
set.seed(1234)
pcr_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "pcr",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")
)
pcr_fit
# Final value used for the model was ncomp = 124

### g. If you apply PCA prior to estimating penalized models via preProcess, does it help to achieve a better fit? 
#(Hint: also include "nzv" to preProcess to drop zero variance features). What is your intuition why this can be the case?

# Ridge

ridge_tune_grid <- expand.grid(
  "alpha" = c(0),
  "lambda" = seq(0.05, 0.5, by = 0.025)
)

set.seed(1234)
ridge_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale", "nzv", "pca"),
  tuneGrid = ridge_tune_grid,
  trControl = trainControl(method = "cv", number = 10, preProcOptions = list(pcaComp = 124))
)

# LASSO

lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = c(tenpowers, tenpowers / 2) 
)

set.seed(1234)
lasso_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale", "nzv", "pca"),
  tuneGrid = lasso_tune_grid,
  trControl = trainControl(method = "cv", number = 10, preProcOptions = list(pcaComp = 124))
)

# Elastic Net

enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = union(lasso_tune_grid[["lambda"]], ridge_tune_grid[["lambda"]])
)

set.seed(1234)
enet_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale", "nzv", "pca"),
  tuneGrid = enet_tune_grid,
  trControl = trainControl(method = "cv", number = 10, preProcOptions = list(pcaComp = 124))
)

models <-
  list("Linear"= linear_reg,
       "Ridge" = ridge_fit,
       "LASSO" = lasso_fit,
       "Elastic Net" = enet_fit)

results5 <- resamples(models) %>% summary()

result_6 <- imap(models, ~{
  mean(results5$values[[paste0(.y,"~RMSE")]])
}) %>% unlist() %>% as.data.frame() %>%
  rename("CV RMSE" = ".")
  
### h. Select the best model of those you’ve trained. Evaluate your preferred model on the test set.

# Still the simple linear model is the best

results7 <- map(models, ~{
  RMSE(predict(.x, newdata = data_test), data_test[["logTotalValue"]])
}) %>% unlist() %>% as.data.frame() %>%
  rename("Holdout RMSE" = ".")
results7
# Holdout RMSE is 0.552


# 2. Clustering on the USArrests dataset ----------------------------------

library(tidyverse)
library(caret)
library(skimr)
library(janitor)
library(factoextra)
library(NbClust)
library(knitr)
library(kableExtra)

data <- USArrests
GGally::ggpairs(data, title = "USArrests data Scatters, Densities & Correlations")

### a. Think about any data pre-processing steps you may/should want to do before applying clustering methods. 
# Are there any?

data %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~key, scales = "free") +
  geom_histogram(fill='cyan4', col="black", alpha=0.7)+
  theme_bw()+
  labs(x="Variable", y="Absolute frequency", title="Distribution of variables")
  
### b. Determine the optimal number of clusters as indicated by NbClust heuristics.

# Checking the optimal number of clusters
fviz_nbclust(data, kmeans, method = "wss")

nb <- NbClust(data, method = "kmeans", min.nc = 2, max.nc = 10, index = "all")

# Around 2 or 3 the optimal number of clusters

### c. Use the k-means method to cluster states using the number of clusters found in a) and anything else 
# that you think that makes sense. Plot observations colored by clusters in the space of urban population 
# and another (crime-related) variable. (See example code from class, use factor(km$cluster) to create a 
# vector of class labels).

plot_clusters_with_centers <- function(features, kmeans_object) {
  
  data_w_clusters <- mutate(features, cluster = factor(kmeans_object$cluster))
  
  centers <- as_tibble(kmeans_object$centers) %>%
    mutate(cluster = factor(seq(nrow(km$centers))), center = TRUE)
  
  data_w_clusters_centers <- bind_rows(data_w_clusters, centers)
  ggplot(data_w_clusters_centers, aes(
    x = UrbanPop, y = Assault,
    color = cluster, size = ifelse(!is.na(center), 2, 1))
  ) +
    geom_point() +
    scale_size(guide = 'none')
}

set.seed(1234)
km <- kmeans(data, centers = 2, nstart = 5)

plot_clusters_with_centers(data, km)


### d. Perform PCA and get the first two principal component coordinates for all observations by

pca_result <- prcomp(data, scale = TRUE)
first_two_pc <- as_tibble(pca_result$x[, 1:2])
city <- rownames(pca_result$x)
first_two_pc <- first_two_pc %>% 
  mutate(clusters = factor(km$cluster),city = city)

ggplot(first_two_pc, aes(x = PC1, y = PC2, color = clusters)) +
  geom_point() + 
  labs(title="Clusters First two principal components")

ggplot(first_two_pc, aes(x = PC1, y = PC2, color = clusters)) +
  geom_point() + geom_text(label = city, size = 2, hjust = -0.1) +
  labs(title="First two principal components by cities")
