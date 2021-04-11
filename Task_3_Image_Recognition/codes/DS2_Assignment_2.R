######################################################
#                                                    #
#   Attila Serfozo - Data Science 2 - Assignment 2   #
#                                                    #
######################################################


# 1. Fashion MNIST --------------------------------------------------------

path <- "D:/Egyetem/CEU/Winter_Term/Data_Science_2/Assignments/Assignment2"
setwd(path)

# Load packages and data
library(tidyverse)
library(keras)
my_seed <- 20210406

# Import data
fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y


# a. Show some example images from the data. ------------------------------

# Save labels into a vector
labels <- c("T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot")

# Create funtion for getting label names
get_label <- function(num) {
  labels[num+1]
}

# Show some clothes
rotateMatrix <- function(data_row) {
  t(apply(data_row, 2, rev))
}

showClothes <- function(data_row, label,x) {
  num <- label
  image(
    rotateMatrix(data_row),
    col = gray.colors(255), xlab = get_label(num), ylab = ""
  )
}

showFirst9Clothes <- function(data, label) {
  par(mfrow = c(3, 3))
  walk(1:9, ~showClothes(data[.,, ], label[.]))
}

showFirst9Clothes(x_train,y_train)


# b. Train a fully connected deep network to predict items. ---------------

# Normalize the data

x_train <- as.matrix(as.data.frame.array(x_train)) / 255
x_test <- as.matrix(as.data.frame.array(x_test)) / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Experiment with network architectures and settings (number of hidden layers, number of nodes, activation functions, dropout, etc.)
# Make sure that you use enough epochs so that the validation error starts flattening out 
# provide a plot about the training history (plot(history))

# Model 1

model1 <- keras_model_sequential()
model1 %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>%   # 128 first layer nodes and 784 input features (pixels)
  layer_dropout(rate = 0.3) %>%                                             # 30% of the data gets randomly droped out
  layer_dense(units = 10, activation = 'softmax')                           # 10 nods for the 10 results

compile(
  model1,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_1 <- fit(
  model1, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(history_1, paste0(path,"/output/model1.rds"))
history_1 <- readRDS(paste0(path,"/output/model1.rds"))

history_1
plot(history_1)


# Model 2 increase number of nods

model2 <- keras_model_sequential()
model2 %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%  
  layer_dropout(rate = 0.3) %>%   
  layer_dense(units = 10, activation = 'softmax')                           

compile(
  model2,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_2 <- fit(
  model2, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(history_2, paste0(path,"/output/model2.rds"))
history_2 <- readRDS(paste0(path,"/output/model2.rds"))

history_2
plot(history_2)
# It has a better accuracy, but the gap increase between the train and validation


# Model 3 add another layer

model3 <- keras_model_sequential()
model3 %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%   
  layer_dropout(rate = 0.3) %>%     
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>%   
  layer_dropout(rate = 0.3) %>%  
  layer_dense(units = 10, activation = 'softmax')            

compile(
  model3,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_3 <- fit(
  model3, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(history_3, paste0(path,"/output/model3.rds"))
history_3 <- readRDS(paste0(path,"/output/model3.rds"))

history_3
plot(history_3)
# validation accuracy have slightly decreased, while training set accuracy decreased, now there is a smaller gap between the two set


# Model 4 increasing add the second layer with the start parameters 

model4 <- keras_model_sequential()
model4 %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.3) %>%            
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>%   
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')          

compile(
  model4,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_4 <- fit(
  model4, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(history_4, paste0(path,"/output/model4.rds"))
history_4 <- readRDS(paste0(path,"/output/model4.rds"))

history_4
plot(history_4)
# increasing the node size in the second layer did not improve the results


# Model 5 decreasing the dropout rate

model5 <- keras_model_sequential()
model5 %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%   
  layer_dropout(rate = 0.2) %>%     
  layer_dense(units = 10, activation = 'softmax')            

compile(
  model5,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_5 <- fit(
  model5, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(history_5, paste0(path,"/output/model5.rds"))
history_5 <- readRDS(paste0(path,"/output/model5.rds"))

history_5
plot(history_5)
# The test set accuracy slightly increased, but the validation set decreased a little


# Model 6 changing the activation function

model6 <- keras_model_sequential()
model6 %>%
  layer_dense(units = 256, activation = 'softmax', input_shape = c(784)) %>%   
  layer_dropout(rate = 0.3) %>%     
  layer_dense(units = 10, activation = 'softmax')            

compile(
  model6,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history_6 <- fit(
  model6, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(history_6, paste0(path,"/output/model6.rds"))
history_6 <- readRDS(paste0(path,"/output/model6.rds"))

history_6
plot(history_6)
# It dropped the accuracy significantly


# The best model was the 2nd where we increased the number of nods to 256
results_1 <- data.frame(
  model1 = round(history_1$metrics$val_accuracy[30],4),
  model2 = round(history_2$metrics$val_accuracy[30],4),
  model3 = round(history_3$metrics$val_accuracy[30],4),
  model4 = round(history_4$metrics$val_accuracy[30],4),
  model5 = round(history_5$metrics$val_accuracy[30],4),
  model6 = round(history_6$metrics$val_accuracy[30],4)
)
results_1

# c. Evaluate the model on the test set. How does test error compare to validation error?

evaluate(model2, x_test, y_test)


# d. Try building a convolutional neural network and see if you can improve test set performance.

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))


# CNN Model 1 using the parameters from class

cnn_model_1 <- keras_model_sequential()
cnn_model_1 %>%
  layer_conv_2d(
    filters = 32,               # We do it 32 times 
    kernel_size = c(3, 3),      # It has to take a 3x3 square and evaluate the pixels, do for all possible 3x3
    activation = 'relu',        # 
    input_shape = c(28, 28, 1)  # What kind of input it expects, it is an array/matrix -> this is the image we add
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%    # pooling layer 2x2, it aggregates the 2x2 layer into 1 pixel
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')  # 10 output nodes

compile(
  cnn_model_1,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

cnn_history_1 <- fit(
  cnn_model_1, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(cnn_history_1, paste0(path,"/output/cnn_model1.rds"))
cnn_history_1 <- readRDS(paste0(path,"/output/cnn_model1.rds"))

cnn_history_1
plot(cnn_history_1)
# CNN model resulted a significantly better model than the simple neural nets

# CNN Model 2 I tried several options for node size (32, 64, 128) and the 32 was the best

cnn_model_2 <- keras_model_sequential()
cnn_model_2 %>%
  layer_conv_2d(
    filters = 32,               
    kernel_size = c(3, 3),      
    activation = 'relu',        
    input_shape = c(28, 28, 1)  
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%    
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  cnn_model_2,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

cnn_history_2 <- fit(
  cnn_model_2, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(cnn_history_2, paste0(path,"/output/cnn_model2.rds"))
cnn_history_2 <- readRDS(paste0(path,"/output/cnn_model2.rds"))

cnn_history_2
plot(cnn_history_2)
# The validation accuracy even increased further and the train accuracy increased significantly


# CNN Model 3 try lower dropout rate

cnn_model_3 <- keras_model_sequential()
cnn_model_3 %>%
  layer_conv_2d(
    filters = 32,               
    kernel_size = c(3, 3),      
    activation = 'relu',        
    input_shape = c(28, 28, 1)  
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%    
  layer_dropout(rate = 0.2) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  cnn_model_3,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

cnn_history_3 <- fit(
  cnn_model_3, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(cnn_history_3, paste0(path,"/output/cnn_model3.rds"))
cnn_history_3 <- readRDS(paste0(path,"/output/cnn_model3.rds"))

cnn_history_3
plot(cnn_history_3)
# decreasing the dropout rate did not improve the accuracy


# CNN Model 4 try other activation function

cnn_model_4 <- keras_model_sequential()
cnn_model_4 %>%
  layer_conv_2d(
    filters = 32,               
    kernel_size = c(3, 3),      
    activation = 'relu',        
    input_shape = c(28, 28, 1)  
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%    
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = 'softmax') %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  cnn_model_4,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

cnn_history_4 <- fit(
  cnn_model_4, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(cnn_history_4, paste0(path,"/output/cnn_model4.rds"))
cnn_history_4 <- readRDS(paste0(path,"/output/cnn_model4.rds"))

cnn_history_4
plot(cnn_history_4)
# Changing the activation function did not increase the accuracy


# CNN Model 5 add one more layer

cnn_model_5 <- keras_model_sequential()
cnn_model_5 %>%
  layer_conv_2d(
    filters = 32,               
    kernel_size = c(3, 3),      
    activation = 'relu',        
    input_shape = c(28, 28, 1)  
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%    
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

compile(
  cnn_model_5,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

cnn_history_5 <- fit(
  cnn_model_5, x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.1
)
#saveRDS(cnn_history_5, paste0(path,"/output/cnn_model5.rds"))
cnn_history_5 <- readRDS(paste0(path,"/output/cnn_model5.rds"))

cnn_history_5
plot(cnn_history_5)
# It could not beat either the second model



# The best model was the 2nd where we increased the number of nods to 32
results_2 <- data.frame(
  cnn_model1 = round(cnn_history_1$metrics$val_accuracy[30],4),
  cnn_model2 = round(cnn_history_2$metrics$val_accuracy[30],4),
  cnn_model3 = round(cnn_history_3$metrics$val_accuracy[30],4),
  cnn_model4 = round(cnn_history_4$metrics$val_accuracy[30],4),
  cnn_model5 = round(cnn_history_5$metrics$val_accuracy[30],4)
)
results_2

evaluate(cnn_model_2, x_test, y_test)