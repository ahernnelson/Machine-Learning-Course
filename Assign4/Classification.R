############# packages ############
library(pacman)
p_load("rpart", "rpart.plot", "caret")
##### Credit data ###################
### Clean #############
# Read in data
credit <- read.csv("credit.csv")
# Look at the data
str(credit)
# Simplify like dc
simple <-  c("months_loan_duration","percent_of_income",
             "years_at_residence","age","default")            
set.seed(123)
id <- sample(1:1000, 522)
creditsub <- credit[id,simple]

##### Apply ##############
# Look at data
str(creditsub)
# Create the model
credit_model <- rpart(formula = default ~ ., 
                      data = creditsub, 
                      method = "class")

# Display the results
rpart.plot(x = credit_model , yesno = 2, type = 0, extra = 0)
#### Train test #############
# Total number of rows in the credit data frame
n <- nrow(credit)

# Number of rows for the training set (80% of the dataset)
n_train <- round(.8 * n) 

# Create a vector of indices which is an 80% random sample
set.seed(123)
train_indices <- sample(1:n, n_train)

# Subset the credit data frame to training indices only
credit_train <- credit[train_indices, ]  

# Exclude the training indices to create the test set
credit_test <- credit[-train_indices, ]
##### Model ###################
# Train the model (to predict 'default')
credit_model <- rpart(formula = default ~ ., 
                      data = credit_train, 
                      method = "class")

# Look at the model output                      
print(credit_model)
rpart.plot(credit_model)
##### Evaluate ##################
library(caret)
class_prediction <- predict(object = credit_model,
                             newdata = credit_test,
                             type = "class") 
confusionMatrix(data = class_prediction,
                reference = credit_test$default)
##### split criteria #########
# Train a gini-based model
credit_model1 <- rpart(formula = default ~ ., 
                       data = credit_train, 
                       method = "class",
                       parms = list(split = "gini"))

# Train an information-based model
credit_model2 <- rpart(formula = default ~ ., 
                       data = credit_train, 
                       method = "class",
                       parms = list(split = "information"))

# Generate predictions on the validation set using the gini model
pred1 <- predict(object = credit_model1, 
                 newdata = credit_test,
                 type = "class")    

# Generate predictions on the validation set using the information model
pred2 <- predict(object = credit_model2, 
                 newdata = credit_test,
                 type = "class")

# Compare classification error
ModelMetrics::ce(actual = credit_test$default, 
   predicted = pred1)
ModelMetrics::ce(actual = credit_test$default, 
   predicted = pred2) 
# confusion
confusionMatrix(data = pred1, credit_test$default)
confusionMatrix(data = pred2, credit_test$default)
