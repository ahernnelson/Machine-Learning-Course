library(pacman)
p_load("gbm", "Metrics", "ipred", "rpart", "rpart.plot", "caret")
# Convert "yes" to 1, "no" to 0
credit_train$default <- ifelse(credit_train$default == "yes", 1, 0)

# Train a 10000-tree GBM model
set.seed(1)
credit_model <- gbm(formula = default ~ ., 
                    distribution = "bernoulli", 
                    data = credit_train,
                    n.trees = 10000)

# Print the model object                    
print(credit_model)

# summary() prints variable importance
summary(credit_model)
# Since we converted the training response col, let's also convert the test response col
credit_test$default <- ifelse(credit_test$default == "yes", 1, 0)

# Generate predictions on the test set
preds1 <- predict(object = credit_model, 
                  newdata = credit_test,
                  n.trees = 10000)

# Generate predictions on the test set (scale to response)
preds2 <- predict(object = credit_model, 
                  newdata = credit_test,
                  n.trees = 10000,
                  type = "response")

# Compare the range of the two sets of predictions
range(preds1)
range(preds2)
# Generate the test set AUCs using the two sets of preditions & compare
auc(actual = credit_test$default, predicted = preds1)  #default
auc(actual = credit_test$default, predicted = preds2)  #rescaled

# Optimal ntree estimate based on OOB
ntree_opt_oob <- gbm.perf(object = credit_model, 
                          method = "OOB", 
                          oobag.curve = TRUE)

# Train a CV GBM model
set.seed(1)
credit_model_cv <- gbm(formula = default ~ ., 
                       distribution = "bernoulli", 
                       data = credit_train,
                       n.trees = 10000,
                       cv.folds = 2)

# Optimal ntree estimate based on CV
ntree_opt_cv <- gbm.perf(object = credit_model_cv, 
                         method = "cv")

# Compare the estimates                         
print(paste0("Optimal n.trees (OOB Estimate): ", ntree_opt_oob))                         
print(paste0("Optimal n.trees (CV Estimate): ", ntree_opt_cv))

# Generate predictions on the test set using ntree_opt_oob number of trees
preds1 <- predict(object = credit_model, 
                  newdata = credit_test,
                  n.trees = ntree_opt_oob)

# Generate predictions on the test set using ntree_opt_cv number of trees
preds2 <- predict(object = credit_model, 
                  newdata = credit_test,
                  n.trees = ntree_opt_cv)   

# Generate the test set AUCs using the two sets of preditions & compare
auc1 <- auc(actual = credit_test$default, predicted = preds1)  #OOB
auc2 <- auc(actual = credit_test$default, predicted = preds2)  #CV 

# Compare AUC 
print(paste0("Test set AUC (OOB): ", auc1))                         
print(paste0("Test set AUC (CV): ", auc2))

# # Generate the test set AUCs using the two sets of predictions & compare
# actual <- credit_test$default
# dt_auc <- auc(actual = actual, predicted = dt_preds)
# bag_auc <- auc(actual = actual, predicted = bag_preds)
# rf_auc <- auc(actual = actual, predicted = rf_preds)
# gbm_auc <- auc(actual = actual, predicted = gbm_preds)
# 
# # Print results
# sprintf("Decision Tree Test AUC: %.3f", dt_auc)
# sprintf("Bagged Trees Test AUC: %.3f", bag_auc)
# sprintf("Random Forest Test AUC: %.3f", rf_auc)
# sprintf("GBM Test AUC: %.3f", gbm_auc)
# List of predictions
# preds_list <- list(dt_preds, bag_preds, rf_preds, gbm_preds)
# 
# # List of actual values (same for all)
# m <- length(preds_list)
# actuals_list <- rep(list(credit_test$default), m)
# 
# # Plot the ROC curves
# pred <- prediction(preds_list, actuals_list)
# rocs <- performance(pred, "tpr", "fpr")
# plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
# legend(x = "bottomright", 
#        legend = c("Decision Tree", "Bagged Trees", "Random Forest", "GBM"),
#        fill = 1:m)