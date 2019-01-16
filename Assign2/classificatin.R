
# Load the 'class' package
signs <- read.table("signs.txt", header=T)
next_sign <- read.table("next_sign.txt", header=T)
test_signs <- read.table("test_signs.txt", header=T)
library(class)



# Create a vector of labels
sign_types <- signs$sign_type


# Classify the next sign observed
signs_actual <- test_signs$sign_type
knn(train = signs[-1], test = next_sign, cl = sign_types)

accuracy <- c()
for(i in 1:15){
  k <- knn(train = signs[-1], test = test_signs[-1], cl = sign_types,
           k=i)
 accuracy <- c(accuracy,mean(signs_actual == k))
}

plot(accuracy, type = 'l')
max(accuracy)
