#############################
# LIBRARY + SET UP
#############################
# Draw tex figures
library(tikzDevice)

library(MASS)

# Cross Validation
library(caret)

#############################
# LOAD DATA SET
#############################
data_set = read.csv("phoneme.data.txt")
data_set.nb_features = 256

#############################
# DATA SPLIT
#############################
# Train Set
train_set= data_set[1:3340, 2:258]
train_set.x = train_set[,1:256]
train_set.y = train_set[,257]

# Test Set
test_set = data_set[3341:4509, 2:258]
test_set.x = test_set[,1:256]
test_set.y = test_set[,257]

#############################
# PRINCIPAL COMPONENTS ANALYSIS
#############################
train_set.x.pca = prcomp(train_set.x, center = TRUE, scale. = TRUE)

#############################
# LINEAR DISCRIMINANT ANALYSIS (LDA)
#############################
# Fit Model
model.lda = lda(g ~ ., data=train_set)

# On Test Set
model.lda.predicted = predict(model.lda,newdata=test_set)
model.lda.predicted.perf = table(test_set$g,model.lda.predicted$class)
model.lda.predicted.accuracy = sum(diag(model.lda.predicted.perf))/dim(test_set)[1]

#############################
# LINEAR DISCRIMINANT ANALYSIS (LDA) + PCR
#############################

