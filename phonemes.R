#############################
# LIBRARY + SET UP
#############################
# Draw tex figures
library(tikzDevice)

# Statistics 
library(MASS)

# Cross Validation
library(caret)

# Neural Network for Multinomial Logistic Regression
library("nnet")

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
pca = prcomp(train_set.x, center = TRUE, scale. = TRUE)

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
# Best M on Test Set
accs = matrix(0, 100, 1)

for (M in 2:100) {
  train_set.pca.x = as.data.frame(pca$x[,1:M])
  train_set.pca = as.data.frame(cbind(train_set.y, train_set.pca.x))
  
  test_set.pca.x = predict(pca, newdata = test_set.x)[,1:M]
  test_set.pca = as.data.frame(cbind(test_set.y, test_set.pca.x))
  
  model.lda.pca = lda(train_set.y ~ ., data = train_set.pca)
  model.lda.pca.predicted = predict(model.lda.pca,newdata=test_set.pca)
  
  perf = table(test_set$g,model.lda.pca.predicted$class)
  accs[M] = sum(diag(perf))/dim(test_set)[1]
}

model.lda.pca.best_M = which.max(accs[-1])

#############################
# Best M on With Cross Validation

accs = matrix(0, 50, 1)
for (M in 2:50) {
  a.train_set.pca.x = as.data.frame(pca$x[,1:M])
  a.train_set.pca = as.data.frame(cbind(train_set.y, a.train_set.pca.x))
  
  folds = createFolds(train_set.pca$train_set.y)
  
  acc = 0;
  for (k in 1:10) {
    
    validation_indexes = folds[[k]]
    a.train_set.x = a.train_set.pca.x[-validation_indexes,]
    a.train_set = a.train_set.pca[-validation_indexes,]
    
    a.validation_set.x = a.train_set.pca.x[validation_indexes,]
    a.validation_set = a.train_set.pca[validation_indexes,]
    
    model.lda.pca = lda(a.train_set$train_set.y ~ ., data = a.train_set)
    model.lda.pca.predicted = predict(model.lda.pca,newdata=a.validation_set)
    
    perf = table(a.validation_set$train_set.y,model.lda.pca.predicted$class)
    acc = acc + sum(diag(perf))/dim(a.validation_set)[1]
  }
  
  acc = acc / 10
  accs[M] = acc
}

model.lda.pca.cv.best_M = which.max(accs[-1])

#############################
# Test best M en Test Set
M = model.lda.pca.best_M
#M = model.lda.pca.cv.best_M

train_set.pca.x = as.data.frame(pca$x[,1:M])
train_set.pca = as.data.frame(cbind(train_set.y, train_set.pca.x))

test_set.pca.x = predict(pca, newdata = test_set.x)[,1:M]
test_set.pca = as.data.frame(cbind(test_set.y, test_set.pca.x))

model.lda.pca = lda(train_set.y ~ ., data = train_set.pca)
model.lda.pca.predicted = predict(model.lda.pca,newdata=test_set.pca)

perf = table(test_set$g,model.lda.pca.predicted$class)
perf
sum(diag(perf))/dim(test_set)[1]

#############################
# QUADRATIC DISCRIMINANT ANALYSIS (QDA)
#############################
# Fit Model
model.qda = qda(g ~ ., data=train_set)

# On Test Set
model.qda.predicted = predict(model.qda,newdata=test_set)
model.qda.predicted.perf = table(test_set$g,model.qda.predicted$class)
model.qda.predicted.accuracy = sum(diag(model.qda.predicted.perf))/dim(test_set)[1]

#############################
# QUADRATIC DISCRIMINANT ANALYSIS (QDA) + PCR
#############################
# Best M on Test Set
accs = matrix(0, 100, 1)

for (M in 2:100) {
  train_set.pca.x = as.data.frame(pca$x[,1:M])
  train_set.pca = as.data.frame(cbind(train_set.y, train_set.pca.x))
  
  test_set.pca.x = predict(pca, newdata = test_set.x)[,1:M]
  test_set.pca = as.data.frame(cbind(test_set.y, test_set.pca.x))
  
  model.qda.pca = qda(train_set.y ~ ., data = train_set.pca)
  model.qda.pca.predicted = predict(model.qda.pca,newdata=test_set.pca)
  
  perf = table(test_set$g,model.qda.pca.predicted$class)
  accs[M] = sum(diag(perf))/dim(test_set)[1]
}

model.qda.pca.best_M = which.max(accs[-1])

#############################
# Best M on With Cross Validation
accs = matrix(0, 50, 1)
for (M in 2:50) {
  a.train_set.pca.x = as.data.frame(pca$x[,1:M])
  a.train_set.pca = as.data.frame(cbind(train_set.y, a.train_set.pca.x))
  
  folds = createFolds(train_set.pca$train_set.y)
  
  acc = 0;
  for (k in 1:10) {
    
    validation_indexes = folds[[k]]
    a.train_set.x = a.train_set.pca.x[-validation_indexes,]
    a.train_set = a.train_set.pca[-validation_indexes,]
    
    a.validation_set.x = a.train_set.pca.x[validation_indexes,]
    a.validation_set = a.train_set.pca[validation_indexes,]
    
    model.qda.pca = qda(a.train_set$train_set.y ~ ., data = a.train_set)
    model.qda.pca.predicted = predict(model.qda.pca,newdata=a.validation_set)
    
    perf = table(a.validation_set$train_set.y,model.qda.pca.predicted$class)
    acc = acc + sum(diag(perf))/dim(a.validation_set)[1]
  }
  
  acc = acc / 10
  accs[M] = acc
}

model.qda.pca.cv.best_M = which.max(accs[-1])

#############################
# Test best M en Test Set
M = model.qda.pca.cv.best_M
#M = model.qda.pca.best_M

train_set.pca.x = as.data.frame(pca$x[,1:M])
train_set.pca = as.data.frame(cbind(train_set.y, train_set.pca.x))

test_set.pca.x = predict(pca, newdata = test_set.x)[,1:M]
test_set.pca = as.data.frame(cbind(test_set.y, test_set.pca.x))

model.qda.pca = qda(train_set.y ~ ., data = train_set.pca)
model.qda.pca.predicted = predict(model.qda.pca,newdata=test_set.pca)

perf = table(test_set$g,model.qda.pca.predicted$class)
perf
sum(diag(perf))/dim(test_set)[1]

#############################
# LOGISTIC REGRESSION (LR)
#############################
# Fit Model
model.lr = multinom(g ~ ., data=train_set, MaxNWts=2000, maxit=1000)

# On Test Set
model.lr.predicted = predict(model.lr,newdata=test_set)
model.lr.predicted.perf = table(test_set$g,model.lr.predicted)
model.lr.predicted.accuracy = sum(diag(model.lr.predicted.perf))/dim(test_set)[1]

#############################
# LOGISTIC REGRESSION (LR) + PCA
#############################
# Best M on Test Set
accs = matrix(0, 20, 1)

for (M in 2:20) {
  train_set.pca.x = as.data.frame(pca$x[,1:M])
  train_set.pca = as.data.frame(cbind(train_set.y, train_set.pca.x))
  
  test_set.pca.x = predict(pca, newdata = test_set.x)[,1:M]
  test_set.pca = as.data.frame(cbind(test_set.y, test_set.pca.x))
  
  model.lr.pca = multinom(train_set.y ~ ., data = train_set.pca)
  model.lr.pca.predicted = predict(model.lr.pca,newdata=test_set.pca)
  
  perf = table(test_set$g,model.lr.pca.predicted)
  accs[M] = sum(diag(perf))/dim(test_set)[1]
}

model.lr.pca.best_M = which.max(accs[-1])

#############################
# Best M on With Cross Validation
accs = matrix(0, 20, 1)
for (M in 2:20) {
  a.train_set.pca.x = as.data.frame(pca$x[,1:M])
  a.train_set.pca = as.data.frame(cbind(train_set.y, a.train_set.pca.x))
  
  folds = createFolds(train_set.pca$train_set.y)
  
  acc = 0;
  for (k in 1:5) {
    
    validation_indexes = folds[[k]]
    a.train_set.x = a.train_set.pca.x[-validation_indexes,]
    a.train_set = a.train_set.pca[-validation_indexes,]
    
    a.validation_set.x = a.train_set.pca.x[validation_indexes,]
    a.validation_set = a.train_set.pca[validation_indexes,]
    
    model.lr.pca = multinom(a.train_set$train_set.y ~ ., data = a.train_set)
    model.lr.pca.predicted = predict(model.lr.pca,newdata=a.validation_set)
    
    perf = table(a.validation_set$train_set.y,model.lr.pca.predicted)
    acc = acc + sum(diag(perf))/dim(a.validation_set)[1]
  }
  
  acc = acc / 10
  accs[M] = acc
}

model.lr.pca.cv.best_M = which.max(accs[-1])

#############################
# Test best M en Test Set
#M = model.lr.pca.best_M
M = model.lr.pca.cv.best_M

train_set.pca.x = as.data.frame(pca$x[,1:M])
train_set.pca = as.data.frame(cbind(train_set.y, train_set.pca.x))

test_set.lr.x = predict(pca, newdata = test_set.x)[,1:M]
test_set.lr = as.data.frame(cbind(test_set.y, test_set.pca.x))

model.lr.pca = multinom(train_set.y ~ ., data = train_set.pca)
model.lr.pca.predicted = predict(model.lr.pca,newdata=test_set.pca)

perf = table(test_set$g,model.lr.pca.predicted)
perf
sum(diag(perf))/dim(test_set)[1]