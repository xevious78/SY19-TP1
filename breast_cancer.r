#############################
# LIBRARY + SET UP
#############################
# Draw tex figures
library(tikzDevice)

# Linear Regression Diagnosis
library(car) 

# Subset Selectin
library(leaps)

# Statistics 
library(MASS)

# kNN
library(FNN)

# kNN with CV
library("kknn")
  
# Principal Component Regression
library(pls)

# Special Linear Regression
library(glmnet)

# Setup Random Number Generator
set.seed(123)

#############################
# LOAD DATA SET
#############################
data_set = read.csv("r_breast_cancer.data.txt")
data_set.nb_features = dim(data_set)[1]

#############################
# DATA SPLIT
#############################
n = dim(data_set)[1]
train_id = sample(1:n, n * 2/3)

# Training Set
train_set = data_set[train_id,]
train_set.x = train_set[,-33]
train_set.y = train_set[,33]

# Test Set
test_set = data_set[-train_id,]
test_set.x = test_set[,-33]
test_set.y = test_set[,33]

#############################
# FIRST ANALYSIS
#############################
# Summary
summary(data_set)

# Plot Time distribution
hist(data_set$Time, ylab = "Time", xlab = "Time", main="")
boxplot(data_set$Time)
# -> High variance

# Plot each feature against Time
for (i in 1:(data_set.nb_features-1)) {
  plot(data_set[,i], data_set$Time, xlab=colnames(data_set)[i], ylab="Time")
}
# -> Very scattered plots

#############################
# Model Analysis with Linear Regression
linreg = lm(Time ~ ., data=data_set)

summary(linreg)
plot(linreg)

# Check Outliers with Cooks Distance
plot(cooks.distance(linreg))
# -> No Outliers

# Check residuals normality 
# With QQ-Plot
qqPlot(linreg, main="QQ Plot")
# -> Non normality + Heteroscedasticity

# With Studentized Residuals
sresid = studres(linreg) 
hist(sresid, freq=FALSE, 
     main="Distribution of Studentized Residuals")
xfit = seq(min(sresid),max(sresid),length=40) 
yfit = dnorm(xfit) 
lines(xfit, yfit)

# Check model linearity
crPlots(linreg)
# -> Some non-linearities

#############################
# k NEAREST NEIGHBOUR
#############################
# Find best k on test set

k_max = 120;
MSE = rep(0,k_max)

for( k in 1:k_max)
{
  model.knn = knn.reg(train=train_set.x, test=test_set.x, y=train_set.y, k=k)
  MSE[k] = mean((test_set.y - model.knn$pred)^2)
}

model.knn.best.k = which.min(MSE)
model.knn.best.mse = MSE[model.knn.best.k]
model.knn.best.k
model.knn.best.mse

plot(1:k_max, MSE, xlab='k', ylab='MSE', main='MSE against k neighbours')
points(x = model.knn.best.k, y =model.knn.best.mse, col = "red", pch = 16)#color the minimum point
abline(h = model.knn.best.mse, col='red')#horizontal red line
abline(v = model.knn.best.k, col='red')#vertical red line

# kNN with best K
model.knn.best = knn.reg(train=train_set.x, test=test_set.x, y=train_set.y, k=model.knn.best.k)
model.knn.best.residuals = test_set.y - model.knn.best$pred

hist(model.knn.best.residuals, freq=FALSE, main="Distribution of Residuals in Knn best case")
residuals.mean = mean(model.knn.best.residuals)
residuals.stdev = sqrt(var(model.knn.best.residuals))
curve(dnorm(x, mean=residuals.mean, sd=residuals.stdev), col="darkblue", lwd=2, add=TRUE, yaxt="n")

#############################
# Find best k with Cross Validation
model.kknn = train.kknn(Time ~., data= train_set, kmax = 30, ks = NULL, distance = 2, kernel = "optimal")
model.kknn.best.k = model.kknn$best.parameters$k

# kNN with best k
model.kknn.best = knn.reg(train=train_set.x, test=test_set.x, y=train_set.y, k=model.kknn.best.k)
plot(test_set.y, model.kknn.best$pred, xlab='y', ylab='prediction')
abline(0,1, col='red')

residuals = test_set.y - model.kknn.best$pred
errors = residuals^2
MSE= mean(errors)

hist(residuals, freq=FALSE, main="Distribution of Residuals in Knn LOOCV best case")
residuals.mean = mean(residuals)
residuals.stdev = sqrt(var(residuals))
curve(dnorm(x, mean=residuals.mean, sd=residuals.stdev), col="darkblue", lwd=2, add=TRUE, yaxt="n")

#############################
# LINEAR REGRESSION
#############################
# Fit Model on Train Set
model.linreg = lm(Time ~ ., data=train_set)
summary(model.linreg)

# On Test Set
model.linreg.predicted = predict(model.linreg, newdata = test_set.x)
model.linreg.predicted.mse = mean((test_set.y - model.linreg.predicted)^2)

#############################
# SUBSET SELECTION
#############################
# Exhaustive Subset Selection
model.linreg.regsubsets = regsubsets(Time ~ ., data=train_set, method = "exhaustive", nvmax = 32)
summary(model.linreg.regsubsets)

# Compare models based on BIC measure
plot(model.linreg.regsubsets, scale="bic")
# -> According to BIC Measure, best features are : 
# - texture_mean
# - fractal_dimension_mean
# - compactness_mean

#############################
# Fit model on BIC subset
model.linreg.best_bic = lm(Time ~ texture_mean+fractal_dimension_mean+concavity_mean, data=train_set)

# Analysis
summary(model.linreg.best_bic)
plot(model.linreg.best_bic)

# On Test Set
model.linreg.best_bic.predicted = predict(model.linreg.best_bic, newdata = test_set.x)
model.linreg.best_bic.predicted.mse = mean((test_set.y - model.linreg.best_bic.predicted)^2)

#############################
# RIDGE REGRESSION
#############################
# Setup Data
train_set.x.matrix = as.matrix(train_set.x)
train_set.y.matrix = as.matrix(train_set.y)
test_set.x.matrix = as.matrix(test_set.x)

grid=10^seq(10,-2, length=100)

# Ridge Regression with various lambda
model.ridge = glmnet(train_set.x.matrix, train_set.y.matrix, alpha=0, lambda=grid)
plot(model.ridge)

#############################
# With Cross Validation to find the best Lambda
model.ridge.cv = cv.glmnet(train_set.x.matrix, train_set.y.matrix,alpha=0)
plot(model.ridge.cv)
model.ridge.best.lambda = model.ridge.cv$lambda.min

# glmnet with best lambda
model.ridge.best = glmnet(train_set.x.matrix, train_set.y.matrix, lambda=model.ridge.best.lambda, alpha=0)
model.ridge.best.pred = predict(model.ridge.best, s=model.ridge.best.lambda, newx=test_set.x.matrix)

# Analysis
residuals = test_set.y - model.ridge.best.pred
errors = residuals^2
model.ridge.best.MSE = mean(errors)

plot(x=test_set.y, y=model.ridge.best.pred)
abline(0,1, col='red')

hist(residuals, freq=FALSE, main="Distribution of Residuals in Ridge")
residuals.mean = mean(residuals)
residuals.stdev = sqrt(var(residuals))
curve(dnorm(x, mean=residuals.mean, sd=residuals.stdev), col="darkblue", lwd=2, add=TRUE, yaxt="n")

print(model.ridge.best.lambda)
print(model.ridge.best.MSE)
coeff = predict(model.ridge, type="coefficients", s=model.ridge.best.lambda)[1:33,]
coeff

#############################
# LASSO REGRESSION
#############################
# Lasso Regression with various lambda
model.lasso = glmnet(train_set.x.matrix, train_set.y.matrix, alpha=1, lambda=grid)
plot(model.lasso)

# Cross Validation to find the best lamda
model.lasso.cv.out = cv.glmnet(train_set.x.matrix, train_set.y.matrix, lambda=grid, alpha=1)
plot(model.lasso.cv.out)
model.lasso.best.lambda = model.lasso.cv.out$lambda.min

# glmnet with best lambda
model.lasso.best = glmnet(train_set.x.matrix, train_set.y.matrix, lambda=model.lasso.best.lambda, alpha=1)
model.lasso.best.pred = predict(model.lasso.best, s=model.lasso.best.lambda, newx=test_set.x.matrix)

# Analysis
residuals = test_set.y - model.lasso.best.pred
errors = residuals^2
model.lasso.MSE = mean(errors)

model.lasso.MSE
model.lasso.best.lambda

plot(x=test_set.y, y=model.lasso.best.pred)
abline(0,1, col='red')

hist(residuals, freq=FALSE, main="Distribution of Residuals in Lasso")
residuals.mean = mean(residuals)
residuals.stdev = sqrt(var(residuals))
curve(dnorm(x, mean=residuals.mean, sd=residuals.stdev), col="darkblue", lwd=2, add=TRUE, yaxt="n")

predict(model.lasso, type="coefficients", s=model.lasso.best.lambda)[1:33,]

#############################
# PRINCIPAL COMPONENT REGRESSION
#############################
# Find the best set of Principal Components with Cross Validation
model.pcr = pcr(Time ~ ., data=train_set, scale=TRUE, validation="CV")

# Analysis
summary(model.pcr)
validationplot(model.pcr, val.type = "MSEP")

#############################
# 4 components yield to the lowest MSE (that may be different for you)
# Fit model with 4 principal components
model.pcr.predicted = predict(model.pcr, test_set.x, ncomp = 4)
model.pcr.predicted.mse = mean((test_set.y - model.pcr.predicted)^2)

#############################
# MODELS COMPARISON
#############################

linreg.residuals = test_set.y - model.linreg.predicted
bic.residuals    = test_set.y - model.linreg.best_bic.predicted
kknn.residuals   = test_set.y - model.kknn.best$pred
ridge.residuals  = test_set.y - model.ridge.best.pred 
lasso.residuals  = test_set.y - model.lasso.best.pred
pcr.residuals  = test_set.y - model.pcr.predicted


