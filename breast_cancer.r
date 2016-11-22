#############################
# LIBRARY + SET UP
#############################
# Draw tex figures
library(tikzDevice)

# Linear Regression Diagnosis
library(car) 

# Subset Selectin
library(leaps)

library(MASS)

# kNN
library(FNN)

# Principal Component Regression
library(pls)

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
# - frctal_dimension_mean
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

#############################
# LASSO REGRESSION
#############################

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




