
# Load libraries.
library(leaps)
source("config.R")
source("partitionData.R")

# get the cleaned data
data = read.csv("data/cleaned/train.csv")

###
# Partition Data
###
partitionData(data,"Survived",use.validation)


###
# Best Subsets Regression
###

# Adjusted R2 method.
lr2 = leaps(data.matrix(training[,3:9]), training[,2], method="adjr2")
# Best combination of adjusted R2 predictors: Suvival ~ Pclass + Sex + Age + Sibsp
print(names(training[,3:9])[lr2$which[match(max(lr2$adjr2), lr2$adjr2),]])

# Mallow's Cp-statistic method.
lcp = leaps(data.matrix(training[,3:9]), training[,2], method="Cp")
# Many different combinations of predictors would be suitable, but at a minimum, PClass, Sex, and Age must be used.
print(lcp$which[lcp$Cp <= 8,])