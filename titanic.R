# Clear environments.
rm(list=ls(), inherits=T) 

skipTraining = F
ignorecores = 2 # Number of cores left free when training.
use.validation = F

source("partitionData.R")
source("activateParallel.R")
activateParallel(ignorecores)

# Read files.
data = read.csv("data/train.csv", header=T, 
                   colClasses=c("integer", "factor", "factor", "character", 
                                 "factor", "numeric", "integer", "integer", 
                                 "character", "numeric", "character", "factor"))
final.test = read.csv("data/test.csv", header=T)

# Set seed.
set.seed(3846)

# Load libraries.
library(caret)
library(plyr) 
library(leaps)

###
# Pre-processing
###

levels(data$Survived) = c("N", "Y")
levels(data$Sex) = c("F", "M")

# Remove Name, Ticket, Cabin, and incomplete cases.
data <- data[complete.cases(data), c(-4, -9, -11)]

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

###
# Training Original Models
###

if(!skipTraining) {   
   
   # Define formula to use for training.
   formula = formula(Survived ~ Age + Pclass + Sex + SibSp)
   
   begin = Sys.time()
   
   # 1- Trees (rpart method)
   message("Training model (rpart)...")
   fitTREE <- train(formula, data=training, method='rpart', preProcess=c("center", "scale"))
   
   # 2- Random Forest
   message("Training model (Random Forests)...")
   fitControlRF <- trainControl(method = "cv", number = 30)
   fitRF <- train(formula, data=training, method='rf', preProcess=c("center", "scale"), 
                  trControl = fitControlRF, prox=TRUE, verbose = FALSE)
   
   # 3- GBM
   message("Training model (GBM)...")
   fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 20)
   fitGBM <- train(formula, data=training, method='gbm', trControl=fitControl, 
                   verbose = FALSE)
   
   # 4- Support Vector Machine (SVM)
   message("Training model (SVM)...")
   fitSVM <- train(formula, data=training, method='svmLinear', trControl=fitControl)
   
   # 5- Neural Net
   message("Training model (Neural Net)...")
   fitANN = train(formula, data=training, method='avNNet')
   
   end = Sys.time()
   message(sprintf("Total time to fit models, with %d cores: %d minutes %d seconds.", 
                   detectCores() - ignorecores, 
                   floor(as.numeric(end-begin, units="mins")), 
                   floor(as.numeric(end-begin, units="secs")) %% 60))
   
   # Store the names of each model.
   method <- c('TREE', 'RF', 'GBM', 'SVM', 'ANN')
   
   # Stores all of the models in a list.
   # Each model name must begin with "fit".
   # The suffix of each model name must be added to 'method', above.
   models = lapply(paste("fit", method, sep=""), get)
}
   
###
#  Predicting With Original Models
###

# Predict on each dataset with the original models. (Not sure yet which of these we're going to need)
train.predict = data.frame(lapply(models, predict, training), training$Survived)

test.predict = data.frame(lapply(models, predict, testing), testing$Survived)

names(train.predict) = names(test.predict) = c(method, "Survived")

if(use.validation) {
   valid.predict = data.frame(lapply(models, predict, validation), validation$Survived)
   names(valid.predict) = c(method, "Survived")
}
   
# Create confusion matrices for each model and store them in a list.
confusion = lapply(train.predict[,-ncol(train.predict)], confusionMatrix, train.predict[,ncol(train.predict)])

###
# Accuracy
###

# Calculate in-sample accuracy for each model.
inSample.Accuracy = sapply(models, function(x) {max(x$results$Accuracy)})

# Estimate OOS accuracy for each model.
outSample.Accuracy = sapply(confusion, function(x) {x$overall["Accuracy"]})

# Create errors data frame.
errors.df <- data.frame(method = method, 
                        inSample.Acc = inSample.Accuracy, 
                        inSample.Err = 1 - inSample.Accuracy, 
                        outSample.Acc = outSample.Accuracy, 
                        outSample.Err = 1 - outSample.Accuracy, 
                        real.outSample.Acc = sapply(subset(test.predict, select = -Survived), 
                            function(x) {sum(x == test.predict$Survived) / nrow(test.predict)}),
                        stringsAsFactors=FALSE)
print(errors.df)

cat(sprintf(
"\nHighest in-sample accuracy: %.02f%% with method %s
Highest estimated out-of-sample accuracy: %.02f%% with method %s
Highest real out-of-sample accuracy: %.02f%% with method %s\r\n", 
            max(errors.df$inSample.Acc) * 100, 
            errors.df[which.max(errors.df$inSample.Acc), 1],
            max(errors.df$outSample.Acc) * 100,
            errors.df[which.max(errors.df$outSample.Acc), 1],
            max(errors.df$real.outSample.Acc) * 100,
            errors.df[which.max(errors.df$real.outSample.Acc), "method"]
   )
)
            

###
# Prediction
###
if(!skipTraining) {      
   # Create stacked model on predictions.
   message("Training stacked model...")
   stackedModel = train(Survived ~ ., method="rf", data=train.predict)
}

# Predict with stacked model.
stacked.train.predict = predict(stackedModel, train.predict)
stacked.test.predict = predict(stackedModel, test.predict)

if(use.validation)
   stacked.valid.predict = predict(stackedModel, valid.predict)

# Print accuracy.
cat(sprintf("In-sample accuracy with stacked model: %.02f%%\n", 
            sum(stacked.train.predict == train.predict$Survived) / length(train.predict$Survived) * 100))

if(use.validation)
   cat(sprintf("Validation set accuracy with stacked model: %.02f%%\n", 
               sum(stacked.valid.predict == valid.predict$Survived) / length(valid.predict$Survived) * 100))

cat(sprintf("Real out-of-sample accuracy with stacked model: %.02f%%\n", 
            sum(stacked.test.predict == test.predict$Survived) / length(test.predict$Survived) * 100))

