# DELETE THIS COMMENT
Mac = F
Windows = T
skipTraining = F
ignorecores = 2 # Number of cores left free when training.

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
new.data <- data[complete.cases(data), c(-4, -9, -11)]

###
# Partition Data
###

# Create data partition with 50% training data, 25% validation data, and 25% test data.
indexSample <- createDataPartition(new.data$Survived, p=.5, list=FALSE)
training <- new.data[indexSample,]
attach(training)
validation <- new.data[-indexSample,]
validationIndices = createDataPartition(validation$Survived, p=.5, list=FALSE)
testing = validation[-validationIndices,]
validation = validation[validationIndices,]

###
# Parameter Pruning
###

# Perform Best Subsets Regression.

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

if(Mac) {
   # Set up parallel processors for Mac.
   library('doMC')
   registerDoMC(cores = detectCores() - ignorecores)
} else if(Windows) {
   # Set up parallel processors for Windows.
   library(doParallel)
   cl = makeCluster(detectCores() - ignorecores)
   registerDoParallel(cl)
}

if(!skipTraining) {
   begin = Sys.time()
   
   # 1- Trees (rpart method)
   message("Training model (rpart)...")
   fitTREE <- train(Survived ~ Age + Pclass + Sex + SibSp, data=training, method='rpart', preProcess=c("center", "scale"))
   
   # 2- Random Forest
   message("Training model (Random Forests)...")
   fitControlRF <- trainControl(method = "cv", number = 30)
   fitRF <- train(Survived ~ Age + Pclass + Sex + SibSp, data=training, method='rf', preProcess=c("center", "scale"), 
                  trControl = fitControlRF, prox=TRUE, verbose = FALSE)
   
   # 3- GBM
   message("Training model (GBM)...")
   fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 20)
   fitGBM <- train(Survived ~ Age + Pclass + Sex + SibSp, data = training, method = 'gbm', trControl = fitControl, 
                   verbose = FALSE)
   
   # 4- Support Vector Machine (SVM)
   message("Training model (SVM)...")
   fitSVM <- train(Survived ~ Age + Pclass + Sex + SibSp, data=training, method = 'svmLinear', trControl = fitControl)
   
   end = Sys.time()
   message(sprintf("Total time to fit models, with %d cores: %d minutes %d seconds.", 
                   detectCores() - ignorecores, 
                   floor(as.numeric(end-begin, units="mins")), 
                   floor(as.numeric(end-begin, units="secs")) %% 60))
}
   
###
# Cross-validation and Training Prediction
###

method <- c('trees', 'RF', 'GBM', 'SVM')

train.predict = data.frame(
                  predict(fitTREE, training), 
                  predict(fitRF, training), 
                  predict(fitGBM, training), 
                  predict(fitSVM, training), 
                  training$Survived)

valid.predict = data.frame(
                  predict(fitTREE, validation), 
                  predict(fitRF, validation), 
                  predict(fitGBM, validation), 
                  predict(fitSVM, validation), 
                  validation$Survived)
names(train.predict) = names(valid.predict) = c(method, "Survived")

# Calculate in-sample accuracy for each model.
inSample.Accuracy <- c(max(fitTREE$results[,2]), 
                       max(fitRF$results[,2]), 
                       max(fitGBM$results[,4]),
                       max(fitSVM$results[,2])
)

# Create confusion matrices for each model and store them in a list.
confusion = lapply(train.predict[,1:4], confusionMatrix, train.predict[,5])

# Estimate OOS accuracy for each model.
outSample.Accuracy = NULL
for(cm in confusion) {
   outSample.Accuracy = c(outSample.Accuracy, cm$overall[1])
}

# Create errors data frame.
errors.df <- data.frame(method = method, 
                        inSample.Acc = inSample.Accuracy, 
                        inSample.Err = 1 - inSample.Accuracy, 
                        outSample.Acc = outSample.Accuracy, 
                        outSample.Err = 1 - outSample.Accuracy, 
                        stringsAsFactors=FALSE)
print(errors.df)

cat(sprintf("Highest in-sample accuracy: %.02f%% with method %s\r\nHighest estimated out-of-sample accuracy: %.02f%% with method %s\n", 
            max(errors.df$inSample.Acc) * 100, 
            errors.df[which.max(errors.df$inSample.Acc), 1],
            max(errors.df$outSample.Acc) * 100,
            errors.df[which.max(errors.df$outSample.Acc), 1]))

###
# Prediction
###
if(!skipTraining) {   
   # Combine predicted values of models on the TRAINING set.
   modelPreds = data.frame(tree=predict(fitTREE, training), rf=predict(fitRF, training), 
                           gbm=predict(fitGBM, training), svm=predict(fitSVM, training), 
                           Survived=training$Survived)
   
   # Create stacked model on predictions.
   message("Training stacked model...")
   stackedModel = train(Survived ~ ., method="gbm", data=modelPreds)
}

# Predict on validation set.
stackedPreds = predict(stackedModel, modelPreds)

# Predict on test set.
tree = predict(fitTREE, testing)
rf = predict(fitRF, testing)
gbm = predict(fitGBM, testing)
svm = predict(fitSVM, testing)
testPreds = data.frame(tree, rf, gbm, svm)
combinedPreds = predict(stackedModel, testPreds)

# Print accuracy.
cat(sprintf("Real out-of-sample prediction accuracy: %.01f%%\n", sum(combinedPreds == testing$Survived) / length(testing$Survived) * 100))