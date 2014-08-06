Mac = T
Windows = F
skipTraining = F
ignorecores = 0 # Number of cores left free when training.

data = read.csv("data/train.csv", header=T, 
                   colClasses=c("integer", "factor", "factor", "character", 
                                 "factor", "numeric", "integer", "integer", 
                                 "character", "numeric", "character", "factor"))
final.test = read.csv("data/test.csv", header=T)

## Set seed
set.seed(3846)

library(caret)
library(plyr) 

###
# Pre-processing
###

levels(data$Survived) = c("N", "Y")
levels(data$Sex) = c("F", "M")

# Remove Name, Ticket, and Cabin
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

# Stepwise function finds that the best formula is Survived ~ SibSp + Age + Pclass + Sex
glm = glm(Survived ~ Fare + Embarked + Pclass + Sex + Age + SibSp + Parch, 
          data=training, family="binomial")
aic = step(glm, direction="both")
print(summary(aic))

# All Subsets Regression finds that the best formula is Survived ~ Age + Pclass + Sex
library(leaps)
leaps <- regsubsets(Survived ~ Fare + Embarked + Pclass + Sex + Age + SibSp + Parch,
                    data = training, 
                    nbest = 1,       # 1 best model for each number of predictors
                    nvmax = NULL,    # NULL for no limit on number of variables
                    force.in = NULL, force.out = NULL,
                    method = "exhaustive")
lot(leaps, scale='adjr2')
## actually, the plot sorts the best models from lower ajusted R2 value to higher adjr2
## In this case it shows that the highest adjr2 value associated with the lowest number
## of variables is the 8th (starting from bottom). I however don't know which R commands
## to use to identify it, so I had to do visually using the plot
## I tried:
# summary.leap <- summary(leaps)
# summary.leap$which[summary.leap$adjr2,]
# but this outcomes the model with highest adjr2 value (rather than the 'best'
# ratio of adjr2 and number of variables)

###
# Training
###


if(Mac) {
   # Set up parallel processors for Mac
   library('doMC')
   registerDoMC(cores = detectCores() - ignorecores)
} else if(Windows) {
   # Set up parallel processors for Windows
   library(doParallel)
   cl = makeCluster(detectCores() - ignorecores)
   registerDoParallel(cl)
}

if(!skipTraining) {
   begin = Sys.time()
   
   # 1- Trees (rpart method)
   message("Training model (rpart)...")
   fitTREE <- train(Survived ~ Age + Pclass + Sex + Embarked + Parch + Fare + SibSp, data=training, method='rpart', preProcess=c("center", "scale"))
   
   # 2- Random Forest
   message("Training model (Random Forests)...")
   fitControlRF <- trainControl(method = "cv", number = 15)
   fitRF <- train(Survived ~ Age + Pclass + Sex + Embarked + Parch + Fare + SibSp, data=training, method='rf', preProcess=c("center", "scale"), 
                  trControl = fitControlRF, prox=TRUE, verbose = FALSE)
   
   # 3- GBM
   message("Training model (GBM)...")
   fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 20)
   fitGBM <- train(Survived ~ Age + Pclass + Sex + Embarked + Parch + Fare + SibSp, data = training, method = 'gbm', trControl = fitControl, 
                   verbose = FALSE)
   
   # 4- Support Vector Machine (SVM)
   message("Training model (SVM)...")
   fitSVM <- train(Survived ~ Age + Pclass + Sex + Embarked + Parch + Fare + SibSp, data=training, method = 'svmLinear', trControl = fitControl)
   
   end = Sys.time()
   message(sprintf("Total time to fit models, with %d cores: %d minutes %d seconds.", 
                   detectCores() - ignorecores, 
                   floor(as.numeric(end-begin, units="mins")), 
                   floor(as.numeric(end-begin, units="secs")) %% 60))
}
   
###
# Cross-validation
###

method <- c('trees', 'RF', 'GBM', 'SVM')

# in-sample error
inSample.Accuracy <- c(max(fitTREE$results[,2]), 
                       max(fitRF$results[,2]), 
                       max(fitGBM$results[,4]),
                       max(fitSVM$results[,2])
)
inSample.Error <- c(1 - max(fitTREE$results[,2]), 
                    1 - max(fitRF$results[,2]), 
                    1 - max(fitGBM$results[,4]),
                    1 - max(fitSVM$results[,2])
)

# TREE prediction on validation set
predictedValuesTREE <- predict(fitTREE, validation)
confM.TREE <- confusionMatrix(validation$Survived, predictedValuesTREE)

# RF prediction on validation set
predictedValuesRF <- predict(fitRF, validation)
confM.RF <- confusionMatrix(validation$Survived, predictedValuesRF)

# GBM prediction on validation set
predictedValuesGBM <- predict(fitGBM, validation)
confM.GBM <- confusionMatrix(validation$Survived, predictedValuesGBM)

# SVM prediction on validation set
predictedValuesSVM <- predict(fitSVM, validation)
confM.SVM <- confusionMatrix(validation$Survived, predictedValuesSVM)

outSample.Accuracy <- c(as.numeric(confM.TREE$overall[1]),
                        as.numeric(confM.RF$overall[1]),
                        as.numeric(confM.GBM$overall[1]),
                        as.numeric(confM.SVM$overall[1])
)
outSample.Error <- c(1 - as.numeric(confM.TREE$overall[1]),
                     1 - as.numeric(confM.RF$overall[1]),
                     1 - as.numeric(confM.GBM$overall[1]),
                     1 - as.numeric(confM.SVM$overall[1])
)
errors.df <- data.frame(method = method, 
                        inSample.Acc = inSample.Accuracy, 
                        inSample.Err = inSample.Error, 
                        outSample.Acc = outSample.Accuracy, 
                        outSample.Err = outSample.Error, 
                        stringsAsFactors=FALSE)
errors.df

declare <- NULL
declare <- rbind(declare, paste('Highest in-sample accuracy: ', 
                                round(as.vector(errors.df[which.max(errors.df$inSample.Acc), 2]), 4)
                                *100, '% with method ', as.vector(errors.df[which.max(errors.df$inSample.Acc), 1]),
                                '.\n', sep=''))
declare <- rbind(declare, paste('Highest out-sample accuracy: ',
                                round(as.vector(errors.df[which.max(errors.df$outSample.Acc), 4]), 4)
                                *100, '% with method ', as.vector(errors.df[which.max(errors.df$outSample.Acc), 1]),
                                '.', sep=''))
cat(declare)

###
# Prediction
###
if(!skipTraining) {   
   # Combine predicted values with validation answers.
   modelPreds = data.frame(tree=predictedValuesTREE, rf=predictedValuesRF, gbm=predictedValuesGBM, svm=predictedValuesSVM, Survived=validation$Survived)
   
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