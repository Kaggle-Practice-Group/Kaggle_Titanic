Mac = F
Windows = T

data <- read.csv("data/train.csv", header=T, 
                    colClasses=c("integer", "factor", "factor", "character", 
                                 "factor", "numeric", "integer", "integer", 
                                 "character", "numeric", "character", "factor"))

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
new.data <- data[, c(-4, -9, -11)]

###
# Partition Data
###

# Create data partition with 90% training data.
indexSample <- createDataPartition(y=new.data$Survived, p=.9, list=FALSE)
training <- new.data[indexSample,]
attach(training)
validation <- new.data[-indexSample,]
validation <- na.omit(validation)

###
# Parameter Pruning
###

# Stepwise function finds that the best formula is Survived ~ SibSp + Age + Pclass + Sex
glm = glm(Survived ~ Fare + Embarked + Pclass + Sex + Age + SibSp + Parch, 
          data=training, family="binomial")
aic = step(glm, direction="both")

if(T) {

###
# Training
###

if(Mac) {
   # Set up parallel processors for Mac
   library('doMC')
   ignorecores = 1 # Number of cores to NOT dedicate.
   registerDoMC(cores = detectCores() - ignorecores) # set number of CPU cores
}
else if(Windows) {
   # Set up parallel processors for Windows
   library(doParallel)
   ignorecores = 1 # Number of cores to NOT dedicate.
   cl = makeCluster(detectCores() - ignorecores)
   registerDoParallel(cl)
}

begin = Sys.time()

# 1- Trees (rpart method)
message("Training model (rpart)...")
fitTREE <- train(Survived ~ SibSp + Age + Pclass + Sex, data=training, method='rpart', preProcess=c("center", "scale"))

# 2- Random Forest
message("Training model (Random Forests)...")
fitControlRF <- trainControl(method = "cv", number = 15)
fitRF <- train(Survived ~ SibSp + Age + Pclass + Sex, data=training, method='rf', preProcess=c("center", "scale"), 
               trControl = fitControlRF, prox=TRUE, verbose = FALSE)

# 3- GBM
message("Training model (GBM)...")
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 20)
fitGBM <- train(Survived ~ SibSp + Age + Pclass + Sex, data = training, method = 'gbm', trControl = fitControl, 
                verbose = FALSE)

# 4- Support Vector Machine (SVM)
message("Training model (SVM)...")
fitSVM <- train(Survived ~ SibSp + Age + Pclass + Sex, data=training, method = 'svmLinear', trControl = fitControl)

end = Sys.time()
message(sprintf("Total time to fit models, with %d cores: %d minutes %d seconds.", 
                detectCores() - ignorecores, 
                floor(as.numeric(end-begin, units="mins")), 
                floor(as.numeric(end-begin, units="secs")) %% 60))

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
}

###
# Predict test cases
###
if(F) {
   test <- read.csv("data/test.csv", header=T, 
                    colClasses=c("integer", "factor", "character", 'factor', 
                                 'numeric', 'integer', 'integer', 'character',
                                 'numeric', 'character', 'factor'))
   test <- test[, c(-3, -8, -10)]
   levels(test$Survived) = c("N", "Y")
   levels(test$Sex) = c("F", "M")
   
   predictedTEST <- predict(fitRF, test)
   predictedTEST
}