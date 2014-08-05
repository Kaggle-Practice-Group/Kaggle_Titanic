data <- read.csv("data/train.csv", header=T, 
                    colClasses=c("integer", "factor", "factor", "character", 
                                 "factor", "numeric", "integer", "integer", 
                                 "character", "numeric", "character", "factor"))

## Set common seed for all team members for reproducibility purposes
set.seed(3846)

library('caret')
library('plyr') 

###
# Pre-processing
###

levels(data$Survived) = c("N", "Y")
levels(data$Sex) = c("F", "M")

# populate empty cells with NAs
data[data == ''] <- NA
# define function to count NAs
nmissing <- function(x) sum(is.na(x))
# use colwise from plyr package to apply the function to columns
missingNA <- colwise(nmissing)(data)
# identify columns that have more than 50% NAs
indexMissingNA <- missingNA >= dim(data)[1]/2
# clean data base on such index
new.data <- data[, !indexMissingNA]
# manually remove variables, such as: 'Name', 'Ticket', ...
new.data <- new.data[, c(-4, -9, -11)]

###
# Exploratory Analysis
###

###
# Sampling
###

# Out of the original dataset, I create a 90% sample for training and 10% sample
# for our validation. If our validation gives good result, then we can apply the 
# Machine Learning algorithm to the 'test.csv' dataset provided

indexSample <- createDataPartition(y=new.data$Survived, p=.9, list=FALSE)
training <- new.data[indexSample,]
validation <- new.data[-indexSample,]
validation <- na.omit(validation)

###
# Run several ML algorithms (training)
###

# load library doMC that will force R to use all cores of Mac in parallel
library('doMC') 
registerDoMC(cores = 2) # set number of CPU cores
##

library('randomForest')
# 1- Trees (rpart method)
fitTREE <- train(Survived ~ ., data=training, method='rpart', preProcess=c("center", "scale"))

# 2- Random Forest
fitControlRF <- trainControl(method = "cv", number = 15)
fitRF <- train(Survived ~ ., data=training, method='rf', preProcess=c("center", "scale"), 
               trControl = fitControlRF, prox=TRUE, verbose = FALSE)

# 3- GBM
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 20)
fitGBM <- train(Survived ~ ., data = training, method = 'gbm', trControl = fitControl, 
                verbose = FALSE)

# 4- Support Vector Machine (SVM)
fitSVM <- train(Survived ~., data=training, method = 'svmLinear', trControl = fitControl)

#### In- / Out-Sample errors and cross-validation

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
# Predict test cases
###

test <- read.csv("data/test.csv", header=T, 
                 colClasses=c("integer", "factor", "character", 'factor', 
                              'numeric', 'integer', 'integer', 'character',
                              'numeric', 'character', 'factor'))
test <- test[, c(-3, -8, -10)]
levels(test$Survived) = c("N", "Y")
levels(test$Sex) = c("F", "M")

predictedTEST <- predict(fitRF, test)
predictedTEST
