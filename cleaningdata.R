# Clear environments.
rm(list=ls(), inherits=T) 

# Read files.
data = read.csv("data/raw/train.csv", header=T, 
                colClasses=c("integer", "factor", "factor", "character", 
                             "factor", "numeric", "integer", "integer", 
                             "character", "numeric", "character", "factor"))


final.test = read.csv("data/raw/test.csv")

names(final.test)
###
# Pre-processing
###

#aggregate(Survived~Embarked,subset(data,Survived=="Y"),length)
data[which(data$Embarked==''),]$Embarked = "C"

levels(data$Survived) = c("N", "Y")
levels(data$Sex) = c("F", "M")


levels(final.test$Sex) = c("F", "M")
final.test$Pclass = as.factor(final.test$Pclass)

source("partitionData.R")
source("activateParallel.R")
activateParallel(0)
partitionData(na.omit(data),"Age",F)

set.seed(1245)
# Define formula to use for training.
formula = formula(Age ~ Fare^2+ Pclass + SibSp + Parch + Embarked)
fitControlRF <- trainControl(method = "cv", number =75)
fitRF <- train(formula, data=training, method='rf',
               trControl = fitControlRF,prox=T  ,verbose = FALSE)
ggplot(fitRF)


missClass = function(values,prediction){sum((prediction/values-1)>0.20)/length(values)}
missClass(testing$Age,predict(fitRF, testing))

data[which(is.na(data$Age)),]$Age <- round(predict(fitRF, data[which(is.na(data$Age)),]))

final.test[which(is.na(final.test$Age)),]$Age <- round(predict(fitRF, final.test[which(is.na(final.test$Age)),]))

summary(data)


# Remove  Name, Ticket, Cabin, and incomplete cases.
data <- data[, c(-4, -9, -11)]
final.test <- final.test[, c(-3, -8, -10)]

#names(data)
#write the new file in data/cleaned
write.csv(data,"data/cleaned/train.csv",row.names = F)
write.csv(final.test,"data/cleaned/test.csv",row.names = F)