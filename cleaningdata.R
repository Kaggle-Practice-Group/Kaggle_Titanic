# Clear environments.
rm(list=ls(), inherits=T) 

# Read files.
data = read.csv("data/raw/train.csv")
                #, header=T, 
                #colClasses=c("integer", "factor", "integer", "character", 
                #             "factor", "numeric", "integer", "integer", 
                #             "character", "numeric", "character", "factor"))
#names(data)

final.test = read.csv("data/raw/test.csv")


# Cleaning Names
#Replace Mlle to Miss
data$Name<- gsub("Mlle","Miss",data$Name)
final.test$Name <- gsub("Mlle","Miss",final.test$Name)

# Replace Countess to Lady
data$Name<- gsub("Countess","Lady",data$Name)
final.test$Name <- gsub("Countess","Lady",final.test$Name)

# Replace Mme to Mrs
data$Name<- gsub("Mme","Mrs",data$Name)
final.test$Name <- gsub("Mme","Mrs",final.test$Name)

# Replace Jonkheer to Miss
data$Name<- gsub("Jonkheer","Miss",data$Name)
final.test$Name <- gsub("Jonkheer","Miss",final.test$Name)

# Replace Dona to Mrs
data$Name<- gsub("Dona","Lady",data$Name)
final.test$Name <- gsub("Dona","Lady",final.test$Name)

# Replace Ms to Mrs
data$Name<- gsub("Ms","Miss",data$Name)
final.test$Name <- gsub("Ms","Miss",final.test$Name)

# Replace Don to Sir
data$Name<- gsub("Don","Sir",data$Name)
final.test$Name <- gsub("Don","Sir",final.test$Name)

#Add title to train
data<-cbind(data,data$Name)
names(data) <- c(names(data)[1:(dim(data)[2]-1)],"Title")
data$Title <- as.character(data$Title)

for (i in 1:dim(data)[1])
{
  tmp <- strsplit(data$Name[i]," ")[[1]]  
  data$Title[i]<-tmp[grep("[A-z][a-z]*\\.",tmp)]
}

#Add title to test
final.test<-cbind(final.test,final.test$Name)
names(final.test) <- c(names(final.test)[1:(dim(final.test)[2]-1)],"Title")
final.test$Title <- as.character(final.test$Title)

for (i in 1:dim(final.test)[1])
{
  tmp <- strsplit(final.test$Name[i]," ")[[1]]  
  final.test$Title[i]<-tmp[grep("[A-z][a-z]*\\.",tmp)]
}


###
# Pre-processing
###

#aggregate(Survived~Embarked,subset(data,Survived=="Y"),length)
data[which(data$Embarked==''),]$Embarked = "C"

data$Survived <- as.factor(data$Survived)
levels(data$Survived) = c("N", "Y")
data$Sex <- as.factor(data$Sex)
levels(data$Sex) = c("F", "M")

final.test$Sex <- as.factor(final.test$Sex)
levels(final.test$Sex) = c("F", "M")

source("partitionData.R")
source("activateParallel.R")
activateParallel(0)
partitionData(na.omit(data),"Age",F)
unique(data$Title)
set.seed(1245)

# Define formula to use for training.
formula = formula(Age ~ Fare+Title+ Pclass + SibSp +Parch )
fitControlRF <- trainControl(method = "cv", number =25)
fitRF <- train(formula, data=training, method='rf',
               trControl = fitControlRF,prox=T  ,verbose = FALSE,importance=T)
ggplot(fitRF)
varImp(fitRF)

missClass = function(values,prediction){sum((prediction/values-1)>0.20)/length(values)}
missClass(testing$Age,predict(fitRF, testing))

data[which(is.na(data$Age)),]$Age <- round(predict(fitRF, data[which(is.na(data$Age)),]))
final.test[which(is.na(final.test$Age)),]$Age <- round(predict(fitRF, final.test[which(is.na(final.test$Age)),]))


# Remove  Name, Ticket, Cabin, and incomplete cases.
data <- data[, c(-4, -9, -11)]
final.test <- final.test[, c(-3, -8, -10)]

#names(data)
#write the new file in data/cleaned
write.csv(data,"data/cleaned/train.csv",row.names = F)
write.csv(final.test,"data/cleaned/test.csv",row.names = F)