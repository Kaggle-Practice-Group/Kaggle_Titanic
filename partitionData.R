#' @title Partition a data frame in a training set and testing set and eventually a inValidation set
#' 
#' @description
#' This function use the createDataPartition from the caret package to create sub partition of the data. By default it split the data in those proportion: 75% training, 25% testing.
#' it's possible to include a validation set by setting the use.validation parameter to TRUE. In this case the data will be split as follow 45% training, 30% validation, 25% testing.
#' 
#' @param data Data frame to partition
#' @param column.partition Column name existing in the data frame that will be used for the split
#' @param use.validation Define if a validation set is needed.(Default value is FALSE)
#' @return create global variables names as follow : training,testing and validation (if use.validation = T)
require(caret)
partitionData <- function(data,column.partition,use.validation= F)
{
  # Split data: 75% training, 25% testing.
  inTrain <- createDataPartition(data[[column.partition]],
                                 p=.85,
                                 list=FALSE)
  training <<- data[inTrain,]
  testing <<- data[-inTrain,]
  
  # If using validation set, split data: 45% training, 30% validation, 25% testing.
  if(use.validation) {
    inValidation <<- createDataPartition(training[[column.partition]],
                                         p=.4,
                                         list=FALSE)   
    validation <<- training[inValidation,]
    training <<- training[-inValidation,]
  }  
}

