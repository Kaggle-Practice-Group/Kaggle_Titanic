training = read.csv("data/train.csv", header=T, 
  colClasses=c("integer", "factor", "factor", "character", "factor", "numeric", "integer", "integer", "character", "numeric", "character", "factor"))

###
# Pre-processing
###

levels(training$Survived) = c("N", "Y")
levels(training$Sex) = c("F", "M")

###
# Exploratory Analysis
###

