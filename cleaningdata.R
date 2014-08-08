# Clear environments.
rm(list=ls(), inherits=T) 

# Read files.
data = read.csv("data/raw/train.csv", header=T, 
                colClasses=c("integer", "factor", "factor", "character", 
                             "factor", "numeric", "integer", "integer", 
                             "character", "numeric", "character", "factor"))


###
# Pre-processing
###

levels(data$Survived) = c("N", "Y")
levels(data$Sex) = c("F", "M")

# Remove  Name, Ticket, Cabin, and incomplete cases.
data <- data[complete.cases(data), c(-4, -9, -11)]

#write the new file in data/cleaned
write.csv(data,"data/cleaned/train.csv",row.names = F)