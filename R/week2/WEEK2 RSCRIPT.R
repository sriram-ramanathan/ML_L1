install.packages("rpart")
library(rpart)
install.packages("dummies")
library(dummies)
install.packages("rpart.plot")
library(rpart.plot)
#Remove Scientific notations
options(scipen=999)

# loading Train and Test data
Train <-read.csv("C:/Users/jasksing/Desktop/ML1 COE/train.csv")
Test <- read.csv("C:/Users/jasksing/Desktop/ML1 COE/test.csv")
full_data <- rbind(Train,Test)


# label ecoding for ordinal variables
names(full_data)

# Education is the only ordinal variable
unique(full_data$Education)

#  "10th"4         "11th"4         "12th"4         "1st-4th"6      "5th-6th"5      "7th-8th"5      "9th"4          "Assoc-acdm"3  
# "Assoc-voc"3    "Bachelors"2    "Doctorate"1    "HS-grad"3      "Masters"1      "Preschool"6    "Prof-school"2  "Some-college"2

table((full_data$Education))
# 10th         11th         12th      1st-4th      5th-6th      7th-8th          9th   Assoc-acdm    Assoc-voc 
# 933         1175          433          168          333          646          514         1067         1382 
# Bachelors    Doctorate      HS-grad      Masters    Preschool  Prof-school Some-college 
# 5355          413        10501         1723           51          576         7291 

levels(full_data$Education) <- c(4,4,4,6,5,5,4,3,3,2,1,3,1,6,2,2)
full_data$Education<- as.numeric(as.character(full_data$Education))
 
    
# Hot-encoding (dummy creation)
#"Workclass","Marital.Status","Occupation","Relationship","Race","Sex","Native.Country" 
full_data_v1 <- dummy.data.frame(full_data, 
                                 sep = "_",
                                 names = c("Workclass","Marital.Status","Occupation",
                                           "Relationship","Race","Sex","Native.Country"))


# Divide into Train and Test
Train1 <-full_data_v1[full_data_v1$ID%in%Train$ID,]
Test1 <-full_data_v1[full_data_v1$ID%in%Test$ID,]

# In order to reduce the number of variables we use PCA
in.pca <- prcomp(Train1[,-c(1,91)],
                 center = TRUE,
                 scale. = TRUE) 
in.pca

# to find the optimal number of PCs
# plot method
plot(in.pca, type = "l")
# PCA5 looks like the optimal point

# proportion of variance explained
std_dev <- in.pca$sdev
#compute variance
pr_var <- std_dev^2

#check variance of first 60 components
pr_var[1:60]

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:60]
plot(prop_varex,xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b")
# This plot suggests that 80% of variance is getting expaliend by first 60 Components

rot<-as.matrix(in.pca$rotation)
View(rot)


# creating train data with required components
Train2<-cbind(Train1[,c(1,91)],in.pca$x)
Train3 <-Train2[,2:62]

#run a decision tree
rpart.model <- rpart(Income.Group ~ .,data = Train3, method = "class")
rpart.model

#transform test into PCs
test_princ <- predict(in.pca, newdata = Test1)
test_princ <- as.data.frame(test_princ)

#select the first 60 components
test_princ <- test_princ[,1:60]

#make prediction on test data
rpart.prediction <- predict(rpart.model,test_princ)

# plot the tree
rpart.plot(rpart.model)

# Model performance
# Predicting on the test data
y_pred <- rpart.prediction[,1]


# selecting the cutoff
cutoff=.7

# converting the probability scores into 1s and 0s
pred<-c()
p<-c()
for (i in 1:length(y_pred)){
  
  if (y_pred[i]>=cutoff) { pred[i] = 1} else {pred[i] = 0 }
  
}
p<-table(pred,Test1[,"Income.Group"])
p

# Computing precision, recall and accuracy 
precision<-p[2,2]/(p[2,1]+p[2,2])
recall<-p[2,2]/(p[1,2]+p[2,2])
accuracy<-(p[1,1]+p[2,2])/(p[1,1]+p[1,2]+p[2,1]+p[2,2])
output <- as.data.frame(cbind(precision,recall,accuracy))
output

# precision    recall  accuracy
# 0.08843874 0.2320657 0.2515099


