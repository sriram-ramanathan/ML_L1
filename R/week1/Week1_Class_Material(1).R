options(scipen =999)
library(readxl)
setwd ("D:\\Personal Study\\COE\\Machine Learning- L1\\Improved Class Content Creation")

#Load excel file into dataframe
class1_data = read_excel("deliveries_1.xlsx")

#Data types for different variables
for (i in 1:ncol(class1_data)) {
  aa[i] = class(class1_data[,i])
  
}

aa = cbind(colnames(class1_data),aa)

#find number of rows and columns

dim(class1_data)
nrow(class1_data)
ncol(class1_data)

#rename column names
class1_data_rename = rename(class1_data, c(match_id = "match_no",
                                           bowling_team = "team_bowling"))

#Statistical measures(min,max,mean,median,mode,standard deviation,quartile values) of single column 
#some columns , all columns

summary(class1_data$match_id)
sapply(class1_data,mean)
max(class1_data$noball_runs)
median(class1_data$match_id)

#subsetting data
#Row Subset
class1_data_subset = subset(class1_data, match_id ==1)
#Column Subset
class1_data_subset_column = class1_data[,1:5]

#Plotting histogram
hist(class1_data$ball)



