install.packages("dummies")
library(dummies)
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



test_princ <- as.data.frame(test_princ)

#select the first 60 components
test_princ <- test_princ[,1:60]



################################ XGB model #######################################3
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)


#General parameters
# silent : The default value is 0. You need to specify 0 for printing running messages, 1 for silent mode
# booster : The default value is gbtree. You need to specify the booster to use: gbtree (tree based) or gblinear (linear function).
# nthread[default=maximum cores available] : Activates parallel computation. Generally, people don't change it as using maximum cores leads to the fastest computation


#  tree specific parameters

# nrounds[default=100]:It controls the maximum number of iterations. For classification, 
# it is similar to the number of trees to grow.Should be tuned using CV

# eta[default=0.3][range: (0,1)] (learning rate): The default value is set to 0.3. You need to specify step size shrinkage used in update to 
# prevent overfitting. After each boosting step, we can directly get the weights of new features and 
# eta actually shrinks the feature weights to make the boosting process more conservative. The range 
# is 0 to 1. Low eta value means model is more robust to overfitting.
# Simplified the relationship of learning rate and the number of trees as an
# approximate ratio: learning rate = [2-10]/trees.

# gamma : It controls regularization (or prevents overfitting). The optimal value of gamma depends on the 
# data set and other parameter values.Higher the value, higher the regularization. 
# Regularization means penalizing large coefficients which don't improve the model's performance. 
# default = 0 means no regularization.
# Tune trick: Start with 0 and check CV error rate.
# If you see train error >>> test error, bring gamma into action. Higher the gamma, lower the 
# difference in train and test CV. If you have no clue what value to use, use gamma=5 and see the performance.
# Remember that gamma brings improvement when you want to use shallow (low max_depth) trees.

# max_depth[default=6][range: (0,Inf)] : The default value is set to 6. You need to specify the maximum depth of a tree.
# The range is 1 to ???.Should be tuned using CV

# min_child_weight[default=1][range:(0,Inf)] : In regression, it refers to the minimum number of instances required in a child node. In classification, if the leaf node has a minimum sum of instance weight (calculated by second order partial derivative) lower than min_child_weight, the tree splitting stops.
# In simple words, it blocks the potential feature interactions to prevent overfitting. 
# Should be tuned using CV.
# Tunes minimum leaf weight as an approximate ratio of 3 over the percentage of the number of rare events.

# max_delta_step : The default value is set to 0. Maximum delta step we allow each tree's weight estimation
# to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can 
# help making the update step more conservative. Usually this parameter is not needed, but it might help in 
# logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
# The range is 0 to ???.

# subsample : The default value is set to 1. You need to specify the subsample ratio of the training instance.
# Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this 
# will prevent overfitting. The range is 0 to 1.

# colsample_bytree : The default value is set to 1. You need to specify the subsample ratio of columns when
# constructing each tree.The range is 0 to 1.

# Learning  Task Parameters

# base_score : The default value is set to 0.5 . You need to specify the initial prediction score
# of all instances, global bias.

# objective : reg:linear - for linear regression
# binary:logistic - logistic regression for binary classification. It returns class probabilities
# multi:softmax - multiclassification using softmax objective. It returns predicted class labels. 
# It requires setting num_class parameter denoting number of unique prediction classes.
# multi:softprob - multiclassification using softmax objective. It returns predicted class probabilities.

# eval_metric : Available error functions are as follows:
# mae - Mean Absolute Error (used in regression)
# Logloss - Negative loglikelihood (used in classification)
# AUC - Area under curve (used in classification)
# RMSE - Root mean square error (used in regression)
# error - Binary classification error rate [#wrong cases/#all cases]
# mlogloss - multiclass logloss (used in classification)

# seed : As always here you specify the seed to reproduce the same set of outputs.


# We use MLR to perform the extensive parametric search and try to obtain optimal accuracy.
#using one hot encoding 
labels <- Train3$Income.Group 
ts_label <- Test1$Income.Group
new_tr <- model.matrix(~.+0,data = Train3[,-1]) 
new_ts <- model.matrix(~.+0,data = test_princ)

#convert factor to numeric 
labels <- ifelse(Train3$Income.Group%in%levels(Train3$Income.Group)[2],1,0)
ts_label <- ifelse(ts_label%in%levels(ts_label)[2],1,0)
#For xgboost, we'll use xgb.DMatrix to convert data table into a matrix (most recommended):
#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)
#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6,
               min_child_weight=1, subsample=1, colsample_bytree=1)

# Using the inbuilt xgb.cv function, let's calculate the best nround for this model.
# In addition, this function also returns CV error, which is an estimate of test error

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, 
                 print.every.n = 10,
                 early.stop.round = 20, maximize = F)
##best iteration = 10
# The model returned lowest error at the 23th (nround) iteration. Also, if you noticed the running messages
# in your console, you would have understood that train and test error are following each other. 

# We'll use this insight in the following code. Now, we'll see our CV error
min(xgbcv$evaluation_log$test_error_mean)
#0.1663742
#CV accuracy (100-16.63742)=83.36258%

# Let's calculate our test set accuracy and determine if this default model makes sense
#first default - model training
xgb1 <- xgb.train (params = params, data = dtrain
                   ,nrounds = 10, watchlist = list(val=dtest,train=dtrain),
                   print_every_n = 10, early_stop_round = 10, maximize = F , eval_metric = "error")

#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

# I've used 0.5 as my cutoff value for predictions. 
# We can calculate our model's accuracy using confusionMatrix() function from caret package

confusionMatrix (xgbpred, ts_label)

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20]) 

# Let's proceed to the random / grid search procedure and attempt to find better accuracy. 
# From here on, we'll be using the MLR package for model building. A quick reminder, 
# the MLR package creates its own frame of data, learner as shown below. Also, keep in mind that 
# task functions in mlr doesn't accept character variables. Hence, we need to convert them to factors
# before creating task


#convert characters to factors(NO need to run these codes as we dont have any factor variable in our data)
# fact_col <- colnames(train)[sapply(train,is.character)]
# 
# for(i in fact_col) set(train,j=i,value = factor(train[[i]]))
# for (i in fact_col) set(test,j=i,value = factor(test[[i]]))

#install.packages("mlr",repos="http://cran.us.r-project.org")
library(mlr)

Train3$Income.Group <- as.factor(Train3$Income.Group)
test_princ <- cbind(test_princ,Income.Group=Test1$Income.Group)
test_princ$Income.Group<-ifelse(test_princ$Income.Group%in%levels(test_princ$Income.Group)[2],"1","0")
#create tasks
traintask <- makeClassifTask (data = Train3,target = "Income.Group")
testtask <- makeClassifTask (data = test_princ,target = "Income.Group")

# we'll set the learner and fix the number of rounds and eta
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), 
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1),
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

# With stratify=T, we'll ensure that distribution of target class is maintained in the resampled data sets.
# If you've noticed above, in the parameter set, I didn't consider gamma for tuning. Simply because during 
# cross validation, we saw that train and test error are in sync with each other. Had either one of them been 
# dragging or rushing, we could have brought this parameter into action.

# We'll use random search to find the best parameters. In random search, we'll build 10 models with different
# parameters, and choose the one with the least error.

#search strategy
ctrl <- makeTuneControlRandom(maxit = 20L)

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc,
                     par.set = params, control = ctrl, show.info = T)

#Result: booster=gbtree; max_depth=3; min_child_weight=4.69; 
# subsample=0.665; colsample_bytree=0.89 : acc.test.mean=0.835
mytune$y 
#acc.test.mean 
#0.8345035 

#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)

#predict model
xgpred <- predict(xgmodel,testtask)

confusionMatrix(xgpred$data$response,xgpred$data$truth)
