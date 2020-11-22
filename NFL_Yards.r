library(corrplot)
library(tidyverse)
library(dummies)
library(caTools)
library(randomForest)
library(caret)
library(devtools)
library("pls")
library(glmnet)
library(dplyr)
library(factoextra)
mydata <- read.csv(file = 'Final_NFL.csv') # Reading a csv into a dataframe

mydata$PlayerHeight <- gsub('-','.',mydata$PlayerHeight)
mydata$PlayerHeight <- as.numeric(as.character(mydata$PlayerHeight))
mydata$PlayerHeight <- mydata$PlayerHeight*12 #Converting player height into inches from feet-inch

#Dropping unnecessary columns which is text
mydata$GameId <- NULL
mydata$PlayId <- NULL
mydata$X.1 <- NULL
mydata$TimeHandoff <- NULL
mydata$TimeSnap <-NULL
mydata$GameClock <- NULL
mydata$FieldPosition <- NULL
mydata$HomeTeamAbbr <- NULL
mydata$VisitorTeamAbbr <- NULL
mydata$GameWeather <- NULL
mydata$Team <- NULL
mydata$PossessionTeam <- NULL
mydata$OffenseFormation <- NULL
mydata$PlayDirection <- NULL
mydata$Position <- NULL
mydata$WindDirection <- NULL
mydata$Turf <- NULL
mydata$Week <- NULL
mydata$Season <- NULL 

sum(is.na(mydata)) #Checking wether the data has null values or not.


library(corrplot)
library(RColorBrewer)
corrplot(cor(mydata), type="upper", order="hclust",
         col=brewer.pal(n=8, name="RdYlBu")) #Correlation Plot

set.seed(111)
split=sample.split(mydata$Yards,SplitRatio = 2/3)
nfl_train=subset(mydata,split==TRUE)
nfl_test=subset(mydata,split==FALSE)

##########################################   PLS   #########################################

data = subset(nfl_train,select = -Yards)

ctrl <- trainControl(method = "cv", number = 10)

# using plsr with 3 components
plsFit = plsr(nfl_train$Yards ~ . , data = data)

## try on test data (316)
plsPred = predict(plsFit, nfl_test, ncomp = 3)
plsValue = data.frame(obs = nfl_test$Yards, pred = plsPred[,,1])
defaultSummary(plsValue)

pls <- train(x = data, y = nfl_train$Yards, method = "pls")
pls

pls <- train(x = data, y = nfl_train$Yards, method = "pls", tuneGrid = expand.grid(ncomp = 3), trControl = ctrl)
pls

testpls <- data.frame(obs = nfl_test$Yards, pred = predict(pls, nfl_test))
defaultSummary(testpls)

plsTune <- train(x = data, y = nfl_train$Yards, method = "pls", tuneGrid = expand.grid(ncomp = 1:20), trControl = ctrl)
plsTune

plsResamples <- plsTune$results
plsResamples$Model <- "PLS"


xyplot(RMSE ~ ncomp,
       data = plsResamples,
       #aspect = 1,
       xlab = "Number of Components",
       ylab = "RMSE (Cross-Validation)",      
       col = c("blue","red"),
       auto.key = list(columns = 2),
       groups = Model,
       type = c("o", "g"))

# variable importance - PLS  Fig. 6.14
plsImp <- varImp(plsTune, scale = FALSE)
plot(plsImp, top = 25, scales = list(y = list(cex = .95)))


#############################################   Lasso, Ridge, Pca, Pcr   ###########################

x <- model.matrix(Yards~.,nfl_train)[,-1]
y <- nfl_train$Yards
dim(x)
summary(x)

#Lasso without tuning parameters
model <- glmnet(x,y,alpha = 1,lambda = NULL)
x.test <- model.matrix(Yards~.,nfl_test)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
data.frame(
  RMSE = RMSE(predictions,nfl_test$Yards)
)

#ridge without tuning parameters
model <- glmnet(x,y,alpha = 0,lambda = NULL)
x.test <- model.matrix(Yards~.,nfl_test)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
data.frame(
  RMSE = RMSE(predictions,nfl_test$Yards)
)



#Ridge using grid
lambda = 10^seq(-3,3,length=100)
set.seed(540)
ridge <- train(Yards~.,data = nfl_train,method = "glmnet",
               trControl = trainControl(method = "repeatedcv",number = 10,repeats = 5),
               preProc=c("center","scale"),
               tuneGrid = expand.grid(alpha = 0, lambda = lambda))
coef(ridge$finalModel,ridge$bestTune$lamdba)
predictions <- ridge %>% predict(nfl_test)
postResample(predictions, nfl_test$Yards)

#Lasso using grid
lambda = 10^seq(-3,3,length=100)
set.seed(540)
lasso <- train(Yards~.,data = nfl_train,method = "glmnet",
               trControl = trainControl(method = "repeatedcv",number = 10,repeats = 5),
               preProc=c("center","scale"),
               tuneGrid = expand.grid(alpha = 1, lambda = lambda))
coef(lasso$finalModel,lasso$bestTune$lamdba)
predictions <- lasso %>% predict(nfl_test)
postResample(predictions, nfl_test$Yards)

models <- list(ridge = ridge,lasso = lasso)
resamples(models)%>%summary(metric="RMSE")
#PCR
set.seed(540)
model <- train(Yards~.,data = nfl_train,method = "pcr",
               trControl = trainControl("cv",number = 10),
               preProc = c("center","scale"),tuneLength = 36)
plot(model)
model$bestTune
summary(model$finalModel)

#PCA
NFL.PCA <- prcomp(mydata,center = TRUE,scale = TRUE)
fviz_eig(NFL.PCA)

fviz_pca_var(NFL.PCA,col.var = "contrib",
             gradient.cols = c("#00AFBB","#E7B800","#FC4E07"),
             repel = TRUE)

#Standard deviation
std_dev <- NFL.PCA$sdev

#variance
NFL_VAR <- std_dev^2
NFL_VAR

summary(NFL.PCA)




#############################################   Random Forest  ######################################


library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 3) # convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(666)
rf <- randomForest(Yards~.,data=nfl_train,ntree=200,mtry=7,importance=TRUE)
stopCluster(cluster)
registerDoSEQ()

min(rf$mse) #minimum mse value for rf model is 1.690553

rf$importance #Importance values for each predictors

varImpPlot(rf) #Variable importance plot

plot(rf) # Error vs Trees plot

measure_importance(rf)

rf_predict = predict(rf,newdata =nfl_test)
rf_test_predicted_results = cbind.data.frame(nfl_test,rf_predict)
rf_test_rmse = sqrt(mean((rf_predict-nfl_test$Yards)^2))
rf_test_rmse #test set rmse is 1.257535

#############################################   XGBOOST   #######################################

# XGBOOST with 10-Fold Cross Validation

library(xgboost)
library(caret)
start_time = Sys.time()
set.seed(222)
ctrl <- trainControl(method="cv",number = 10)
xgb.cv <- train(Yards ~ ., data=nfl_train, method="xgbTree",trControl=ctrl,metric="RMSE",verbose=1)

end_time = Sys.time()

xgb.cv$bestTune 

# Tuning parameter 'gamma' was held constant at a value of 0
# Tuning parameter 'min_child_weight' was
# held constant at a value of 1
# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were nrounds = 150, max_depth = 3, eta = 0.4, gamma = 0,
# colsample_bytree = 0.8, min_child_weight = 1 and subsample = 0.75.



library(xgboost)
library(caret)
set.seed(123)
trainm <- model.matrix(Yards ~ .,data = nfl_train) #Creating a matrix of train data
train_label <- nfl_train$Yards

train_matrix <- xgb.DMatrix(data =as.matrix(trainm),label=as.numeric(train_label)) #converting matrix to xgb.DMatrix

testm <- model.matrix(Yards ~., data=nfl_test) ##Creating a matrix of test data
test_label <- nfl_test$Yards
test_matrix <- xgb.DMatrix(data=as.matrix(testm), label=as.numeric(test_label)) # #converting matrix to xgb.DMatrix

set.seed(333)
parameterGrid <- expand.grid(eta=0.4,colsample_bytree=0.8,max_depth=3,gamma=0,min_child_weight=1,subsample=0.75) #parameter grid which is best from the cross validation
xgb_model <- xgboost(data = train_matrix,metric="RMSE",nrounds=150,tuneGrid=parameterGrid,importance=TRUE) #Training xgb model on xgb.DMatrix on Train data


xgb_model$evaluation_log #rmse values for each iteration
xgb_model$feature_names #predictors used in this model
xgb_model$nfeatures # count of predictors which is 20

imp <- xgb.importance(colnames(train_matrix),model = xgb_model) #importance values for each predictors.
xgb.plot.importance(imp,rel_to_first = TRUE, xlab="Relative Importance")

xgb_predict <- predict(xgb_model,newdata = test_matrix)

xgb_test_predicted_results <- cbind(test_label,xgb_predict) #XGBOOST predicted results combined as a column to test data

xgb_test_rmse <- sqrt(mean((test_label - xgb_predict)^2))

xgb_test_rmse # test rmse is 4.54339


save.image(file = "Katiki_Nimmala_Pulipaka_Pulluri_Valeti_Vulupala_NFL.RData")


