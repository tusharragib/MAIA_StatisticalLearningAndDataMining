#*** Student's Name: S M Ragib Shahriar Islam
#*Problem name: Final Assignment_MCICN
#***Including all Libraries

library(Boruta)
library(dummies)
library(caret)
library(MASS)
library(e1071)
library(randomForest)
library(class)
library(pROC)

#***Data reading and analysis
# Reading the Training and Testing Dataset

train_dataf<-read.csv('E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Final Project/MCICN/MCICNtrain.csv', header = TRUE)
test_dataf<-read.csv('E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Final Project/MCICN/MCICNtest.csv', header = TRUE)

train_df<-train_dataf
test_df<-test_dataf

str(train_df, list.len = 1000)
#Setting the Labels as Factor type
train_df$Labels <- as.factor(train_df$Labels)

str(train_df, list.len = 1000)

#Removing the non-predictor variables
train_df = train_df[,-1]
test_df = test_df[,-1]

str(train_df, list.len = 1000)


#***Running The Boruta algorithm for Feature Selection with 2000 Iterations
#install.packages("Boruta")
set.seed(10)
boruta<- Boruta(Labels~., data = train_df, doTrace = 2, maxRuns = 2000)
print(boruta)
bor<-TentativeRoughFix(boruta)
print(bor)
plot(bor, las = 2)
getNonRejectedFormula(bor)


#*** Data split for Train and Validation(tdata and vdata)
set.seed(10)
ind<-sample(2, nrow(train_df), replace = T, prob = c(0.8,0.2))
tdata = train_df[ind ==1,]
vdata = train_df[ind ==2,]
str(tdata, list.len = 1000)
str(vdata, list.len = 1000)

#*** Creating Dummy Data for ...........
dummy.tdata = dummy.data.frame(tdata)
dummy.tdata = dummy.tdata[,-423]
dummy.vdata = dummy.data.frame(vdata)
dummy.vdata = dummy.vdata[,-423]

#***Applying Logistic Regression
#library(glm)
glm_model = glm(LabelsCN ~ Hippocampus_L + Hippocampus_R + ParaHippocampal_R + 
                  Amygdala_L + Amygdala_R + Calcarine_R + Occipital_Sup_L + 
                  Occipital_Mid_L + Cerebelum_10_L + HLA.DQA1 + NPIPB6, data = dummy.tdata, family = 'binomial',control = list(maxit = 500))
summary(glm_model)
set.seed(10)
logR.predict.vdata = predict(glm_model, dummy.vdata, type = "response")
#Converting the probability values to 1 or 0 with threshold value of 0.5
pred.logR.predict.vdata = ifelse(logR.predict.vdata>0.5, 1, 0)
# Processing the "pred.logR.predict.vdata" vector for creating confusion matrix 
length(pred.logR.predict.vdata)
for(i in 1:length(pred.logR.predict.vdata))
{
  if(pred.logR.predict.vdata[i] == 1)
    pred.logR.predict.vdata[i] <- "CN"
  else
    pred.logR.predict.vdata[i]<- "MCI"
}
pred.logR.predict.vdata <- as.factor(pred.logR.predict.vdata)
confusionMatrix(pred.logR.predict.vdata, vdata$Labels)

##Prediction for Test data.
set.seed(10)
logR.predict.test = predict(glm_model, test_df, type = "response")
pred.logR.predict.test = ifelse(logR.predict.test>0.5, 1, 0)
# Processing the "pred.logR.predict.test" vector for creating confusion matrix 
length(pred.logR.predict.test)
for(i in 1:length(pred.logR.predict.test))
{
  if(pred.logR.predict.test[i] == 1)
    pred.logR.predict.test[i] <- "CN"
  else
    pred.logR.predict.test[i]<- "MCI"
}
pred.logR.predict.test <- as.factor(pred.logR.predict.test)



#***Applying LDA 
lda_model = lda(LabelsCN ~ Hippocampus_L + Hippocampus_R + ParaHippocampal_R + 
                  Amygdala_L + Amygdala_R + Calcarine_R + Occipital_Sup_L + 
                  Occipital_Mid_L + Cerebelum_10_L + HLA.DQA1 + NPIPB6, data = dummy.tdata)
set.seed(10)
lda.predict.vdata = predict(lda_model, dummy.vdata)
# Processing the "pred.logR.predict.vdata" vector for creating confusion matrix
pred.lda_class.predict.vdata = lda.predict.vdata$class
pred.lda_class.predict.vdata = as.numeric(pred.lda_class.predict.vdata)
length(pred.lda_class.predict.vdata)
for(i in 1:length(pred.lda_class.predict.vdata))
{
  if(pred.lda_class.predict.vdata[i] == 1)
    pred.lda_class.predict.vdata[i] <- "CN"
  else
    pred.lda_class.predict.vdata[i]<- "MCI"
}
pred.lda_class.predict.vdata = as.factor(pred.lda_class.predict.vdata)

confusionMatrix(pred.lda_class.predict.vdata, vdata$Labels)

##Prediction for Test data.
set.seed(10)
lda.predict.test = predict(lda_model, test_df)
# Processing the "pred.lda_class.predict.test" vector for creating confusion matrix
pred.lda_class.predict.test = lda.predict.test$class
pred.lda_class.predict.test = as.numeric(pred.lda_class.predict.test)
length(pred.lda_class.predict.test)
for(i in 1:length(pred.lda_class.predict.test))
{
  if(pred.lda_class.predict.test[i] == 1)
    pred.lda_class.predict.test[i] <- "CN"
  else
    pred.lda_class.predict.test[i]<- "MCI"
}
pred.lda_class.predict.test = as.factor(pred.lda_class.predict.test)


#Applying Support Vector Machine(SVM)
svm_model = svm(Labels ~ Hippocampus_L + Hippocampus_R + ParaHippocampal_R + 
                  Amygdala_L + Amygdala_R + Calcarine_R + Occipital_Sup_L + 
                  Occipital_Mid_L + Cerebelum_10_L + HLA.DQA1 + NPIPB6, data = tdata, kernel = "linear", cost = 1, scale = TRUE)
summary(svm_model)
set.seed(10)
pred.svm.predict.vdata = predict(svm_model, vdata)
confusionMatrix(pred.svm.predict.vdata, vdata$Labels)
pred.svm.predict.test = predict(svm_model, test_df)


#Plotting ROC
#Plotting ROC
pred.svm.predict.vdata.bin <- as.character(pred.svm.predict.vdata)
for(i in 1:length(pred.svm.predict.vdata.bin))
{
  if(pred.svm.predict.vdata.bin[i] == "MC")
    pred.svm.predict.vdata.bin[i] <- 1
  else
    pred.svm.predict.vdata.bin[i]<- 0
}
pred.svm.predict.vdata.bin<-as.numeric(pred.svm.predict.vdata.bin)
par(pty="s") 
SVMROC <- roc(dummy.vdata$Labels ~ pred.svm.predict.vdata.bin,plot=TRUE,print.auc=TRUE,col="green",lwd =4,legacy.axes=TRUE,main="ROC Curves")



#***Applying Random Forest Classifier
RandomForest.fit = randomForest(Labels ~ Hippocampus_L + Hippocampus_R + ParaHippocampal_R + 
                                  Amygdala_L + Amygdala_R + Calcarine_R + Occipital_Sup_L + 
                                  Occipital_Mid_L + Cerebelum_10_L + HLA.DQA1 + NPIPB6, data = tdata)
set.seed(10)
RF.predict.vdata <- predict(RandomForest.fit, vdata)
confusionMatrix(RF.predict.vdata, vdata$Labels)
RF.predict.test_df = predict(RandomForest.fit, test_df)

#Plotting ROC
RF.predict.vdata.bin <- as.character(RF.predict.vdata)
for(i in 1:length(RF.predict.vdata.bin))
{
  if(RF.predict.vdata.bin[i] == "CN")
    RF.predict.vdata.bin[i] <- 1
  else
    RF.predict.vdata[i]<- 0
}
RF.predict.vdata.bin<-as.numeric(RF.predict.vdata.bin)
par(pty="s") 
RFROC <- roc(dummy.vdata$Labels ~ RF.predict.vdata.bin,plot=TRUE,print.auc=TRUE,col="green",lwd =4,legacy.axes=TRUE,main="ROC Curves")


#Making RData file
patient_id <- as.vector(train_dataf$Subject_id)
save(patient_id, pred.logR.predict.test, pred.lda_class.predict.test, pred.svm.predict.test, RF.predict.test_df, file = "E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Final Project/MCICN/file1.RData")
features <- data.frame (train_dataf$Hippocampus_L , train_dataf$Hippocampus_R , train_dataf$ParaHippocampal_R , 
                          train_dataf$Amygdala_L , train_dataf$Amygdala_R , train_dataf$Calcarine_R , train_dataf$Occipital_Sup_L , 
                          train_dataf$Occipital_Mid_L , train_dataf$Cerebelum_10_L , train_dataf$HLA.DQA1 , train_dataf$NPIPB6)
save(features, file = "E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Final Project/MCICN/file2.RData")

