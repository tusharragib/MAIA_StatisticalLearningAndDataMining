#*** Student's Name: S M Ragib Shahriar Islam
#*Problem name: Final Assignment_ADCN
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

train_dataf<-read.csv('E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Final Project/ADCN/ADCNtrain.csv', header = TRUE)
test_dataf<-read.csv('E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Final Project/ADCN/ADCNtest.csv', header = TRUE)

train_df <-train_dataf
test_df <-test_dataf

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
dummy.tdata = dummy.tdata[,-568]
dummy.vdata = dummy.data.frame(vdata)
dummy.vdata = dummy.vdata[,-568]

#***Applying Logistic Regression
#library(glm)
glm_model = glm(LabelsAD ~ G_Frontal_Mid_Orb.1.R + G_Supramarginal.4.L + G_SupraMarginal.6.L + 
                G_Angular.1.R + G_Parietal_Inf.1.R + S_Intraparietal.3.L + 
                G_Insula.anterior.1.L + S_Sup_Temporal.1.L + S_Sup_Temporal.1.R + 
                S_Sup_Temporal.3.R + S_Sup_Temporal.4.L + G_Temporal_Mid.2.L + 
                G_Temporal_Mid.2.R + G_Temporal_Mid.3.L + G_Temporal_Mid.3.R + 
                G_Temporal_Inf.1.R + G_Temporal_Inf.2.R + G_Temporal_Inf.3.R + 
                G_Temporal_Inf.4.L + G_Temporal_Inf.4.R + G_Temporal_Pole_Mid.2.L + 
                G_Temporal_Pole_Mid.3.L + G_Temporal_Pole_Mid.3.R + G_Cingulum_Post.1.R + 
                G_Cingulum_Post.2.R + G_Paracentral_Lobule.1.L + G_Precuneus.8.R + 
                S_Parietooccipital.1.L + S_Parietooccipital.2.R + G_Lingual.5.L + 
                G_Hippocampus.1.L + G_Hippocampus.1.R + G_Hippocampus.2.R + 
                G_ParaHippocampal.1.L + G_ParaHippocampal.1.R + G_ParaHippocampal.2.R + 
                G_ParaHippocampal.3.R + G_Fusiform.1.R + G_Fusiform.2.R + 
                G_Fusiform.7.L + N_Thalamus.8.L + EMILIN1 + ITLN1 + LGALS8.AS1 + 
                TGM2, data = dummy.tdata, family = 'binomial',control = list(maxit = 500))
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
    pred.logR.predict.vdata[i] <- "AD"
  else
    pred.logR.predict.vdata[i]<- "CN"
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
    pred.logR.predict.test[i] <- "AD"
  else
    pred.logR.predict.test[i]<- "CN"
}
pred.logR.predict.test <- as.factor(pred.logR.predict.test)



#***Applying LDA 
lda_model = lda(LabelsAD ~ G_Frontal_Mid_Orb.1.R + G_Supramarginal.4.L + G_SupraMarginal.6.L + 
                  G_Angular.1.R + G_Parietal_Inf.1.R + S_Intraparietal.3.L + 
                  G_Insula.anterior.1.L + S_Sup_Temporal.1.L + S_Sup_Temporal.1.R + 
                  S_Sup_Temporal.3.R + S_Sup_Temporal.4.L + G_Temporal_Mid.2.L + 
                  G_Temporal_Mid.2.R + G_Temporal_Mid.3.L + G_Temporal_Mid.3.R + 
                  G_Temporal_Inf.1.R + G_Temporal_Inf.2.R + G_Temporal_Inf.3.R + 
                  G_Temporal_Inf.4.L + G_Temporal_Inf.4.R + G_Temporal_Pole_Mid.2.L + 
                  G_Temporal_Pole_Mid.3.L + G_Temporal_Pole_Mid.3.R + G_Cingulum_Post.1.R + 
                  G_Cingulum_Post.2.R + G_Paracentral_Lobule.1.L + G_Precuneus.8.R + 
                  S_Parietooccipital.1.L + S_Parietooccipital.2.R + G_Lingual.5.L + 
                  G_Hippocampus.1.L + G_Hippocampus.1.R + G_Hippocampus.2.R + 
                  G_ParaHippocampal.1.L + G_ParaHippocampal.1.R + G_ParaHippocampal.2.R + 
                  G_ParaHippocampal.3.R + G_Fusiform.1.R + G_Fusiform.2.R + 
                  G_Fusiform.7.L + N_Thalamus.8.L + EMILIN1 + ITLN1 + LGALS8.AS1 + 
                  TGM2, data = dummy.tdata)
set.seed(10)
lda.predict.vdata = predict(lda_model, dummy.vdata)
# Processing the "pred.logR.predict.vdata" vector for creating confusion matrix
pred.lda_class.predict.vdata = lda.predict.vdata$class
pred.lda_class.predict.vdata = as.numeric(pred.lda_class.predict.vdata)
length(pred.lda_class.predict.vdata)
for(i in 1:length(pred.lda_class.predict.vdata))
{
  if(pred.lda_class.predict.vdata[i] == 1)
    pred.lda_class.predict.vdata[i] <- "AD"
  else
    pred.lda_class.predict.vdata[i]<- "CN"
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
    pred.lda_class.predict.test[i] <- "AD"
  else
    pred.lda_class.predict.test[i]<- "CN"
}
pred.lda_class.predict.test = as.factor(pred.lda_class.predict.test)


#Applying Support Vector Machine(SVM)
svm_model = svm(Labels ~ G_Frontal_Mid_Orb.1.R + G_Supramarginal.4.L + G_SupraMarginal.6.L + 
                  G_Angular.1.R + G_Parietal_Inf.1.R + S_Intraparietal.3.L + 
                  G_Insula.anterior.1.L + S_Sup_Temporal.1.L + S_Sup_Temporal.1.R + 
                  S_Sup_Temporal.3.R + S_Sup_Temporal.4.L + G_Temporal_Mid.2.L + 
                  G_Temporal_Mid.2.R + G_Temporal_Mid.3.L + G_Temporal_Mid.3.R + 
                  G_Temporal_Inf.1.R + G_Temporal_Inf.2.R + G_Temporal_Inf.3.R + 
                  G_Temporal_Inf.4.L + G_Temporal_Inf.4.R + G_Temporal_Pole_Mid.2.L + 
                  G_Temporal_Pole_Mid.3.L + G_Temporal_Pole_Mid.3.R + G_Cingulum_Post.1.R + 
                  G_Cingulum_Post.2.R + G_Paracentral_Lobule.1.L + G_Precuneus.8.R + 
                  S_Parietooccipital.1.L + S_Parietooccipital.2.R + G_Lingual.5.L + 
                  G_Hippocampus.1.L + G_Hippocampus.1.R + G_Hippocampus.2.R + 
                  G_ParaHippocampal.1.L + G_ParaHippocampal.1.R + G_ParaHippocampal.2.R + 
                  G_ParaHippocampal.3.R + G_Fusiform.1.R + G_Fusiform.2.R + 
                  G_Fusiform.7.L + N_Thalamus.8.L + EMILIN1 + ITLN1 + LGALS8.AS1 + 
                  TGM2, data = tdata, kernel = "linear", cost = 1, scale = TRUE)
summary(svm_model)
set.seed(10)
pred.svm.predict.vdata = predict(svm_model, vdata)
confusionMatrix(pred.svm.predict.vdata, vdata$Labels)
pred.svm.predict.test = predict(svm_model, test_df)

#Plotting ROC
pred.svm.predict.vdata.bin <- as.character(pred.svm.predict.vdata)
for(i in 1:length(pred.svm.predict.vdata.bin))
{
  if(pred.svm.predict.vdata.bin[i] == "AD")
    pred.svm.predict.vdata.bin[i] <- 1
  else
    pred.svm.predict.vdata.bin[i]<- 0
}
pred.svm.predict.vdata.bin<-as.numeric(pred.svm.predict.vdata.bin)
par(pty="s") 
SVMROC <- roc(dummy.vdata$Labels ~ pred.svm.predict.vdata.bin,plot=TRUE,print.auc=TRUE,col="green",lwd =4,legacy.axes=TRUE,main="ROC Curves")


#***Applying Random Forest Classifier
RandomForest.fit = randomForest(Labels ~ G_Frontal_Mid_Orb.1.R + G_Supramarginal.4.L + G_SupraMarginal.6.L + 
                                  G_Angular.1.R + G_Parietal_Inf.1.R + S_Intraparietal.3.L + 
                                  G_Insula.anterior.1.L + S_Sup_Temporal.1.L + S_Sup_Temporal.1.R + 
                                  S_Sup_Temporal.3.R + S_Sup_Temporal.4.L + G_Temporal_Mid.2.L + 
                                  G_Temporal_Mid.2.R + G_Temporal_Mid.3.L + G_Temporal_Mid.3.R + 
                                  G_Temporal_Inf.1.R + G_Temporal_Inf.2.R + G_Temporal_Inf.3.R + 
                                  G_Temporal_Inf.4.L + G_Temporal_Inf.4.R + G_Temporal_Pole_Mid.2.L + 
                                  G_Temporal_Pole_Mid.3.L + G_Temporal_Pole_Mid.3.R + G_Cingulum_Post.1.R + 
                                  G_Cingulum_Post.2.R + G_Paracentral_Lobule.1.L + G_Precuneus.8.R + 
                                  S_Parietooccipital.1.L + S_Parietooccipital.2.R + G_Lingual.5.L + 
                                  G_Hippocampus.1.L + G_Hippocampus.1.R + G_Hippocampus.2.R + 
                                  G_ParaHippocampal.1.L + G_ParaHippocampal.1.R + G_ParaHippocampal.2.R + 
                                  G_ParaHippocampal.3.R + G_Fusiform.1.R + G_Fusiform.2.R + 
                                  G_Fusiform.7.L + N_Thalamus.8.L + EMILIN1 + ITLN1 + LGALS8.AS1 + 
                                  TGM2, data = tdata)
set.seed(10)
RF.predict.vdata <- predict(RandomForest.fit, vdata)
confusionMatrix(RF.predict.vdata, vdata$Labels)
RF.predict.test_df = predict(RandomForest.fit, test_df)

#Plotting ROC
RF.predict.vdata.bin <- as.character(RF.predict.vdata)
for(i in 1:length(RF.predict.vdata.bin))
{
  if(RF.predict.vdata.bin[i] == "AD")
    RF.predict.vdata.bin[i] <- 1
  else
    RF.predict.vdata[i]<- 0
}
RF.predict.vdata.bin<-as.numeric(RF.predict.vdata.bin)
par(pty="s") 
RFROC <- roc(dummy.vdata$Labels ~ RF.predict.vdata.bin,plot=TRUE,print.auc=TRUE,col="green",lwd =4,legacy.axes=TRUE,main="ROC Curves")


#Making RData file
patient_id <- as.vector(train_dataf$Subject_id)
save(patient_id, pred.logR.predict.test, pred.lda_class.predict.test, pred.svm.predict.test, RF.predict.test_df, file = "E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Final Project/ADCN/file1.RData")
features <- data.frame (train_dataf$G_Frontal_Mid_Orb.1.R , train_dataf$G_Supramarginal.4.L , train_dataf$G_SupraMarginal.6.L , 
  train_dataf$G_Angular.1.R , train_dataf$G_Parietal_Inf.1.R , train_dataf$S_Intraparietal.3.L , 
  train_dataf$G_Insula.anterior.1.L , train_dataf$S_Sup_Temporal.1.L , train_dataf$S_Sup_Temporal.1.R , 
  train_dataf$S_Sup_Temporal.3.R , train_dataf$S_Sup_Temporal.4.L , train_dataf$G_Temporal_Mid.2.L , 
  train_dataf$G_Temporal_Mid.2.R , train_dataf$G_Temporal_Mid.3.L , train_dataf$G_Temporal_Mid.3.R , 
  train_dataf$G_Temporal_Inf.1.R , train_dataf$G_Temporal_Inf.2.R , train_dataf$G_Temporal_Inf.3.R , 
  train_dataf$G_Temporal_Inf.4.L , train_dataf$G_Temporal_Inf.4.R , train_dataf$G_Temporal_Pole_Mid.2.L , 
  train_dataf$G_Temporal_Pole_Mid.3.L , train_dataf$G_Temporal_Pole_Mid.3.R , train_dataf$G_Cingulum_Post.1.R , 
  train_dataf$G_Cingulum_Post.2.R , train_dataf$G_Paracentral_Lobule.1.L , train_dataf$G_Precuneus.8.R , 
  train_dataf$S_Parietooccipital.1.L , train_dataf$S_Parietooccipital.2.R , train_dataf$G_Lingual.5.L , 
  train_dataf$G_Hippocampus.1.L , train_dataf$G_Hippocampus.1.R , train_dataf$G_Hippocampus.2.R , 
  train_dataf$G_ParaHippocampal.1.L , train_dataf$G_ParaHippocampal.1.R , train_dataf$G_ParaHippocampal.2.R , 
  train_dataf$G_ParaHippocampal.3.R , train_dataf$G_Fusiform.1.R , train_dataf$G_Fusiform.2.R , 
  train_dataf$G_Fusiform.7.L , train_dataf$N_Thalamus.8.L , train_dataf$EMILIN1 , train_dataf$ITLN1 , train_dataf$LGALS8.AS1 , 
  train_dataf$TGM2)
save(features, file = "E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Final Project/ADCN/file2.RData")
