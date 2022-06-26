#*** Student's Name: S M Ragib Shahriar Islam
#*Problem name: Mid-term Assignment


#***Data reading and analysis
# Reading the Training and Testing Dataset
train_df<-read.csv('E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Midterm Project/train_ch.csv', header = TRUE)
test_df<-read.csv('E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Midterm Project/test_ch.csv', header = TRUE)

#Removing the non-predictor variables
train_df = train_df[,-1]
test_df = test_df[,-1]

#Running the value type and summary test of the training data
str(train_df)
summary(train_df)

#Creating Histogram of the variables
hist(train_df$v1)
hist(train_df$v2)
hist(train_df$v3)
hist(train_df$v4)
hist(train_df$v5)
hist(train_df$v6)
hist(train_df$v7)
hist(train_df$v8)
hist(train_df$v9)

#Creating Boxplots of the variables
boxplot(train_df$v1)
boxplot(train_df$v2)
boxplot(train_df$v3)
boxplot(train_df$v4)
boxplot(train_df$v5)
boxplot(train_df$v6)
boxplot(train_df$v7)
boxplot(train_df$v8)
boxplot(train_df$v9)

#Bi-variate Analysis
pairs(~train_df$Y+train_df$v1+train_df$v2+train_df$v3+train_df$v4+train_df$v5+train_df$v6+train_df$v7+train_df$v8+train_df$v9)

#Correcting the outliers of variable V2
uv <- 1.5 * quantile(train_df$v2, 0.80)
train_df$v2[train_df$v2 > uv] <-uv

#Checking the status of the corrected outliers in variable V2
summary(train_df$v2)
hist(train_df$v2)
boxplot(train_df$v2)

#Correcting the outliers of variable V3
lv <- 0.1 * quantile(train_df$v3, 0.05)
train_df$v3[train_df$v3 < lv] <-lv

#Checking the status of the corrected outliers in variable V3
summary(train_df$v3)
hist(train_df$v3)
boxplot(train_df$v3)


#***Linear Regression
#Checking the lnear relation between the variables
cor(train_df)
round(cor(train_df),2)

#Training and running the Linear Model
fit = lm(Y~.,data = train_df)
fit_train = predict(fit,train_df)
lm_pred = predict(fit,test_df)
mean((train_df$Y - fit_train)^2)
abline(fit, col="blue", lwd=3, lty=2)


#***KNN Regression
  #install.packages("rJava") #Please install this package if not installed already. To install "RWeka" this package will be needed.
  #install.packages("RWeka") #Please install this package if not installed already. It will be needed to apply the IBk() function.

#including RWeka library package
library(RWeka)

#Training and running the KNN regression model
model_knn = IBk(Y~., data = train_df, control = Weka_control(K = 31, X = TRUE))
knn_pred_train = predict(model_knn, train_df)
knn_pred = predict(model_knn, test_df)
mean((train_df$Y - knn_pred_train)^2)


#***Making the .RDATA file
save(fit, model_knn, lm_pred, knn_pred, file = "E:/Academic/MAIA/Semester_2/Main/Statistical Learning and Data Mining/Midterm Project/xy.RData")
