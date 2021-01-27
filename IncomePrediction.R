#*********************************************#
#****************Part Part********************#
#*********************************************#
# load packages
library(ggplot2)
library(plyr)
library(gridExtra)
library(gmodels)
library(grid)
library(vcd)
library(scales)
library(ggthemes)
library(knitr)
library(readxl)
library(funModeling) 
library(Hmisc)
library(skimr)

# load the data
adult <- read.table("adult.data", sep = ",", header = FALSE)

# add column names to the adult dataset
colnames(adult) <- c("age", "workclass", "fnlwgt", 
                     "education", "education_num", 
                     "marital_status", "occupation",
                     "relationship", "race", "sex", 
                     "capital_gain", "capital_loss", 
                     "hours_per_week", "native_country", "income")

head(adult)
str(adult)

# check out the levels of each factor variable
levels_factors <- function(adult) {
  col_names <- names(adult)
  for (i in 1:length(col_names)) {
    if (is.factor(adult[, col_names[i]])) {
      message(noquote(paste("Covariate ", "*", 
                            col_names[i], "*", 
                            " with factor levels:", 
                            sep = "")))
      print(levels(adult[, col_names[i]]))
     }
  }
}

levels_factors(adult)


###########################
# Cleaning missing values #
###########################
adult <- read.table("adult.data",
                    sep = ",", 
                    header = FALSE, 
                    na.strings = " ?")

colnames(adult) <- c("age", "workclass", "fnlwgt", "education", 
                     "education_num", "marital_status", "occupation",
                     "relationship", "race", "sex", "capital_gain", 
                     "capital_loss", "hours_per_week", "native_country", "income")

# delete all rows containing missing values
adult <- na.omit(adult)
# reenumerate the rows of the data frame
row.names(adult) <- 1:nrow(adult)


###############################
# Transformations on the Data #
###############################
### variable hours_per_week ###
summary(adult)

# box plot
ggplot(aes(x = factor(0), y = hours_per_week),
       data = adult) + 
  geom_boxplot() +
  stat_summary(fun = mean, 
               geom = 'point', 
               shape = 19,
               color = "red",
               cex = 2) +
  scale_x_discrete(breaks = NULL) +
  scale_y_continuous(breaks = seq(0, 100, 5)) + 
  xlab(label = "") +
  ylab(label = "Working hours per week") +
  ggtitle("Box Plot of Working Hours per Week") 

# create a new factor variable called “hours_w” with 5 categories
adult$hours_w[adult$hours_per_week < 40] <- " less_than_40"
adult$hours_w[adult$hours_per_week >= 40 & 
                adult$hours_per_week <= 45] <- " between_40_and_45"
adult$hours_w[adult$hours_per_week > 45 &
                adult$hours_per_week <= 60  ] <- " between_45_and_60"
adult$hours_w[adult$hours_per_week > 60 &
                adult$hours_per_week <= 80  ] <- " between_60_and_80"
adult$hours_w[adult$hours_per_week > 80] <- " more_than_80"

# make the new variable “hours_w” a factor variable
adult$hours_w <- factor(adult$hours_w,
                        ordered = FALSE,
                        levels = c(" less_than_40", 
                                   " between_40_and_45", 
                                   " between_45_and_60",
                                   " between_60_and_80",
                                   " more_than_80")) 

summary(adult$hours_w)

# check the percentage of people for different work hour categories
for(i in 1:length(summary(adult$hours_w))){
  
  print(round(100*summary(adult$hours_w)[i]/sum(!is.na(adult$hours_w)), 2)) 
}

### varible native country ###
levels(adult$native_country)

# group the countires by global regions
# define the regions
AsiaEast <- c(" Cambodia", " China", " Hong", " Laos", " Thailand",
              " Japan", " Taiwan", " Vietnam")

AsiaCentral <- c(" India", " Iran")

CentralAmerica <- c(" Cuba", " Guatemala", " Jamaica", " Nicaragua", 
                    " Puerto-Rico",  " Dominican-Republic", " El-Salvador", 
                    " Haiti", " Honduras", " Mexico", " Trinadad&Tobago")

SouthAmerica <- c(" Ecuador", " Peru", " Columbia")


EuropeWest <- c(" England", " Germany", " Holand-Netherlands", " Ireland", 
                " France", " Greece", " Italy", " Portugal", " Scotland")

EuropeEast <- c(" Poland", " Yugoslavia", " Hungary")

# creat a new column - native_region
adult <- mutate(adult, 
                native_region = ifelse(native_country %in% AsiaEast, " EastAsia",
                                       ifelse(native_country %in% AsiaCentral, " CentralAsia",
                                              ifelse(native_country %in% CentralAmerica, " CentralAmerica",
                                                     ifelse(native_country %in% SouthAmerica, " SouthAmerica",
                                                            ifelse(native_country %in% EuropeWest, " EuropeWest",
                                                                   ifelse(native_country %in% EuropeEast, " EuropeEast",
                                                                          ifelse(native_country == " UnitedStates", " UnitedStates", 
                                                                                 " OutlyingUS" ))))))))
# transform the new variable into a factor
adult$native_region <- factor(adult$native_region, ordered = FALSE)

### Varible "workclass"
summary(adult$workclass)

# remove the factor level “Never-worked”
adult$workclass <- droplevels(adult$workclass)

levels(adult$workclass)

################################
#Preprocessing the Test Dataset#
################################
# load the test dataset
test <- read.table("adult.test", sep = ",", header = FALSE, skip = 1, 
                   na.strings = " ?")

colnames(test) <- c("age", "workclass", "fnlwgt", "education",
                    "education_num", "marital_status", "occupation",
                    "relationship", "race", "sex", "capital_gain",
                    "capital_loss", "hours_per_week",
                    "native_country", "income")

# Cleaning missing values
test <- na.omit(test)
row.names(test) <- 1:nrow(test)

head(test, 10)

# remove "."
levels(test$income)[1] <- " <=50K"
levels(test$income)[2] <- " >50K"
levels(adult$income)[1] <- " <=50K"
levels(adult$income)[2] <- " >50K"

levels(test$income)

str(test$income)

# create a new variable called “hours_w”
test$hours_w[test$hours_per_week < 40] <- " less_than_40"
test$hours_w[test$hours_per_week >= 40 & 
               test$hours_per_week <= 45] <- " between_40_and_45"
test$hours_w[test$hours_per_week > 45 &
               test$hours_per_week <= 60  ] <- " between_45_and_60"
test$hours_w[test$hours_per_week > 60 &
               test$hours_per_week <= 80  ] <- " between_60_and_80"
test$hours_w[test$hours_per_week > 80] <- " more_than_80"

# convert it into factor
test$hours_w <- factor(test$hours_w,
                       ordered = FALSE,
                       levels = c(" less_than_40", 
                                  " between_40_and_45", 
                                  " between_45_and_60",
                                  " between_60_and_80",
                                  " more_than_80"))

# create varible native_region
test <- mutate(test, 
               native_region = ifelse(native_country %in% AsiaEast, " EastAsia",
                                      ifelse(native_country %in% AsiaCentral, " Central-Asia",
                                             ifelse(native_country %in% CentralAmerica, " CentralAmerica",
                                                    ifelse(native_country %in% SouthAmerica, " SouthAmerica",
                                                           ifelse(native_country %in% EuropeWest, " EuropeWest",
                                                                  ifelse(native_country %in% EuropeEast, " EuropeEast",
                                                                         ifelse(native_country == " UnitedStates", " UnitedStates", 
                                                                                " OutlyingUS" ))))))))

test$native_region <- factor(test$native_region, ordered = FALSE)

# drop the unused level “Never-worked” 
test$workclass <- droplevels(test$workclass)

# export the cleaned and preprocessed train and test datasets
library(openxlsx)
write.xlsx(adult, "adult_df.xlsx", row.names = FALSE)
write.xlsx(test, "test_df.xlsx", row.names = FALSE)

# Part Two: Modeling

#*********************************************#
#****************Part Two ********************#
#*********************************************#
# export the cleaned and preprocessed train and test datasets
library(openxlsx)

# read data, removed the extra space " >50K"
adult <- read.xlsx("adult_df1.xlsx")
test <- read.xlsx("test_df1.xlsx")

### convert Y variable (income) into 0 and 1
adult$income <- as.factor(ifelse(adult$income == ">50K", 1 , 0))
test$income <- as.factor(ifelse(test$income == ">50K", 1 , 0))

table(adult$income)
levels(test$income)
levels(adult$income)

# convert sex into 0 and 1
adult$sex <- as.factor(ifelse(adult$sex == " Male", 1 , 0))
test$sex <- as.factor(ifelse(test$sex == " Male", 1 , 0))
table(adult$sex)
levels(adult$sex)

#######################
# Logistic Regression #
#######################
library(corrgram)
library(GGally)

ggcorr(adult, method = c("everything", "pearson"), label = TRUE)

# EDA
basic_eda <- function(adult)
{
  glimpse(adult)
  df_status(adult)
  freq(adult) 
  profiling_num(adult)
  plot_num(adult)
  describe(adult)
}
basic_eda(adult)


############################
###### Train the model #####
############################
set.seed(123)
## 75% of the sample size
smp_size <- floor(0.75 * nrow(adult))

## set the seed to make your partition reproducible
train_ind <- sample(seq_len(nrow(adult)), size = smp_size)

train <- adult[train_ind, ]
test1 <- adult[-train_ind, ]
names(adult)

# deleted Holand-Netherlands in order to have the same levels
train$native_country <- as.factor(as.character(train$native_country))
test1$native_country <- as.factor(as.character(test1$native_country))
levels(train$native_country)
levels(test1$native_country)

# Build the model
lg <- glm(income ~., data = train, family = binomial('logit'))
summary(lg)

coef(lg)

#anova
anova(lg,test = "Chisq")

# drop “fnlwgt”, “education_num”, and “native_region” 
# didn't chose this option, because the accuracy decreased and RMSE increased
#lg1 <- glm(income ~ -fnlwgt -education_num -native_region, data = train, 
           #family = binomial('logit'))

##############################################################################
# This logistic regression model appears to fit the data well. 
# To explore the possibility of a parsimonious model, both the 
# forward and backward stepwise selection algorithms using AIC are performed.
##############################################################################
# full model is the model just fitted
#lg_full <- lg
#lg_null <- glm(income ~1 , data = train, family = binomial('logit'))
#summary(lg_full)

# backward selection
#step(lg_full, trace = F, scope = list(lower=formula(lg_null), upper=formula(lg_full)),
#     direction = 'backward')
# AIC: 14740

# Perform stepwise variable selection
# forward selection
#step(lg_null, trace = F, scope = list(lower=formula(lg_null), upper=formula(lg_full)),
#     direction = 'forward')
# AIC: 14720
# the AIC for forward is lower then backward stepwise selection
# Thus, the forward model is chosen 

# stepwise forward regression
#Select the most contributive variables
#library(MASS)
#lg_step <- lg_full %>% stepAIC(trace = FALSE)
#coef(lg_step)

#lg_step <- glm(income ~., data = train, family = binomial('logit')) %>%
#  stepAIC(trace = FALSE)

# Summarize the final selected model
#summary(lg_step)

# The orgianl model is chosen.
#############################
######## Prediction #########
#############################
#predicted <- plogis(predict(m1, test1))
lgPred <- predict(lg, test1, type="response")
count(test1)

library(InformationValue)
optCutOff <- optimalCutoff(test1$income, lgPred)[1]
optCutOff

misClassError(test1$income, lgPred, threshold = optCutOff)

library(pROC)

# ROC curve
roc_lg <- roc(test1$income, lgPred)
roc_lg

#plot the roc
plotROC(test1$income, lgPred)

# AUC curve
roc_lg_auc <- auc(roc_lg)
roc_lg_auc

Concordance(test1$income, lgPred)

sensitivity(test1$income, lgPred, threshold = optCutOff)

specificity(test1$income, lgPred, threshold = optCutOff)

confusionMatrix(test1$income, lgPred,threshold = .9)

#F1 score
library(MLmetrics)

#predicted <- ifelse(m1$fitted.values < 0.9047, 0, 1)
F1_Score(lgPred, test1$income, positive = "1")

Pred <- as.factor(ifelse(lgPred >= "0.9047", 1 , 0))

Pred <- as.numeric(as.factor(Pred))
test1$income <- as.numeric(as.factor(test1$income))
LogLoss(Pred,test1$income)

# RMSE
str(Pred)

str(test1$income)

RMSE(Pred,test1$income)

# Model accuracy
lgAcc <- mean(Pred == test1$income)
lgAcc

start_time <- proc.time()
time.logistic <- proc.time() - start_time
time.logistic


###########################################################################
########################## Random Forest ##################################
###########################################################################
library(randomForest)

##Check for null
sapply(train, function(x) sum(is.na(x))) 

#Convert all character variable into factor 
library(dplyr)
train_fac=train %>% mutate_if(is.character, as.factor)
test_fac=test1 %>% mutate_if(is.character, as.factor)

set.seed(123)
rfm<- randomForest(income ~., data = train_fac,ntree=1000)
print(rfm)
       
library(caret)
library(e1071)
library(ROCR)
       
summary(rfm)

p_rfm = predict(rfm, newdata=test_fac)

cm = table(test_fac$income, p_rfm)
importance(rfm)

varImpPlot(rfm)

#confusion matrix for test set
table(p_rfm,test1$income)

# ROC curve
library(pROC)
p_rfm1 <- predict(rfm, newdata=test_fac,type = "prob")
roc_rfm <- roc(test_fac$income,p_rfm1[,2])
plot(roc_rfm,col = "green", main = 'ROC for Random Forest')

#Area Under Curve(AUC) for each ROC curve, the higher the better
roc_rfm_auc <- auc(roc_rfm)
roc_rfm_auc

# Model accuracy
rfmAcc <- sum(diag(cm)) / sum(cm)
rfmAcc
       
# To check classwose error
plot(rfm)
varImpPlot(rfm)

optCutOff1 <- optimalCutoff(test1$income, rfmAcc)[1]
optCutOff1

misClassError(test1$income, rfmAcc, threshold = optCutOff)

#F1 score
F1_Score(p_rfm, test_fac$income, positive = "1")

p_rfm3 <- as.numeric(as.factor(p_rfm))
test_fac$income1 <- as.numeric(as.factor(test_fac$income))
LogLoss(p_rfm3,test_fac$income1)

# RMSE
RMSE(p_rfm3,test_fac$income1)

start_time <- proc.time()
time.rf <- proc.time() - start_time
       
#################################################
######## Decision Tree ##########################
#################################################
library(rpart)
library(rpart.plot)
# build model
dt <- rpart(income ~ ., data = train, method = 'class')

summary(dt)

rpart.plot (dt, extra = 106, type = 1, branch=.3, under = TRUE)
       
rpart.rules(dt) 
       
prp(dt)
       
#make prediction
p_dt <- predict(dt, test1, type = 'class')
table_mat <- table(test1$income, p_dt)
table_mat
       
# Accuracy
dt_Acc <- sum(diag(table_mat)) / sum(table_mat)
dt_Acc
printcp(dt)

rsq.rpart(dt) 
print(dt)
plot(dt)


control <- rpart.control(minsplit = 10,
                  minbucket = round(20/ 3),
                  maxdepth = 4,
                  cp = 0)
ct_dt <- rpart(income~., data = train, method = 'class', control = control)

p_cdt <- predict(ct_dt, test1, type = 'class')
table_mat_ct <- table(test1$income, p_cdt)
table_mat_ct
       
# Accuracy
dtAcc <- sum(diag(table_mat_ct)) / sum(table_mat_ct)
dtAcc
rpart.plot(ct_dt, extra = 106)
       
#10 fold CV is used
numFolds = trainControl( method = "cv", number = 10 )
       
#The search grid for cp - the complexity paramater is defined
cartGrid = expand.grid( .cp = seq(0.002,0.1,0.002))
save_CV<-train(income~.,data=train,method="rpart",trControl=numFolds, 
               tuneGrid=cartGrid)
       
plot(save_CV)
       
plot(save_CV$results$cp, save_CV$results$Accuracy, type="l", xlab="cp", ylab="accuracy")
       
cpbest<-save_CV$bestTune
CensusCV <- rpart(income ~ ., data=train, method="class", cp=cpbest)
PredictCV = predict(CensusCV, newdata = test1, type = "class")

#confusion matrix
cmat_CART_CV<-table(test1$income, PredictCV) #confusion matrix
cmat_CART_CV
accu_CART_CV <- (cmat_CART_CV[1,1] + cmat_CART_CV[2,2])/sum(cmat_CART_CV)
accu_CART_CV
prp(CensusCV)    
save_CV
printcp(CensusCV)
rsq.rpart(CensusCV)
print(CensusCV)
plot(CensusCV)
       
#F1 score
F1_Score(PredictCV, test1$income, positive = "1")
       
p_cdt2 <- as.numeric(as.factor(PredictCV))
test1$income3 <- as.numeric(as.factor(test1$income))
LogLoss(p_cdt2,test1$income3)

# RMSE
RMSE(p_cdt2,test1$income3)
       
# ROC curve
roc_dt1 <- roc(test1$income3, p_cdt2)
roc_dt1
       
#plot the roc
plotROC(test1$income3, p_cdt2)
       
# AUC curve
roc_dt_auc1 <- auc(roc_dt1)
roc_dt_auc1

# time
start_time <- proc.time()
time.dt <- proc.time() - start_time
       
################################################################
############### Support Vector Machine #########################
################################################################

library(kernlab)
       
svm1 <- svm(income ~ . , data = train, type = 'C-classification')
summary(svm1)
       
# Prediction
p_svm <- predict(svm1, newdata = test1)

# Confusion matrix
svm_table <- table(test1$income, p_svm)
svm_table

#F1 score
F1_Score(p_svm, test1$income, positive = "1")
time.dt
       
p_svm1 <- as.numeric(as.factor(p_svm))
test1$income3 <- as.numeric(as.factor(test1$income))
LogLoss(p_svm1,test1$income3)

# RMSE
RMSE(p_svm1,test1$income3)

library(pROC)
# ROC curve
roc_svm <- roc(test1$income3, p_svm1)
roc_svm
       
#plot the roc
plotROC(test1$income3, p_svm1)
       
# AUC curve
roc_svm_auc <- auc(roc_svm)
roc_svm_auc

#Accuracy
svmAcc <- sum(diag(svm_table)) / sum(svm_table)
svmAcc

start_time <- proc.time()
time.svm <- proc.time() - start_time
time.svm

##############################
# Performace Comparison
##############################
Accuracy<-data.frame(Model=c('Logistic Regression','Random Forest','Decision Tree',
                             'Support Vector Machine'),Accuracy=c(lgAcc,rfmAcc,dtAcc,svmAcc))
Accuracy
       
gg<-ggplot(Accuracy,aes(x=Model,y=Accuracy,fill=Model))+geom_bar(stat = 'identity')+
  theme_bw()+ggtitle('Accuracies of Models')#+geom_hline(yintercept = benchmark,color='red')
gg
       
# We use the logistic regression model to see which features contribute more to the income.
summary(lg)
