# Machine Learning Algorithms on Census Income Data Prediction

## Tool and Resources Used 
**R Studio**   
**Packages:** ggplot2, plyr, gridExtra, gmodels, grid, vcd, scales, ggthemes, knitr, readxl, funModeling, Hmisc, skimr         
**Data Sources:** 
https://archive.ics.uci.edu/ml/datasets/census+income

## Problem and Goal
The problem of income inequality has become a part of a huge concern for everyone in the world. The purpose of this project is two fold. First, construct a model that can predict whether an individual’s annual income is lesser or greater than $50,000 dollars, based on given attributes, and provide a solution to the income inequality problem by using machine learning and data mining techniques. And second, perform a comprehensive analysis to find out the key factors that are essential in benefiting the population's income in general.

## Data Mining Task and Algorithms
In this project we will use four different supervised machine learning algorithms - logistic regression, random forest, decision tree, and support vector machine (SVM). We will compare if these algorithms give approximately the same prediction accuracy and select the best fit predictive model, in the end.

We will trace the following steps in our project: get the data, explore and clean the data, use exploratory data analysis to identify patterns and relationships in the data, build the prediction model, validate the prediction model, conduct an analysis of the prediction accuracy, sensitivity, specificity and computational time of the fitted models.

We will also use a classification approach to identify if an individual’s annual income is lesser or greater than $50,000. Thus, we can assume that there are two outcomes, less than $50,000 dollars a year (Y = 0) and more than $50,000 dollars a year (Y = 1).

## Evaluation and Comparison
As this is a classification problem, we will use F1 scores, confusion matrix and receiver operating characteristics (ROC) curve to evaluate and compare the performance of different predictive models.

## Data Plan
The UCI Machine Learning Repository Adult Dataset has been used for this purpose. Barry Becker extracted this data by using the 1994 Census database. There are 14 attributes for 42 counties and a total of 32,561 observations in this dataset. Within the 14 attributes, 8 of them are categorical (workclass, education, marital status, occupation, relationship, race, sex, and native country) while 6 of them are continuous (age, fnlwgt, education num, capital gain, capital loss and hours-per week).
Based on the given attributes, a new column was created to determine whether an individual makes more than $50,000 dollars a year (1) or less than $50,000 dollars a year (0). We will also split the dataset into a training set (75%) and testing set (25%).

## Data Processing
*	**Data Cleaning**
*	**Handling Missing Values**
*	**Data Transformation**
*	**Dummy Variable**
*	**Splitting**
*	**Handling Missing Values**

## Exploratory Data Analysis

## Correlation Matrix
*	**Income and Age**
*	**Income and Gender**
*	**Income and Working Hours Per Week**
*	**Income and Workclass**

## Experiments on Models
*	**ANOVA Chi-square Test**
After we built the logistic regression model, we performed the ANOVA Chi-square test to check the overall effect of variables on the dependent variable. We noticed variables “fnlwgt”, “education_num”, and “native_region” are not significant, so we dropped these three variables and ran the model again expecting to get a higher model accuracy. However, the classifier returns 75.16% accuracy when the model excludes these three predictors compared to 78.81% accuracy when the model includes all the predictors in it.

* The Area Under the Receiver Operator Characteristic Curve (AUROC) is an effective way to summarize the overall diagnostic accuracy of the test. Receiver Operator Characteristic Curve (ROC) is a curve of probability. An excellent model has an area under the ROC curve (AUC) near to 1, which means it has good measure of separability. A poor model has AUC near to 0, which means it has the worst measure of separability. In addition, when AUC is .5, it means the model has no class separation capacity. When we excluded those three independent variables, the value of ACU is a lot smaller than .5, which suggests no discrimination. Thus, we included all the variables in the final model instead.

*	**Forward and Backward Stepwise Regression**
The logistic regression model appears to fit the data well. To explore the possibility of a parsimonious model, both the forward and backward stepwise selection algorithms using AIC are performed. It is a relative measure of model fit. Lower value of AIC suggests a better model.

* The AIC of the forward stepwise regression (AIC = 14720) is lower than the backward (AIC = 14740) and initial model (AIC = 14746). Therefore, the forward stepwise regression is chosen by using the R function stepAIC( ). After we ran the forward stepwise model, the AIC returned 14735 and the model accuracy increased from 78.81% (original model) to 78.82% (forward stepwise model). However, the value of AUC for forward stepwise regression is .09 (Figure 15), which indicates it is a very poor model. Since, the value of AUC for the original regression model is .91, the final model returned by the original regression model.

*	**Cross Validation Decision Tree**
10 fold cross validation is used to improve the decision tree model accuracy. As a result, both the accuracy and the AUC of the optimized model are improved. The accuracy returns 85.8% compared to 84.3% without the cross validation.

## Evaluation and Comparisons
*	**Accuracy**
*	**Root Mean Square Error(RMSE)**
*	**F1 Score**
*	**AUC Curve**
Based on the results, random forest has the highest accuracy of 86.3%. When it comes to RMSE, the lower the better. As seen above, random forest has the lowest RMSE of .370. Random forest also has the highest F1 score of .105, while logistic regression has the lowest F1 score of .032.

*	**ROC Curve**
Being a classification problem, we use the ROC Curve to check which classifier performed the best. Support vector machine appears to have the lowest AUC. Random forest again performed the best here with the maximum AUC of .913.

*	**Confusion Matrix**
Confusion matrix is one of the most powerful and widely used evaluations. It allows us to generate other metrics which helps us to measure the performance of a classification model. Correct and incorrect predictions are provided in the results with their values and broken down by each class.

*	**Precision**
*	**Specificity**
*	**Sensitivity**
logistic regression has the highest precision. Support vector machine has the best specificity. Decision tree has the highest sensitivity and relatively low specificity. Random forest has the second highest precision and relatively high sensitivity

*	**Computational Time**
In terms of computational time, the best model is the decision tree. The decision tree model also has a good prediction accuracy and lower RMSE. The runtime for the logistic regression model is .471s and is very close to the best one ,  .420s for the decision tree. On the other hand, random forest has the highest runtime 1.343s.

* In conclusion, we used several Machine Learning models and concluded with an accuracy of 86.3%, RMSE of .370, precision of .932, specificity of .761, sensitivity of .890 and F1 Score of .105 with random forest and maximum AUC of .913. 

## Conclusions and Future Work
Random forest is known for its robustness and less sensitivity. As expected, random forest and decision tree performed the best followed by support vector machine and logistic regression. This is preferable as we aim to make the models as simple as possible, parsimony, and the results, where the simpler models actually performed better, aligned with our goal.

Going forward, there are a lot of different areas that can be carried out. In our study, we admit possible limitations. One of the main limitations of this study was that the data used in this study was not the recent census data. In order to make the models more applicable for today’s society, future studies should, therefore, extend the scope of this research by gathering the up to date census data.

It is clear that as new information is constantly generated by people, further studies could use these updated data to investigate other relative efficacy variables on income in the United States. It is also expected that future research will explore other advanced preprocessing methods and datasets from different countries to see the differences of country-based income prediction. 
