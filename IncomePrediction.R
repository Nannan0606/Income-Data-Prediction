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
