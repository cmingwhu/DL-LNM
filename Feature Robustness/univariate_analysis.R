library(Hmisc); 
library(grid); 
library(lattice);
library(Formula); 
library(ggplot2);
library(rms);
library(pROC)
library(ROCR)
library(WGCNA)
library(tidyverse)


data<-read.csv('../data/clin_P.csv',header=TRUE)
dim(data)
data[is.na(data)] <- 0


data$P_N<-factor(data$P_N, levels=c(0,1), labels= c("LNM(-)","LNM(+)"))
data$Location<-factor(data$Location, levels=c(1,2,3,4), labels=c("Cardia","Body","Antrum","More 2/3 stomach"))
data$Gender <-
  factor(data$Gender,
         levels = c(0, 1),
         labels = c("Female", "male")
  )
data$CT_T <-
  factor(
    data$CT_T,
    levels = c( 2, 3, 4),
    labels = c("T2", "T3", "T4")
  )
data$cN <-
  factor(
    data$cN,
    levels = c( 0, 1),
    labels = c("0", "1")
  )
data$CT_TT<-
  factor(data$CT_TT,
         levels = c(0, 1),
         labels = c("0", "1"))
data$cTNM<-
  factor(data$cTNM,
         levels = c(1, 2, 3, 4),
         labels = c("1", "2", "3", "4"))

head(data)

names(data)


summary(data)

str(data)


x1<-c("Diameter","Thickness","Age")
x2<-c("Gender","Location","CT_T","cN","cTNM")

set.seed(3)
# split data
N <- nrow(data)
test_index <- sample(N,0.3*N)
train_index <- c(1:N)[-test_index]
test_data <- data[test_index,]
train_data <- data[train_index,]
sprintf('training： %s', nrow(test_data))
sprintf('testing： %s', nrow(train_data))


library(tableone)

train_P <- CreateTableOne(vars=c(x1,x2),
                         data = train_data,
                         factorVars = x2,
                         strata = 'P_N', addOverall = TRUE)
results1 <- print(train_P,showAllLevels = FALSE)


test_P <- CreateTableOne(vars=c(x1,x2),
                         data = test_data,
                         factorVars = x2,
                         strata = 'P_N', addOverall = TRUE)
results2 <- print(test_P,showAllLevels = FALSE) 

