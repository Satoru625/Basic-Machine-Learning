library(MASS)
library(tidyverse)
train <- read.csv("./titanic/d_train.csv")
test_raw <- read.csv("./titanic/input/test.csv")
test <- read.csv("./titanic/d_test.csv")

fit <- glm(Survived~., data=train, family = binomial)
# search the most fitted combination of variables by AIC
aic <- stepAIC(fit)
formula <- aic$formula
fit.2 <- glm(formula = formula,data=train, family = binomial)
fit.2$coefficients
# (Intercept)      Pclass     Sexmale         Age       SibSp 
# 6.06120081 -1.34526511 -2.73779904 -0.05453505 -0.46442316 
# odds should be calculated as the fomula below
# odds = exp(6.06120081 - 1.34526511*Pclass - 2.73779904*Sexmale - 0.05453505*Age - 0.46442316*SibSp)

pred.test <- predict(fit.2,test)
logit.test <- 1/(1+exp(-pred.test))
result.test <- ifelse(logit.test<0.5,0,1) # 0.5 is set as the border of survival, which can be changed

ans <- data.frame("PassengerId"=test_raw$PassengerId,"Survived"=result.test)