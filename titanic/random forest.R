library(tidyverse)
library(randomForest)
train <- read.csv("d_train.csv")
test_raw <- read.csv("./input/test.csv")
test <- read.csv("d_test.csv")

#tuning
set.seed(1234)
tune <- tuneRF(train[,-1],train[,1],plot = T)

rf <- randomForest(Survived~.,data=train, type="classification", mtry = tune[which.min(tune[, 2]), 1])


res <- predict(rf, test)
res <- ifelse(res<=0.5,0,1)

ans <- data.frame("PassengerId"=test_raw$PassengerId,"Survived"=res)
