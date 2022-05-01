library(tidyverse)
library(randomForest)
train <- read.csv("./titanic/d_train.csv")# %>% select(-c("PassengerId","Name","Ticket"))
test_raw <- read.csv("./titanic/input/test.csv")
test <- read.csv("./titanic/d_test.csv")# %>% select(-c("PassengerId","Name","Ticket")) %>% mutate(Cabin = str_sub(Cabin,1,1))

#tuning
set.seed(1234)
tune <- tuneRF(train[,-1],train[,1],plot = T)

rf <- randomForest(Survived~.,data=train, type="classification", mtry = tune[which.min(tune[, 2]), 1])


res <- predict(rf, test)
res <- ifelse(res<=0.5,0,1)

ans <- data.frame("PassengerId"=test_raw$PassengerId,"Survived"=res)
write.csv(ans,"./titanic/myans.csv",row.names = FALSE)
