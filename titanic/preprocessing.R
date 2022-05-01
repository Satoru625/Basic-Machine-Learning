library(tidyverse)
d <- read.csv("./titanic/input/train.csv")
d_lm <- d[,-c(1,4,9)] %>% mutate(Cabin = str_sub(Cabin,1,1))

mr <- lm(Age~.,data = d_lm)
pred <- predict(mr, d_lm)

d_train <- d_lm %>% mutate(Cabin = str_sub(Cabin,1,1),pred=ifelse(pred<0,0,pred)) %>% 
  mutate(Age = ifelse(is.na(Age==T),pred,Age)) %>% mutate(Age=ifelse(Age<0,0,Age)) %>% select(-pred)

write.csv(d_train,"./titanic/d_train.csv",row.names = FALSE)

#process test
d_lm2 <- read.csv("./titanic/input/test.csv")
d_lm2 <- d_lm2[,-c(1,3,8)] %>% mutate(Cabin = str_sub(Cabin,1,1))

mr2 <- lm(Age~.,data = d_lm2)
pred <- predict(mr2, d_lm2)

d_test <- d_lm2 %>% mutate(Cabin = str_sub(Cabin,1,1),pred=ifelse(pred<0,0,pred)) %>% 
  mutate(Age = ifelse(is.na(Age==T),pred,Age)) %>% mutate(Age=ifelse(Age<0,0,Age)) %>% select(-pred)

write.csv(d_test,"./titanic/d_test.csv",row.names = FALSE)
