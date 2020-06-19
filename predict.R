# Load Required Libraries
library(tidyverse)
library(data.table)
library(caret)
library(stringr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(knitr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#Seperate the title from the year
edx <- edx %>% mutate(year = as.numeric(substr(str_extract(title,"\\(\\d\\d\\d\\d\\)"),2,5)),
                      title = substr(title,1,regexpr("\\(\\d\\d\\d\\d\\)",title)-2))

validation <- validation %>% 
  mutate(year = as.numeric(substr(str_extract(title,"\\(\\d\\d\\d\\d\\)"),2,5)),
         title = substr(title,1,regexpr("\\(\\d\\d\\d\\d\\)",title)-2))

#create test and train sets
set.seed(1,sample.kind = "Rounding")
test_index <- createDataPartition(y=edx$rating,times = 1,p = 0.2,list = FALSE) # test set = 20%
train_set <- edx %>% slice(-test_index)
temp <- edx %>% slice(test_index)
#cleanup
rm(test_index)

test_set <- temp %>% semi_join(train_set,by = "movieId","userId")
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
#cleanup
rm(temp,removed)

#Average rating of all movies
mu <- mean(train_set$rating)

#RMSE for mu
rmse_mu <- RMSE(mu,test_set$rating)

#create table to store method and RMSE
RMSE_table <- data.frame(method="Average Model",RMSE=rmse_mu)

#movie bias 
bi <- train_set %>% group_by(movieId) %>%  # grouping by the movies
  summarize(b_i = mean(rating - mu))   # mean of the residual

# predict with mean and movie bias
predict_mu_bi <- 
  test_set %>% left_join(bi,by = "movieId") %>% 
  mutate(mu_bi = mu + b_i) # get the movie bias and add it to average rating

#RMSE for mu and bi
rmse_mu_bi <- RMSE(predict_mu_bi$mu_bi,test_set$rating)
#add it to the table
RMSE_table <- bind_rows(RMSE_table,data.frame(method="Average and Movie bias Model",RMSE=rmse_mu_bi))

# User bias
bu <- train_set %>% left_join(bi,by = "movieId") %>% #join the movie bias table
  group_by(userId) %>%  # grouping by the users and then movies that they rated
  summarize(b_u = mean(rating - mu - b_i )) # mean of the residual

#Predict with mean,movie and user bias
predict_mu_bi_bu <- 
  test_set %>% left_join(bi,by = "movieId") %>% # join test set and train set movie bias on movie id
  left_join(bu,by = "userId") %>% # join test set and train set user bias on user id
  mutate(mu_bi_bu = mu + b_u + b_i) # get the movie bias, user bias and add it to average rating

#RMSE for mu and bi and bu
rmse_mu_bi_bu <- RMSE(predict_mu_bi_bu$mu_bi_bu,test_set$rating)
#add it to the table
RMSE_table <- bind_rows(RMSE_table,data.frame(method="Average, Movie bias and User bias Model",
                                              RMSE=rmse_mu_bi_bu))
bg <- 
  train_set %>% 
  left_join(bi, by="movieId") %>%
  left_join(bu, by="userId") %>%
  group_by(genres) %>%    # join the tables geneated earlier and group them by genre
  summarize(b_g = mean(rating-b_u-b_i-mu))
  
#Predict
predict_mu_bi_bu_bg <- 
  test_set %>% 
  left_join(bi, by = "movieId") %>%
  left_join(bu, by = "userId") %>%
  left_join(bg, by = "genres") %>%
  mutate(mu_bi_bu_bg = mu + b_i + b_u + ifelse(is.na(b_g),0,b_g)) 
# not all genres would have been captured in trainset and hence some movies 
# in the testset may generate an NA value for genre bias 

rmse_mu_bi_bu_bg <- RMSE(predict_mu_bi_bu_bg$mu_bi_bu_bg,test_set$rating) 

RMSE_table <- bind_rows(RMSE_table,data.frame(
  method="Average, Movie,User and Genre bias Model",RMSE=rmse_mu_bi_bu_bg))

#Regularization

#function to get the regularized RMSE for the lamda passed
rmse_reg <- function(lambda){
  #Train on trainset
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_u - b_i - mu)/(n()+lambda)) 
  #Predict on testset
  predict_rating <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(mu_bi_bu_bg = mu + b_i + b_u + b_g)
  #RMSE
  return(RMSE(predict_rating$mu_bi_bu_bg, test_set$rating))
}
# we found that 5.0 is what generates the lowest RMSE
rmse_regu <- rmse_reg(5.0)
RMSE_table <- bind_rows(RMSE_table,data.frame(
  method="Regularized Average, Movie, User and Genre bias Model",RMSE=rmse_regu))
#function to get the regularized RMSE for the lamda passed to final model
rmse_result <- function(lambda){
  #Train on edx
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <- edx %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_u - b_i - mu)/(n()+lambda)) 
  #Predict on validation
  predict_rating <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(mu_bi_bu_bg = mu + b_i + b_u + b_g)
  #RMSE
  return(RMSE(predict_rating$mu_bi_bu_bg, validation$rating))
}
# execute the function with lambda set to 5.0
rmse_final <- rmse_result(5.0)
# Result when run on validation set
RMSE_table <- bind_rows(RMSE_table,data.frame(
  method="Result when model run on validation set",RMSE=rmse_final))
RMSE_table %>% kable()









