##### Load packs and read in data #####
library(tidyverse)
library(readr)
library(fs)
library(caret)

setwd("~/05. Scripts/06. Fantasy Football/data/")
files <- list.files(path = getwd(), pattern = "[0-9].csv")
main <- data.frame()

for (i in files) {
  df <- read.csv(i)
  df$Tgt <- NULL
  df$Year <- rep(substr(files[i], start = 1, 4), nrow(df))
  main <- rbind(main, df)
}

main <- main %>% 
  mutate(Year = as.double(Year)) %>% 
  mutate(PPG = FantasyPoints/G) %>% 
  select(Player, Tm, Pos, Year, PPG)
  
colnames(main) <- tolower(colnames(main))

file_2020 <- read.csv(paste0(getwd(), "/", "2020_special.csv")) %>% 
  rename(ppg = `FPTS.G`, player = NAME, tm = TEAM, pos = POS) %>% 
  select(player, tm, pos, ppg) 

file_2020$year <-  rep(2020, nrow(file_2020))

main <- rbind(main, file_2020)

##### Create first year and active player dfs #####

first_year <- main %>% 
  group_by(player, pos) %>% 
  slice(which.min(year)) %>% 
  select(player, year)

active <- main %>% 
  group_by(player, pos) %>% 
  count()

##### Brief exploratory data analysis #####
unique <- main %>% 
  left_join(first_year, by = c('player', 'pos')) %>%
  mutate(active = year.x - year.y + 1) %>%  
  rename(year = year.x) %>% 
  select(-year.y)

unique
unique %>% filter(active>20)

unique %>% 
  filter(active <15 & pos != 0) %>%
  ggplot(mapping = aes(active, ppg, color = pos)) +  # active and pos
  geom_smooth(se=T)

unique %>% 
  filter(active <15 & pos != 0) %>% 
  ggplot(mapping = aes(pos, ppg)) +
  geom_boxplot()

unique %>% 
  filter(active <15 & pos != 0) %>% 
  ggplot(unique, mapping = aes(tm, ppg)) +
  geom_boxplot() +
  coord_flip()

##### Create modeling dataset, explore + think about transformations #####

active_players <- main %>% 
  filter(year == 2020) %>% 
  select(player, pos)

model_dat <- main %>% 
  semi_join(active_players, by = c('player', 'pos')) %>% 
  left_join(first_year, by = c('player', 'pos')) %>% 
  mutate(active = year.x - year.y + 1) %>% 
  select(-year.x, -year.y)

model_dat$pos[model_dat$pos == 0] <- 'other'
model_dat$tm <- factor(model_dat$tm)
model_dat$pos <- factor(model_dat$pos)

mean(model_dat$ppg)
sd(model_dat$ppg)
var(model_dat$ppg)

table(model_dat$tm)
table(model_dat$pos)
table(model_dat$active)

model_dat %>% count(ppg<0)
model_dat %>% filter(active > 19) %>% distinct(player)

model_dat <- model_dat %>% 
  filter(ppg>0) %>%  # remove non-zero items 
  filter(active < 20) # remove players over 20

hist(model_dat$ppg)
hist(sqrt(model_dat$ppg))

model_dat$ppg <- sqrt(model_dat$ppg) # transform the dependent variable

model_dat$player <- NULL # such that the model is agnostic of player as a factor
model_dat %>% head()

##### Generate training and hold out sets #####

set.seed(333)
index <- sample(nrow(model_dat),round(0.8*nrow(model_dat)))
train <- model_dat[index, ]
test <- model_dat[-index, ]

##### Train several models and hypertune parameters as needed #####

# set CV methodology (exception being random forest)
kfold <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# least squares
ols <- train(
  ppg ~ ., 
  data = train,
  method = 'lm',
  metric = 'RMSE',
  trControl = kfold
)

print(ols)
ols_lm <- lm(ppg ~ ., data = train)
plot(ols_lm)

# k-nearest neighbors
knn <- train(
  ppg ~ ., 
  data = train,
  method = 'knn',
  metric = 'RMSE',
  trControl = kfold, 
  tuneGrid = expand.grid(k = seq(5, 40, length.out = 8))
)
print(knn)

# MARS
mars <- train(
  ppg ~ .,
  data = train, 
  method = 'earth',
  metric = 'RMSE',
  trControl = kfold, 
  tuneGrid = expand.grid(
    degree = 1:3, 
    nprune = seq(2, 100, length.out = 10) %>% floor()
  )
)

print(mars)

# elastic net
enet <- train(
  ppg ~ .,
  data = train, 
  method = 'glmnet', 
  metric = 'RMSE', 
  trControl = kfold, 
  tuneLength = 10
)

print(enet)

rf <- train(
  ppg ~ .,
  data = train,
  method = 'rf',
  metric = 'RMSE',
  trControl = trainControl(method='boot', number=50, search='grid'), # bootstrap
  tuneGrid = expand.grid(mtry = c(1, 2, 3, 4))
)

print(rf)

knn$results[, 'RMSE'] %>% min()
mars$results[, 'RMSE'] %>% min()
enet$results[, 'RMSE'] %>% min()
ols$results[, 'RMSE'] %>% min()
rf$results[, 'RMSE'] %>% min()


##### Check each model on the holdout sets #####

mse_fun <- function(mod) {
  actuals <- test$ppg
  preds <- predict(mod, test, type = "raw") 
  summand <- 0
  for (i in 1:nrow(test)) {
    summand <- summand + sum(actuals[i]-preds[i])^2
  }
  mse <- summand/nrow(test)
  return(mse)
}

mse_fun(ols)
mse_fun(knn)
mse_fun(enet)
mse_fun(mars)
mse_fun(rf) # best model!

##### Fit best model to the entire dataset #####
model <- train(
  ppg ~ .,
  data = model_dat,
  method = 'rf',
  metric = 'RMSE',
  trControl = trainControl(method='boot', number=100), # bootstrap
  tuneGrid = expand.grid(mtry = which(row_number(rf$results[, 2]) == 1)) # optimal mtry
)

##### Prepare newdat, fit model, and produce results #####
newdat <- file_2020 %>% 
  left_join(first_year, by = c('player', 'pos')) %>% 
  mutate(active=year.x - year.y + 2) %>% 
  select(-ppg,-year.x,-year.y)

newdat$tm <- factor(newdat$tm)
newdat$pos[newdat$pos == 0] <- 'other'
newdat$pos <- factor(newdat$pos)
table(newdat$pos)

preds <- as.data.frame(predict(model, newdat, type = 'raw')^2)
output <- cbind(newdat, preds)
output <- 1:100

setwd('../output')
write.csv(output, "results.csv")

#####