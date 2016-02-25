# The purpose of this script was to try the xgboost library, using a subset of
# my Pokemon dataset

require(jsonlite)
require(xgboost)
set.seed(151)

pokemon.data <- 
  fromJSON("https://agile-shelf-4613.herokuapp.com/pokemon/api/v1.0/pokedex")

df <- data.frame(pokemon.data$primary_type, 
                 pokemon.data$secondary_type, pokemon.data$region)

# Get just the dual-type Pokemon's
df <- df[complete.cases(df), ]
colnames(df) <- c('primary.type', 'secondary.type', 'region')

m <- as.matrix(sapply(df, as.numeric))

index <- sample(2, nrow(m), replace=TRUE, prob=c(0.8, 0.2))
train.data <- m[index == 1, ]
test.data <- m[index == 2, ]

# To fit an xgboost model, the data has to be in a xgb.DMatrix structure
d <- xgb.DMatrix(train.data[,1:2], label= as.matrix(train.data[,3] - 1))
t <- xgb.DMatrix(test.data[,1:2])

# cross validation model used to learn about the data, and
# to choose the parameters
# good example at: http://wiselily.com/2015/07/12/xgboost-data-mining-example-1/
history <- xgb.cv(data = d, nrounds = 200, num_class = 18,
                  objective='multi:softmax', nfold=2, prediction = TRUE,
                  eval_metric='mlogloss')
print (history)
cv.mlogloss.mean <- as.data.frame(history$dt)$test.mlogloss.mean
plot(cv.mlogloss.mean)

# Check which row has the lowest mlogloss mean, and get its value
nrounds <- cv.mlogloss.mean[which.min(cv.mlogloss.mean)]
nrounds <- which.min(cv.mlogloss.mean)


# The xgboost function is the simplest way to fit a model
# The function xgb.train is more advanced.
xgboost.model <- xgboost(data = d, nrounds = nrounds, num_class = 18, 
                         objective="multi:softmax", verbose = 2)

pred.result <- predict(xgboost.model, test.data[,1:2])
sum(pred.result + 1 == test.data[,3])