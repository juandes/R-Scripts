# Load the required packages and data
require(arules)
require(ggplot2)
require(randomForest)
require(dplyr)

set.seed(2512)

titanic <- read.csv("~/Desktop/titanic.csv", stringsAsFactors=FALSE)

# Display a summary of the data
summary(titanic)
str(titanic)

# Table with frequency of survivors and non-survivors
titanic.survive.table <- as.data.frame(table(titanic$Survived))
colnames(titanic.survive.table) <- c('Status', 'Frequency')
titanic.survive.table[,1] <- c('Did not survived', 'Survived')
show(titanic.survive.table)

# Plot of survivors
qplot(x = Status, y = Frequency, data = titanic.survive.table, geom='bar', stat='identity',
      fill = I('light blue'), main = 'Frequency of survivors and non-survivors')

# Histogram of passenger classes
qplot(Pclass, data = titanic, geom = 'histogram', binwidth = 0.6,
      fill = I('light blue'), main = 'Histogram of passenger classes')

# Number of passengers traveling in 2nd and 3rd class
second.class.total <- length(which(titanic$Pclass == 2))
third.class.total <- length(which(titanic$Pclass == 3))

# Distribution of group ages
summary(titanic$Age)
hist(titanic$Age, col = 'light blue', prob = TRUE, breaks = 12, 
     main = 'Distribution of group ages')
rug(titanic$Age)
lines(density(titanic$Age), col="blue", lwd = 2)
lines(density(titanic$Age, adjust = 2), lty="dotted", col="darkgreen",
      lwd = 2)

# Boxplot of ages
boxplot(titanic$Age, col = 'light blue', outcol = 'red', pch = 16,
        main = 'Boxplot of age distribution', ylab = 'Age')

# ASSOCIATION RULES SECTION

# Create the new attribute
titanic$Stage <- ifelse(titanic$Age <= 5, 0, 1)
show(table(titanic$Stage))

# Create new dataframe
titanic.asr <- data.frame(Sex = titanic$Sex, Stage = as.factor(titanic$Stage),
                          Survived = as.factor(titanic$Survived))

# Create apriori model
rules <- apriori(titanic.asr,
                 appearance = list(default = 'lhs', rhs = c('Survived=0', 'Survived=1')))
inspect(rules)


# PREDICTION PHASE

# Preparing the data for the random forest

# Create a new attribute which is the sum of SibSp and Parch
titanic$Total.Family <- titanic$SibSp + titanic$Parch
titanic$Survived <- as.factor(titanic$Survived)
titanic$Sex <- as.factor(titanic$Sex)
titanic$Stage <- as.factor(titanic$Stage)

# Create a new attribute made of the title of the name, e.g. mr, mrs and so on
Title.Prefix <- sapply(titanic$Name, function (x) gsub('[ .] ', '', strsplit(x, '.')[[1]][2]))
Title.Prefix <- as.vector(Title.Prefix)
titanic$Title.Prefix <- as.factor(Title.Prefix)

# Split the dataset into a training set and a test set
index <- sample(2, nrow(titanic), replace=TRUE, prob=c(0.7, 0.3))
train.data <- titanic[index == 1, ]
test.data <- titanic[index == 2, ]

# Build the model and predict
rf <- randomForest(Survived ~ Stage + Sex + Total.Family + SibSp + Pclass  + Title.Prefix,
                   data = train.data, ntree = 290, do.trace = FALSE, importance = TRUE)
print(rf)

# Prediction using the test data
pred <- predict(rf, test.data)
pred.confusion.matrix <- table(pred, test.data$Survived)
show(pred.confusion.matrix)
test.classification.error <- round((pred.confusion.matrix[2, 1] + pred.confusion.matrix[1, 2]) /
                                     sum(pred.confusion.matrix) * 100, 2)
show(test.classification.error)

plot(rf, main = 'Relation between the error percentage of the training phase and
     the number of trees')

# Calculate the feature importance
feature.importance <- importance(rf, 1)
feature.importance <- data.frame(feature = row.names(feature.importance), 
                                 importance = feature.importance[, 1])
feature.importance <- arrange(feature.importance, importance)
show(feature.importance)
ggplot(feature.importance, aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat="identity", fill="light blue") +
  ggtitle('Importance of each attribute') +
  xlab('Feature') +
  ylab('Importance')

