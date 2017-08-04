library(data.table)
library(dplyr)
library(tidyr)
library(caret)

# Setting the directory to current location and reading data 
train.in <- fread("pml-training.csv")
test.in <- fread("pml-testing.csv")

## There are 19622 observations and 160 variables
dim(train.in)

## There are five classes from 'A' to 'E', data a bit more skewed to the label 'A', 
## but still, discrepancy is not that strong. 
table(train.in$classe)

# Transformation of the data

## Apparently there are "#DIV/0!"s in some columns and so they were coerced in the 
## wrond way to character vectors. The fix is to replace "#DIV/0!" to NA and coerce 
## them to numeric vectors.  
has.div.by.zero <- function(x) {
  sum(x == "#DIV/0!", na.rm = T) > 0
}

replace.div.by.zero <- function(x) {
  x[x == "#DIV/0!"] <- NA
  as.numeric(x)
}

train.in <- train.in %>% mutate_if(has.div.by.zero, replace.div.by.zero)

## Removing near zero vars
fit <- nearZeroVar(train.in)
train.in <- train.in %>% select(-fit)

## Removing predictors with many NA's
nas.ratios <- apply(train.in, 2, function(x) { sum(is.na(x)) / length(x) })

sum(nas.ratios < .1)
sum(nas.ratios < .95)

fit <- names(train.in)[nas.ratios < .1]
train.in <- train.in %>% select(fit)

## Removing columns with unique values
unique.ratios <- apply(train.in, 2, function(x) { length(unique(x)) / length(x) })

names(unique.ratios)[unique.ratios > .9]

fit <- names(train.in)[unique.ratios < .9]
train.in <- train.in %>% select(fit)

## Removing "cvtd_timestamp" column since it is duplicated by "raw_timestamp_part_1" and
## "raw_timestamp_part_2"
train.in <- train.in %>% select(-cvtd_timestamp)

## Transforming character columns to factors
### Theer are only two non numeric columns and one of them is "classe" column
is.numeric.columns <- sapply(train.in, is.numeric)
table(is.numeric.columns)
names(train.in)[!is.numeric.columns]

train.in <- train.in %>% mutate_if(!is.numeric.columns, as.factor)
columns.to.select <- train.in %>% select(-classe) %>% names
test.in <- test.in %>% select(columns.to.select)

## There are There are 19622 observations and 57 variables after after all removal procedures
dim(train.in)

library(ggplot2)

## Since there is only one factor variable it is easy to check how it is
## correlated with classe variable. There is no bias at all distribution
## is highly balanced across the class labels.
ggplot(train.in, aes(classe)) + 
  geom_bar(aes(fill = user_name)) +
  ggtitle("Distribution by users")

library(GGally)

set.seed(20170803)
columns.to.select <- train.in %>% select(-classe) %>% names
columns.to.plot <- sample(columns.to.select, 12)

to.plot.one <- train.in %>% select(columns.to.plot[1:6], classe) %>% sample_frac(0.2)
to.plot.two <- train.in %>% select(columns.to.plot[7:12], classe) %>% sample_frac(0.2)

ggpairs(to.plot.one, aes(colour = classe, alpha = 0.4))
ggpairs(to.plot.two, aes(colour = classe, alpha = 0.4))

rm(to.plot.one, to.plot.two)
gc()

# Building models
## Splitting data into train and test
inTrain <- createDataPartition(train.in$classe, p = 0.85)[[1]]

train = train.in[inTrain, ]
test = train.in[-inTrain, ]

fit.rf <- train(classe ~ ., data = train, method = "rf")
fit.gbm <- train(classe ~ ., data = train, method = "gbm")


