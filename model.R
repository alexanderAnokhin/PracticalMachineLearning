library(data.table)
library(dplyr)
library(tidyr)

# Setting directory to current location and reading data 
train.in <- fread("pml-training.csv")
test.in <- fread("pml-testing.csv")

## There are 19622 observations and 160 variables
dim(train.in)

library(caret)

fit <- nearZeroVar(train.in)
fit
table(train.in[, 13])
summary(train.in[, 13])

sum(train.in[, 13] == "#DIV/0!")

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

train.in2 <- train.in %>% mutate_if(has.div.by.zero, replace.div.by.zero)

# Removing near zero vars
fit <- nearZeroVar(train.in2)
train.in3 <- train.in2 %>% select(-fit)  

## There are There are 19622 observations and 124 variables after removing near zero variance variables.
