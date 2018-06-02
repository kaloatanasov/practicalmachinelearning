## Executive Summary:

## The following analysis is done on the Weight Lifting Exercise Dataset (Velloso,
## E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition
## of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation
## with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.). The
## data had been gathered from accelerometers on the belt, forearm, arm, and dumbbell
## of 6 participants. The following Course Project predicts the manner in which
## the exercise was performed, i.e. the "classe" variable, with five categories.
## The current report describes how the prediction model was built, the way cross
## validation was used, as well as in the sample and out of sample error estimates.

## Analysis:

## First, we load the data sets from the urls that were provided in the assignment.
## We need to specify that all blank cells within the training set need to be replaced
## with NA values.

fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileUrl, destfile = "pml-training.csv", method = "curl")

dataTraining <- read.csv("pml-training.csv", header = TRUE, na.strings = c("", "NA"))


fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileUrl2, destfile = "pml-testing.csv", method = "curl")

dataTesting <- read.csv("pml-testing.csv")

## Next, let's look at the structure and the summary of the training data set, in
## order to get beter understanding of the variables and assume the most appropriate
## cross-validation approach.

str(dataTraining)
summary(dataTraining$classe)

## As a next step, we need to perform data slicing and split the training data set
## to training and testing sets. We will use the new training set to build
## the model, the testing set to test and fine tune it. At the end we will use the
## original testing set, as provided in the project, to predict only once and calculate
## our out of sample error. We will use the caret package for data slicing and model
## fitting.

library(caret)

inTrain <- createDataPartition(y = dataTraining$classe, p = 0.7, list = FALSE)

training <- dataTraining[inTrain, ]
testing <- dataTraining[-inTrain, ]

## Next, we need to explore the data and find the best predictors, before we decide
## on the appropriate prediction model.

summary(training)

## First, let's check the proportion of missing values within each variable. We
## may need to establish a treshold for dimenshion reduction due to a large amount
## of missing data.

propmiss <- function(dataframe) {
        m <- sapply(dataframe, function(x) {
                data.frame(n=length(x),
                           nmiss=sum(is.na(x)),
                           propmiss=sum(is.na(x))/length(x))
        })
        d <- data.frame(t(m))
        d <- sapply(d, unlist)
        d <- as.data.frame(d)
        d$variable <- row.names(d)
        row.names(d) <- NULL
        d <- cbind(d[ncol(d)],d[-ncol(d)])
        return(d[order(d$propmiss), ])
}

missVal <- propmiss(training)

## Note: Function by Stephen Turner (article: "Summarize Missing Data for all
## Variables in a Data Frame in R", article url:
## http://www.gettinggeneticsdone.com/2011/02/summarize-missing-data-for-all.html)

round(missVal$propmiss, digits = 2)

sum(missVal$propmiss > 0.9)

## 62.5% of the variables, or 100 out of 160, have over 90% missing values within them.
## The rest of the 60 variables have 0% missing values. Theoretically, variables
## with over 20%-30% missing data can be excluded from the data set. With that said,
## we can set our treshold to 90%, which can exlude 100 variables from our data set,
## and exclude them as predictor variables when building the prediction model.

## With that said, let's start with our dimensionality reduction, by singling out
## the variables with missing values above 90%.

varNoNa <- missVal$variable[missVal$propmiss < 0.9]

training2 <- subset(training, select = varNoNa)

str(training2)
summary(training2)

## Next, let's check for variables with near zero variance, which is an indicator
## for a non-informative predictor variable.

nearZeroVar(training2, names = TRUE)
summary(training2$new_window)

## Since the "new_window" variable is a near zero variable, let's remove it from
## the training data set, as a possible predictor.

training3 <- training2[, -6]

str(training3)

## Next, let's use some domain knowledge and continue with our dimention reduction.
## Going further, predict exercise quality, the "classe" variable, based on the
## various measurements collected,we can assume that the variables "X", "user_name",
## "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", and "num_window"
## will not be useful. With that said, let's remove them from our training data set.

training4 <- training3[, -c(1:6)]

str(training4)

## After cleaning the data of the variables that will not be good predictors, let's
## run a model using Random Forests. Let's do it with parallel processing, to optimise
## the computation process.

x <- training4[, -53]
y <- training4[, 53]

## First, we configure the parallel processing.

library(parallel)
install.packages("doParallel")
library(doParallel)

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

## Next, we configure the trainControl object. We use K-fold cross-validation. We
## substitute the default boostrapping with K-fold in order to reduce the number
## of sampling from 25 to 5 folds, and reduce computation power.

fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

## Next, we develop the training model.

modFit <- train(x, y, method = "rf", data = training4, trControl = fitControl)

## Finally, as a very important step, we de-register parallel processing cluster.

stopCluster(cluster)
registerDoSEQ()

## Once we have our model fit, we can check its accuracy and kappa values.

modFit
modFit$resample
confusionMatrix.train(modFit)

## Analysis: With this model we achieved an Accuracy average of 0.9904, which
## makes an in the sample error of 0.0094 (or 1 - 0.9904).

## Next, let's make a table of the predictions from our model and see how accurately
## we can predict with it.

pred1 <- predict(modFit, testing)

testing1 <- testing
testing1$predRight1 <- pred1 == testing1$classe
table(pred1, testing1$classe)

## Next, we will apply model ensembling to improve the accuracy of our model. First,
## we fit 2nd model with boosting with trees, measure its performance, and later
## combine the two models for even better prediction.

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

modFit2 <- train(classe ~ ., method = "gbm", data = training4, verbose = FALSE)

stopCluster(cluster)
registerDoSEQ()

modFit2
modFit2$resample
confusionMatrix.train(modFit2)

## Analysis: With this model we achieved an Accuracy average of 0.9564, which
## makes an in the sample error of 0.0436 (or 1 - 0.9564).

pred2 <- predict(modFit2, testing)

testing2 <- testing
testing2$predRight2 <- pred2 == testing2$classe
table(pred2, testing2$classe)

qplot(pred1, pred2, colour = classe, data = testing, geom = "jitter")

## Analysis: As the plot of the two predictors against each other show, they are
## close, but do not exactly match.

## Next, we fit a model that combines the two predictors. First, we create a new
## data set consisted of the predictions from the two models, and create a "classe"
## variable, which is the "classe" variable from the testing data set. Then, we
## fit the combined model, which will have the "classe" variable as the outcome
## and the two prediction variables as the predictor variables.

predDF <- data.frame(pred1, pred2, classe = testing$classe)
combModFit <- train(classe ~ ., ethod = "gam", data = predDF)

combModFit
combModFit$resample
confusionMatrix.train(combModFit)

## Analysis: With this model we achieved an Accuracy average of 0.9926, which
## makes an in the sample error of 0.0074 (or 1 - 0.9926), which is the lowest of
## the three models so far.

## Next, we predict on the combined data set.

combPred <- predict(combModFit, predDF)

testingComb <- testing
testingComb$predRightComb <- combPred == testingComb$classe
table(combPred, testingComb$classe)

## Analysis: When we compare the three tables with predicted values vs. the real values
## of the outcome variable "classe" from the testing set (as part of the training
## set) we can see that the combined model improved the prediction by correctly
## predicting 4 more categories of the C way of weights lifting, making the
## combined model the most accurate at this point.

qplot(pred1, combPred, colour = classe, data = testing, geom = "jitter")

## Analysis: As the plot of the first, more accurate predictor, against the combined
## one show, they are much closer now.

## An expected ot of sample error can be calculated with 1 - the accuracy
## of the combined prediction model, raised on the 20 degree "because the
## probability of the total is equal to the product of the independent probabilities".
## As the formula below shows, that estimation is 0.1380434, or 14%.

expError <- 1 - 0.9926^20
expError

## Finally, we can estimate the out of sample error when we predict on the initial
## testing set, which was provided with the Course Project. Since we will predict
## 20 cases

predFinal1 <- predict(modFit, dataTesting)
predFinal2 <- predict(modFit2, dataTesting)

predFinalDF <- data.frame(pred1 = predFinal1, pred2 = predFinal2)

combFinalPred <- predict(combModFit, predFinalDF)
combFinalPred

## Finally, the true out of sample error can be calculated after submitting the
## quiz, as part of the Course Project, by predicting the 20 cases. In this particular
## project, all 20 cases were predicted correctly, which makes up for 100% accuracy,
## and thus 0% out of sample error.
