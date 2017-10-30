#########################  1. set up path and initialization #################
## setup working directory
setwd("C:/Users/Lele/Documents/R/win-library/3.3/KaggleCases/breast cancer case")
# open the installed 'easypackages' library, which can install and open multiple packages simutaneously
library(easypackages)
# load all needed packages
libraries("readr", "dplyr", "ggplot2", "corrplot", "gridExtra", "pROC", "MASS", "caTools", "caret",
          "caretEnsemble", "reshape2", "randomForest", "nnet")

## initialize setting parameters
# split ratio for splitting training and testing data
split_ratio <- 0.7

# set the fitControl
fitControl <- trainControl(method="cv",  # k-fold cross-validation
                           number = 10,   # the folds No. in K-fold cv or No. of bootstraping resampling iterations
                           # whether class probabilities should be computed for held-out samples during resample
                           classProbs = TRUE, 
                           # a function to evaluate performance summaries
                           # "twoClassSummary": compute the sensitivity, specificity and area under the ROC curve
                           summaryFunction = twoClassSummary)


######################### 2. load and wrangling data ########################
data <- read.csv("C:/Users/Lele/Documents/R/win-library/3.3/KaggleCases/breast cancer case/data.csv")
# check each column or variable name and type
str(data)
# #33 column are all N/A, delete this column
data[,33] <- NULL
# look at some statistical analysis for each column
summary(data[,2:ncol(data)])
# the range difference for each variable is huge, so need to normalize in preprocessing

## Data Splicing into training and testing set
# get a reproducible random sequence for later
set.seed(1234)
# using bootstraping random selection to select split_ratio (e.g. 70%) of data as training data
#split_ratio <- 0.7   # already set at the beginning
data_index <- createDataPartition(data$diagnosis, p = split_ratio, list = FALSE)
# 70% randomly selected data goes to train set, omit the first column (ID #)
train_data <- data[data_index, -1]
# the rest 30% goes to test set, omit the first column
test_data <- data[-data_index, -1]
# check the diagnosis proportion for M and B, which is to calculate M/sum(M+B) and B/sum(M+B)
prop.table(table(train_data$diagnosis))

## wrangling data on training set
## check the boxplot distribution for all variables with "_mean", can repeat this for "_se" and "_worst"
# select all variables with "_mean" in variable names vector
vn <- variable.names(train_data)
meanInd <- grep("_mean", vn)
mean_data <- train_data[, c(1, meanInd)]      # combine all mean variables with diagnosis
mean_m <- melt(mean_data)                     # group all variables based on diagnosis
head(mean_m)
# using featurePlot in caret package for boxplot all mean values by diagnosis
f1 <- featurePlot(x = mean_data[, 2:11], 
            y = mean_data$diagnosis, 
            plot = "box", 
            ## format x and y axis and layout
            scales = list(y = list(relation="free")),  
            layout = c(5,2))
# save figure1 as png file
png("boxplot_mean.png", width = 5*500, height = 5*300, res = 200)
print(f1)
dev.off()
# display figure1 in plot panel
f1

# using featurePlot in caret package for densityplot all mean values by diagnosis
f4 <- featurePlot(x = mean_data[, 2:11],
                  y = mean_data$diagnosis,
                  plot = "density",
                  ## format x and y axis and layout, 'relation = free' means no same x and y axis range
                  scales = list(y = list(relation="free"), x = list(relation="free")),
                  layout = c(5,2))
# save figure1 as png file
png("density_mean.png", width = 5*500, height = 5*300, res = 200)
print(f4)
dev.off()
# display figure1 in plot panel
f4

## check the boxplot/density distribution for all variables with "_worst"
worstInd <- grep("_worst", vn)
worst_data <- train_data[, c(1, worstInd)]      # combine all worst variables with diagnosis
worst_m <- melt(worst_data)                     # group all variables based on diagnosis
#head(worst_m)
# using featurePlot in caret package for boxplot all worst values by diagnosis
f2 <- featurePlot(x = worst_data[, 2:11], 
                  y = worst_data$diagnosis, 
                  plot = "box", 
                  ## format x and y axis and layout
                  scales = list(y = list(relation="free")),  
                  layout = c(5,2))
# save figure1 as png file
png("boxplot_worst.png", width = 5*500, height = 5*300, res = 200)
print(f2)
dev.off()
# display figure1 in plot panel
f2

# using featurePlot in caret package for densityplot all worst values by diagnosis
f3 <- featurePlot(x = worst_data[, 2:11],
                  y = worst_data$diagnosis,
                  plot = "density",
                  ## format x and y axis and layout, 'relation = free' means no same x and y axis range
                  scales = list(y = list(relation="free"), x = list(relation="free")),
                  layout = c(5,2))
# save figure1 as png file
png("density_worst.png", width = 5*500, height = 5*300, res = 200)
print(f3)
dev.off()
# display figure1 in plot panel
f3


## check the boxplot/density distribution for all variables with "_se"
seInd <- grep("_se", vn)
se_data <- train_data[, c(1, seInd)]      # combine all se variables with diagnosis
se_m <- melt(se_data)                     # group all variables based on diagnosis
#head(se_m)
# using featurePlot in caret package for boxplot all se values by diagnosis
f5 <- featurePlot(x = se_data[, 2:11], 
                  y = se_data$diagnosis, 
                  plot = "box", 
                  ## format x and y axis and layout
                  scales = list(y = list(relation="free")),  
                  layout = c(5,2))
# save figure1 as png file
png("boxplot_se.png", width = 5*500, height = 5*300, res = 200)
print(f5)
dev.off()
# display figure1 in plot panel
f5

# using featurePlot in caret package for densityplot all worst values by diagnosis
f6 <- featurePlot(x = se_data[, 2:11],
                  y = se_data$diagnosis,
                  plot = "density",
                  ## format x and y axis and layout, 'relation = free' means no same x and y axis range
                  scales = list(y = list(relation="free"), x = list(relation="free")),
                  layout = c(5,2))
# save figure1 as png file
png("density_se.png", width = 5*500, height = 5*300, res = 200)
print(f6)
dev.off()
# display figure1 in plot panel
f6


######################### 3. Modeling #############################
# use the same seed for all models for reproducibility and comparison
set.seed(825)
## 1) Logistic Regression model
model_lr <- train(diagnosis~., train_data, 
                  # generalized linear model
                  method = "glm", 
                  preProc = c("center", "scale"),
                  trControl = fitControl)
model_lr
# prediction
pred_lr <- predict(model_lr, test_data)
# performance
cm_lr <- confusionMatrix(pred_lr, test_data$diagnosis, positive = "M")
#cm_lr

## 2) classification tree model
model_tree <- train(diagnosis~.,
                    train_data,
                    method = "rpart",
                    preProcess = c('center', 'scale'),
                    trControl = fitControl)
model_tree
# prediction
pred_tree <- predict(model_tree, test_data)
# performance
cm_tree <- confusionMatrix(pred_tree, test_data$diagnosis, positive = "M")
# plot tree models
png("Classification Tree.png", width = 5*500, height = 5*300, res = 200)
plot(model_tree$finalModel, uniform = TRUE, main = "Classification Tree Model")
text(model_tree$finalModel, use.n = TRUE, all = TRUE, cex = 0.8)
dev.off()


## 3) random forest model
#set.seed(825)
model_rf <- train(diagnosis~.,
                  train_data,
                  method = "rf",  
                  prox = TRUE, 
                  preProcess = c('center', 'scale'),
                  trControl = fitControl)
model_rf
# prediction
pred_rf <- predict(model_rf, test_data)
# performance
cm_rf <- confusionMatrix(pred_rf, test_data$diagnosis, positive = "M")
#cm_rf


## 4) svm model
#set.seed(825)
model_svm <- train(diagnosis~., 
                   train_data, 
                   # svm with Radial Basis Function Kernel
                   method = "svmRadial", 
                   preProc = c("center", "scale"),
                   trControl = fitControl)
# prediction
pred_svm <- predict(model_svm, test_data)
# performance
cm_svm <- confusionMatrix(pred_svm, test_data$diagnosis, positive = "M")
#cm_svm


#### compare between models
model_list <- list(LR = model_lr,
                   tree = model_tree,
                   RF = model_rf,
                   SVM = model_svm) 
# summarize fitted results in each model
resamps <- resamples(model_list)
summary(resamps)

# plot ROC, sensitivity and specificity for each model
#bwplot(resamps, metric = "ROC")
png("Model Comparison.png", width = 600, height = 500, res = 120)
f7 <- bwplot(resamps, layout = c(4, 1))
print(f7)
dev.off()
f7

## compare confusion matrix
cm_list <- list(LR = cm_lr, 
                tree = cm_tree, 
                RF = cm_rf, 
                SVM = cm_svm)
cm_results <- sapply(cm_list, function(x) x$byClass)
cm_results
cm_results_max <- apply(cm_results, 1, which.is.max)
output_report <- data.frame(metric=names(cm_results_max), 
                            best_model=colnames(cm_results)[cm_results_max],
                            value=mapply(function(x,y) {cm_results[x,y]}, 
                                         names(cm_results_max), 
                                         cm_results_max))
rownames(output_report) <- NULL
output_report


############################## 4. Exploring Features #####################
## compare variable importance
lrImp <- varImp(model_lr)
treeImp <- varImp(model_tree)
rfImp <- varImp(model_rf)
svmImp <- varImp(model_svm)
# rank all variables according to importance, and color-coded y labels according to "_worst","_mean","_se" group
png("Variable Importance.png", width = 5*500, height = 5*300, res = 250)
v1 <- plot(treeImp, top = 5, main = "Classification Tree", 
           scales = list(y = list(cex = 1.2, col = c("red","blue","blue","blue","blue"))))
v2 <- plot(lrImp, top = 5, main = "Logistic Regression",
           scales = list(y = list(cex = 1.2, col = c("red","red","green","green","red"))))
v3 <- plot(rfImp, top = 5, main = "Random Forest",
           scales = list(y = list(cex = 1.2, col = c("red","blue","blue","blue","blue"))))
v4 <- plot(svmImp, top = 5, main = "svmRadial",
           scales = list(y = list(cex = 1.2, col = c("blue","red","blue","blue","blue"))))
# layout four plots in specific format
grid.arrange(v1, v3, v4, v2, nrow = 2, ncol = 2)
dev.off()
grid.arrange(v1, v3, v4, v2, nrow = 2, ncol = 2)


## 1) boxplot for all area values
# select all the variables containing "area" and melt them with diagnosis
areaInd <- grep("area", vn)
area_data <- train_data[, c(1, areaInd)]
area_m <- melt(area_data)
# two types of boxplot side by side
png("area_boxplot.png", width = 5*500, height = 5*300, res = 300)
#par(cex.axis = 1.5, cex.lab = 1.5)
a1 <- qplot(diagnosis, value, data = area_m, fill = variable, geom = c("boxplot")) + theme(legend.position = "top")
a2 <- ggplot(data = area_m, aes(x = diagnosis, y = value)) + 
      geom_boxplot() + geom_jitter(aes(colour = variable)) + theme(legend.position = "top")
grid.arrange(a2, a1, ncol = 2)
dev.off()
grid.arrange(a2, a1, ncol = 2)

## 2) boxplot for concave.points
concaveInd <- grep("concave.points", vn)
concave_data <- train_data[, c(1, concaveInd)]
concave_m <- melt(concave_data)
# two types of boxplot side by side
png("concave_boxplot.png", width = 5*250, height = 5*300, res = 350)
#par(cex.axis = 1.5, cex.lab = 1.5)
#c1 <- qplot(diagnosis, value, data = concave_m, fill = variable, geom = c("boxplot")) + theme(legend.position = "top")
c2 <- ggplot(data = concave_m, aes(x = diagnosis, y = value)) + 
      geom_boxplot() + geom_jitter(aes(colour = variable)) + 
      theme(legend.position=c(0,1), legend.justification=c(0,1), legend.background=element_blank())
#grid.arrange(c2, c1, ncol = 2)
print(c2)
dev.off()
#grid.arrange(c2, c1, ncol = 2)
c2

## 3) boxplot for radius
radiusInd <- grep("radius", vn)
radius_data <- train_data[, c(1, radiusInd)]
radius_m <- melt(radius_data)
# two types of boxplot side by side
png("radius_boxplot.png", width = 5*250, height = 5*300, res = 350)
#par(cex.axis = 1.5, cex.lab = 1.5)
#r1 <- qplot(diagnosis, value, data = radius_m, fill = variable, geom = c("boxplot")) + theme(legend.position = "top")
r2 <- ggplot(data = radius_m, aes(x = diagnosis, y = value)) + 
      geom_boxplot() + geom_jitter(aes(colour = variable)) + 
      theme(legend.position=c(0,1), legend.justification=c(0,1), legend.background=element_blank())
print(r2)
#grid.arrange(r2, r1, ncol = 2)
dev.off()
#grid.arrange(r2, r1, ncol = 2)
r2

## for loop t-test between B and M  for each variable 
# initialize the pvalue matrix for all variables [30, 1]
ttestp <- numeric(ncol(train_data) - 1)
for (i in 2:ncol(train_data))
{
  y <- train_data[, i]
  # independent 2 group t-test, unequal variance
  t <- t.test(y ~ train_data$diagnosis)
  ttestp[i-1] <- t$p.value
}
# put all p values in data frame with corresponding variable names
ttest_df <- data.frame(Variables = colnames(train_data)[2:ncol(train_data)], Pvalues = ttestp)
# sort the pvalue descendingly
ttest_df <- ttest_df[order(ttest_df$Pvalues), ]
ttest_df
