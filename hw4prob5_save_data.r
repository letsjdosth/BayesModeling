library(datasets)
data(mtcars)
head(mtcars)
lm(qsec~., data=mtcars)
length(mtcars[,1])


write.table(mtcars, "C:/gitProject/BayesModeling/dataset/mtcars.csv", row.names=FALSE, sep=",")
