library(tidytuesdayR)
library(lubridate)
raw <- tidytuesdayR::tt_load(2020, week = 18)$grosses
raw$year = lubridate::year(raw$week_ending)
data <- subset(raw, year >= 2000)
tab <- table(data$show)
sub <- names(tab)[which(tab > 370)]
data <- subset(data, show %in% sub)
# Change the outcome to million dollar scales
broadway <- data.frame(gross = data$weekly_gross / 1e6,
show = data$show,
week_ending = data$week_ending,
id.show = match(data$show, sub),
id.year = data$year - 1999,
id.week = data$week_number)
head(broadway)
# simple scatter plot
library(ggplot2)
ggplot(broadway) + aes(x = week_ending, y = gross) + geom_point(size = .5) + facet_wrap(~show)
write.table(broadway, "C:/gitProject/BayesModeling/dataset/broadway.csv", row.names=FALSE, sep=",")
