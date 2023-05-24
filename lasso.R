df_all <- read.csv("data_malignant.csv")
sapply(df_all, function(x) sum(is.na(x)))
data <- na.omit(data)
dimension <- dim(df_all)
dimension

