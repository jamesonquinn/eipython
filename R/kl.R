
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(reshape2)
library(data.table)

denses = fread("../testresults/demo_amortized_laplace.fitted.csv")
names(denses)
denses[,list(mean(as.numeric(modelKL)),
             mean(as.numeric(laplace)),
             mean(as.numeric(meanfield)), by=list(obs,df,sig)]
