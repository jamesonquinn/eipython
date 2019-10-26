rm(list=ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dirname(rstudioapi::getActiveDocumentContext()$path)

library(data.table); library(plyr)
set.seed(129)

load("ProcessedData/data_for_analysis.Rdata")
sample=data_for_anaysis[contest = "PRES"]
