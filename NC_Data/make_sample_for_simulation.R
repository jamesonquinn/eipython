rm(list=ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dirname(rstudioapi::getActiveDocumentContext()$path)

library(data.table); library(plyr); library(sampling)
set.seed(129)
# how many precincts we want to sample
s = 60

load("ProcessedData/data_for_analysis.Rdata")
data=data_for_analysis[contest == "PRES" & year == 2016  & party == "NEITHER",.(county,precinct,white_reg,black_reg,other_reg)]
N= dim(data)[1]

sampling_probabilities = rep(s/N,N)
data[,test:=UPminimalsupport(sampling_probabilities)]
write.csv(data,file = "NC_precincts_2016_with_sample.csv")
