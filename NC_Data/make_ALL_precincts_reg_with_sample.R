rm(list=ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dirname(rstudioapi::getActiveDocumentContext()$path)

library(data.table); library(plyr); library(sampling)
set.seed(129)
# how many precincts do we want to sample?
s = 60

#what year?
y="2016"

load("ProcessedData/NC_reg_data.Rdata")
data = data.table(reg)[year == y]
data = dcast(data,county+precinct~race, value.var="numreg")
data=data[,.(county,precinct,white_reg=W,black_reg=B,other_reg=other)]

N= dim(data)[1]
sampling_probabilities = rep(s/N,N)
data[,test:=UPminimalsupport(sampling_probabilities)]
write.csv(data, quote = FALSE, file = paste("ALL_precincts_",y,"_reg_with_sample_",s,".csv", sep=""))
