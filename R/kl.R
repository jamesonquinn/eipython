
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(reshape2)
library(data.table)
library(rjson)
library(mvtnorm)

denses = fread("../testresults/demo_amortized_laplace.fitted.csv")
names(denses)
denses[,list(mean(as.numeric(modelKL)),
             mean(as.numeric(laplace)),
             mean(as.numeric(meanfield))), by=list(obs,df,sig)]
denses[obs=="7.0",]


fitting = fread("../testresults/demo_amortized_laplace.fitting.csv")
names(fitting)

fitting[,mean(tail(loss,400)), by=list(guidename,obs,df,sig)]

fitting[,list(meanfield=mean(tail(loss[guidename=="meanfield"],400)), laplace=mean(tail(loss[guidename=="laplace"],400))), by=list(obs,df,sig)]

o = fromJSON(file="../testresults/fit_amortized_laplace_0_N44_mu1.0_sigma-2.0_nu-1.0.csv")

rawhess = unlist(o$hessian)

d = sqrt(length(rawhess))
hess = matrix(rawhess,d,d)
dim(hess)

getDensity(vec, )

results = extract(fittin)
