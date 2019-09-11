
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(reshape2)
library(data.table)
library(rstan)
rstan_options(auto_write = TRUE)

model = stan_model("../stan/multisite.stan")

data = fread("../testresults/scenario_N44_mu1.0_sigma2.0_nu3.0.csv")

fit1 = sampling(model, data = list(N=length(data[,s]), se=data[,s], x=data[,x]))

plot(fit1)

data = fread("../testresults/scenario_N44_mu1.0_sigma2.0_nu-1.0.csv")

fit2 = sampling(model, data = list(N=length(data[,s]), se=data[,s], x=data[,x]))

plot(fit2)

fitframe = extract(fit1)
names(fitframe)
fitframe$T



