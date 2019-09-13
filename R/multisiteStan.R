
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(reshape2)
library(data.table)
library(rstan)
library(rjson)
library(mvtnorm)
library(GetoptLong)
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

fitmat = as.matrix(fit1)

dim(fitmat)

head(fitmat)





o = fromJSON(file="../testresults/fit_amortized_laplace_0_N44_mu1.0_sigma-2.0_nu-1.0.csv")

rawhess = unlist(o$hessian)

d = sqrt(length(rawhess))
hess = matrix(rawhess,d,d)
dim(hess)

hess[0:8,0:5]

#getDensity(vec, )

ts = function(x){x}

base_scale = 1.
modal_effect = .5*base_scale
tdom_fat_params = list(modal_effect=ts(modal_effect),
                       df=ts(-1.),
                       t_scale=ts(2.))
#
ndom_fat_params = list(modal_effect=ts(modal_effect),
                       df=ts(-1.),
                       t_scale=ts(-2.))
#
tdom_norm_params = list(modal_effect=ts(modal_effect),
                        df=ts(3.),
                        t_scale=ts(2.))
#
ndom_norm_params = list(modal_effect=ts(modal_effect),
                        df=ts(3.),
                        t_scale=ts(-2.))


specify_decimal = function(x, k=1) trimws(format(round(x, k), nsmall=k))

nameWithParams = function(filebase, trueparams, N=44){
  qq("@{filebase}_N@{N}_mu@{specify_decimal(trueparams$modal_effect)}_sigma@{specify_decimal(trueparams$t_scale)}_nu@{specify_decimal(trueparams$df)}.csv")
}

getScenario = function(params) {
  fread(nameWithParams("../testresults/scenario",params))
}

maxError = 0.27889007329940796


getMCMCfor = function(params) {
  scenario = getScenario(params)
  guide ="amortized_laplace"
    
  jsonName = nameWithParams(qq("../testresults/fit_@{guide}_0"),params)
  print(jsonName)
  fittedGuide = fromJSON(file=jsonName)
  
  rawhess = unlist(fittedGuide$hessian)
  
  d = sqrt(length(rawhess))
  hess = matrix(rawhess,d,d)
  mean = c(fittedGuide$modal_effect, 
           maxError/2+exp(fittedGuide$t_scale_raw),
           fittedGuide$df,
           fittedGuide$t_part)
  
  
  
  
  afit = sampling(model, data = list(N=length(scenario[,s]), 
                                     se=scenario[,s], 
                                     x=scenario[,x],
                                     maxError=maxError))
  amat = as.matrix(afit)
  
  print(rbind(amat[99:100,1:d],mean,sqrt(1/diag(hess))))
  return(cbind(amat,dmvnorm(amat[,1:d],mean,solve(hess),log=TRUE)))
  #guide = "meanfield"
}
