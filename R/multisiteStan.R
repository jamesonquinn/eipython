
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(reshape2)
library(data.table)
library(rstan)
library(rjson)
library(mvtnorm)
library(GetoptLong)
rstan_options(auto_write = TRUE)





maxError = 0.27889007329940796
min_DF = 2.7




model = stan_model("../stan/multisite.stan")

if (FALSE) { #old noodling-around code
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
  
  rawhess = unlist(o$raw_hessian)
  
  d = sqrt(length(rawhess))
  hess = matrix(rawhess,d,d)
  dim(hess)
  
  hess[0:8,0:5]
  
  #getDensity(vec, )
}

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
  qq("@{filebase}_N@{N}_mu@{specify_decimal(trueparams$modal_effect)}_sigma@{specify_decimal(trueparams$t_scale)}_nu@{specify_decimal(min_DF)}+exp@{specify_decimal(trueparams$df)}.csv")
}

getScenario = function(params) {
  fread(nameWithParams("../testresults/scenario",params))
}


getMCMCfor = function(params) {
  scenario = getScenario(params)
  
  
  afit = sampling(model, data = list(N=length(scenario[,s]), 
                                     se=scenario[,s], 
                                     x=scenario[,x],
                                     maxError=maxError,
                                     mindf=min_DF))
  amat = as.matrix(afit)
  return(amat)
}

getFitFor = function(params,guide ="amortized_laplace"){
  
  jsonName = nameWithParams(qq("../testresults/fit_@{guide}_0"),params)
  print(jsonName)
  fittedGuide = fromJSON(file=jsonName)
  if (guide=="meanfield") {
    hess = diag(fittedGuide$auto_scale**2)
    mean = fittedGuide$auto_loc
    d = length(mean)
  } else {
    rawhess = unlist(fittedGuide$raw_hessian)
    
    d = sqrt(length(rawhess))
    hess = matrix(rawhess,d,d)
    mean = c(fittedGuide$ahat_data$modal_effect, 
             maxError/2+exp(fittedGuide$ahat_data$t_scale_raw),
             fittedGuide$df,
             fittedGuide$ahat_data$t_part)
    
  }
  return(list(mean=mean,hess=hess,d=d))
}

all_guides = c("amortized_laplace",
           "unamortized_laplace",
           "meanfield")

get_kls_for = function(params,guides = all_guides) {
  amat = getMCMCfor(params)
  results = list()
  print(paste("guides:",guides))
  for (guide in guides) {
    print(paste("guide:",guide))
    meanhess = getFitFor(params,guide)
    mean = meanhess$mean
    hess = meanhess$hess
    d = meanhess$d
    covar = solve(hess)
    #print("mean")
    #print(mean)
    dens = dmvnorm(amat[,1:d],mean,covar,log=TRUE)
    #print(rbind(amat[99:100,1:d],mean,sqrt(1/diag(hess))))
    #print(paste(guide,head(dens)))
    results[[guide]] = klOfLogdensities(amat[,48],dens)
    #guide = "meanfield"
  }
  return(results)
}

klOfLogdensities = function(a,b) {
  return(mean(a) - mean(b))
}

arrowhead = function(a,b,c,n) {
  result = matrix(0,n,n)
  result[1,1] = a - 2 * b[1] - c[1]
  result[1,] = result[1,] + b
  result[,1] = result[,1] + b
  result = result + diag(c,n)
  return(result)
}


arrowblockhead = function(a,b,c,n,p) {
  result = matrix(0,n,n)
  result[1:p,1:p] = a - 2 * b[1:p,] - c[1]
  result[1:p,] = result[1:p,] + b
  result[,1:p] = result[,1:p] + t(b)
  result = result + diag(c,n)
  return(result)
}
