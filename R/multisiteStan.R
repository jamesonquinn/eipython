
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
DEFAULT_N = 44
SMALL_S = 22

#globals copied from python
MIN_DF = 2.5
SMEAN = 0 #ie, 1
SSCALE = 1
DMEAN = 1 #ie, 2.7
DSCALE = 1.5

var_names = c("mu","sigma","df","T[1]")

all_guides = c("amortized_laplace",
               "unamortized_laplace",
               "meanfield")


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

nameWithParams = function(filebase, trueparams, S=NA, N=DEFAULT_N){
  if (is.na(S)) {
    qq("@{filebase}_N@{N}_mu@{specify_decimal(trueparams$modal_effect)}_sigma@{specify_decimal(trueparams$t_scale)}_nu@{specify_decimal(MIN_DF)}+exp@{specify_decimal(trueparams$df)}.csv")
  } else {
    qq("@{filebase}_N@{N}_S@{S}_mu@{specify_decimal(trueparams$modal_effect)}_sigma@{specify_decimal(trueparams$t_scale)}_nu@{specify_decimal(MIN_DF)}+exp@{specify_decimal(trueparams$df)}.csv")
  }
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
                                     mindf=MIN_DF,
                                     smean=SMEAN,
                                     dmean=dMEAN,
                                     dscale=DSCALE,
                                     sscale=SSCALE))
  amat = as.matrix(afit)
  return(amat)
}

getRawFitFor = function(params,S,guide ="amortized_laplace"){
  
  jsonName = nameWithParams(qq("../testresults/fit_@{guide}_0"),params,S)
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
             fittedGuide$ahat_data$t_scale_raw,
             log(fittedGuide$df - MIN_DF),
             fittedGuide$ahat_data$t_part)
    
  }
  return(list(mean=mean,hess=hess,d=d))
}



getFitFor = function(params,S,guide ="amortized_laplace"){
  
  jsonName = nameWithParams(qq("../testresults/fit_@{guide}_0"),params,S)
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
             fittedGuide$ahat_data$t_scale_raw,
             log(fittedGuide$df - MIN_DF),
             fittedGuide$ahat_data$t_part)
    
  }
  return(list(mean=mean,hess=hess,d=d))
}

get_coverages = function(samples, mymean, mycovar, alpha=c(0.05,.5)) {
  z_interval = qnorm(1-alpha/2)
  raw_result = matrix(NA,length(mymean),length(alpha))
  for (i in 1:length(mymean)) {
    for (j in 1:length(alpha)) {
      raw_result[i,j] = mean(mymean[i] - z_interval[j] * sqrt(mycovar[i,i]) < samples[,i] &
                               samples[,i] < mymean[i] + z_interval[j] * sqrt(mycovar[i,i]))
    }
  }
  result = raw_result[1:4,]
  result[4,] = colMeans(raw_result[4:length(mymean),])
  return(result)
}


graph_coverages = function(samples, mymean, mycovar, guide, S) {
  print(colnames(samples))
  for (i in 1:4) {
      print(ggplot(data.table(mcmc=c(samples[,i],
                                     mymean[i])), aes(x=mcmc)) + #cheating to force axis
              geom_histogram(aes(y=..density..)) +
              stat_function(fun=dnorm, args = list(mean=mymean[i], sd = sqrt(mycovar[i,i]))) +
              labs(title=qq("@{guide}; @{S} subsamples"),
                   x = var_names[i]))
            
  }
}

get_metrics_for = function(params,guides = all_guides, dographs=all_guides) {
  amat = getMCMCfor(params)
  leftelbows = list()
  coverages = list()
  print(paste("guides:",guides))
  for (guide in guides) {
    for (S in c(DEFAULT_N,SMALL_S) ) {
      print(paste("guide:",guide))
      meanhess = getRawFitFor(params,S,guide)
      mean = meanhess$mean
      hess = meanhess$hess
      d = meanhess$d
      covar = solve(hess)
      #print("mean")
      #print(mean)
      dens = dmvnorm(amat[,1:d],mean,covar,log=TRUE)
      #print(rbind(amat[99:100,1:d],mean,sqrt(1/diag(hess))))
      #print(paste(guide,head(dens)))
      if (!(guide %in% names(leftelbows))) {
        leftelbows[[guide]] = list()
        coverages[[guide]] = list()
        
      }
      leftelbows[[guide]][[toString(S)]] = klOfLogdensities(amat[,48],dens)
      coverages[[guide]][[toString(S)]] = get_coverages(amat,mean,covar)
      if (guide %in% dographs) {
        graph_coverages(amat,mean,covar,guide,S)
      }
      #guide = "meanfield"
    }
  }
  return(list(leftelbows=leftelbows,coverages=coverages))
}

fat_metrics = get_metrics_for(ndom_fat_params)
norm_metrics = get_metrics_for(ndom_norm_params,dographs=c())

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
