
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(gridExtra)
library(ggpubr)
library(reshape2)
library(data.table)
library(rstan)
library(rjson)
library(mvtnorm)
library(GetoptLong)
library(latex2exp)
rstan_options(auto_write = TRUE)





maxError = 0.27889007329940796
maxError = 1.
DEFAULT_N = 400
SMALL_S = 11

VARS_TO_PLOT = c(1:4,20,45)
#VARS_TO_PLOT = c(1,45)

#globals copied from python
MIN_DF = 2.5
SMEAN = 0. #ie, 1
SSCALE = 2.
DMEAN = 1. #ie, 2.7
DSCALE = 1.5
MIN_SIGMA_OVER_S = 1.9
###########

BASE_DIRECTORY = "../testresults"

var_names = c(TeX("$\\mu$"),TeX("$\\varsigma$"),TeX("$d$"))
for (i in 1:44) {
  var_names = c(var_names,TeX(qq("$T_{@{i}}$")))
}
              

all_guides = c("amortized_laplace",
               "unamortized_laplace",
               "meanfield")

guide_colors = c("red",
                 "blue",
                 "green")
guide_labels = c("amortized Laplace",
               "unamortized Laplace",
               "mean-field")
names(guide_colors) = all_guides
names(guide_labels) = all_guides

SUBSAMPLE_NS = c(400, 150,50,25,12) 
subsample_line_types = c(1,2,3,4,5)
names(subsample_line_types) = as.character(SUBSAMPLE_NS)
subsample_labels = as.character(SUBSAMPLE_NS)
subsample_labels[1] = "un-subsampled"
names(subsample_labels) = as.character(SUBSAMPLE_NS)

#temp: no 400
SUBSAMPLE_NS = c(400,100,50,10) 
subsample_line_types = 1:length(SUBSAMPLE_NS)
names(subsample_line_types) = as.character(SUBSAMPLE_NS)
subsample_labels = as.character(SUBSAMPLE_NS)
names(subsample_labels) = as.character(SUBSAMPLE_NS)


PARTICLE_NS = c(1,3) 
particle_widths = c(.5,1.5)
names(particle_widths) = as.character(PARTICLE_NS)
particle_labels = as.character(PARTICLE_NS)
names(particle_labels) = c("one","three")#as.character(PARTICLE_NS)

graph_combo_nums =c(10,1,
                    10,3
                    ,50,1
                    ,50,3
                    ,100,1
                    ,100,3
                    #,400,1
                    #,400,3
)
graph_combos=t(matrix(graph_combo_nums,2,length(graph_combo_nums)/2))

ts = function(x){x}
dict = function(...){list(...)}

base_scale = 1.
modal_effect = 1.*base_scale
tdom_fat_params = dict(modal_effect=ts(modal_effect),
                       df=3.,
                       t_scale=10.)
#
ndom_fat_params = dict(modal_effect=ts(modal_effect),
                       df=3.,
                       t_scale=2.)
#
tdom_norm_params = dict(modal_effect=ts(modal_effect),
                        df=30.,
                        t_scale=10.)
#
ndom_norm_params = dict(modal_effect=ts(modal_effect),
                        df=30.,
                        t_scale=2.)




model = stan_model("../stan/multisite.stan")


specify_decimal = function(x, k=1) trimws(format(round(x, k), nsmall=k))
#testresults\fit_amortized_laplace_0_N400_S10_mu1.0_sigma-2.3025851249694824_nu2.5+exp-0.6931471824645996.csv
nameWithParams = function(filebase, trueparams, S=NA, N=DEFAULT_N){
  if (is.na(S)) {
    qq("@{filebase}_N@{N}_mu@{specify_decimal(trueparams$modal_effect)}_sigma@{specify_decimal(trueparams$t_scale)}_nu@{specify_decimal(trueparams$df)}.csv")
  } else {
    qq("@{filebase}_N@{N}_S@{S}_mu@{specify_decimal(trueparams$modal_effect)}_sigma@{specify_decimal(trueparams$t_scale)}_nu@{specify_decimal(trueparams$df)}.csv")
  }
}

getScenario = function(params) {
  fread(nameWithParams(qq("@{BASE_DIRECTORY}/scenario"),params))
}

#ndom_norm_params = dict(modal_effect=ts(modal_effect),
#                        df=ts(3.),
#                        t_scale=ts(-1.))

toMCMClanguage = function(params,x) {
  giveVals = function(...) {
    result = list()
    result$mu = params$modal_effect
    result$d = params$df
    result$varsigma = params$t_scale
    result$T = x
    return(result)
  }
  return(giveVals)
}

getMCMCfor = function(params) {
  scenario = getScenario(params)
  
  
  afit = sampling(model, data = list(N=length(scenario[,s]), 
                                     se=scenario[,s], 
                                     x=scenario[,x],
                                     maxError=maxError,
                                     MIN_SIGMA_OVER_S=MIN_SIGMA_OVER_S,
                                     mindf=MIN_DF,
                                     smean=SMEAN,
                                     dmean=DMEAN,
                                     dscale=DSCALE,
                                     sscale=SSCALE)
                  #,init=toMCMClanguage(params,scenario[,x])
                  )
  amat = as.matrix(afit)
  return(amat)
}
getRawFitFor = function(params,S,guide ="amortized_laplace",particles=1,iter=0){
  
  jsonName = nameWithParams(qq("@{BASE_DIRECTORY}/fit_@{guide}_@{iter}_parts@{particles}"),params,S)
  #print(jsonName)
  fittedGuide = fromJSON(file=jsonName)
  if (guide=="meanfield") {
    hess = diag(c(fittedGuide$mode_sigma,
                  fittedGuide$ltscale_sigma,
                  fittedGuide$ldfraw_sigma,
                  fittedGuide$t_part_sigma))
    mean = c(fittedGuide$mode_hat,
             fittedGuide$ltscale_hat,
             fittedGuide$ldfraw_hat,
             fittedGuide$t_part_hat
             )
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
  
  jsonName = nameWithParams(qq("@{BASE_DIRECTORY}/fit_@{guide}_0"),params,S)
  #print(jsonName)
  fittedGuide = fromJSON(file=jsonName)
  if (guide=="meanfield") {
    hess = diag(c(fittedGuide$mode_sigma,
                  fittedGuide$ltscale_sigma,
                  fittedGuide$ldfraw_sigma,
                  fittedGuide$t_part_sigma))
    mean = c(fittedGuide$mode_hat,
             fittedGuide$ltscale_hat,
             fittedGuide$ldfraw_hat,
             fittedGuide$t_part_hat
    )
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


















###########################################################################################
#graphing stuff
###########################################################################################
graph_mcmc = function(samples, graphs=list(), vars_to_plot=VARS_TO_PLOT, base_format=get_base_formatting()) {
  for (i in vars_to_plot) {
    ii = toString(i)
    if (!(ii %in% names(graphs))) {
      graphs[[ii]] = add_base_formatting(
        ggplot(data.table(mcmc=samples[,i]), aes(x=mcmc)) + #cheating to force axis
        geom_histogram(aes(y=..density..)) +
        labs(x = var_names[i]))
        
        
    }
    #print("raw")
    #print(graphs[[ii]])
  }
  return(graphs)
}

add_base_formatting = function(rawgraph, guides=all_guides, subsamples=SUBSAMPLE_NS) {
  
  return(rawgraph +
     scale_colour_manual(name="Guide family",
                      values=guide_colors,
                      labels=guide_labels) +
    scale_linetype_manual(name="Subsampling?",
                         values =subsample_line_types,
                         labels = subsample_labels) +
    scale_size_manual(name="Guide samples/step",
                          values = particle_widths,
                          labels = particle_labels))
}

add_coverages = function(graphs, mymean, mycovar, guide, S, particles, vars_to_plot=VARS_TO_PLOT) {
  
  ridiculous_var = paste(guide,toString(S+particles+mymean+mycovar))
  #print(ridiculous_var)
  ridiculous_closure = function(graphs, mymean, mycovar, guide, S, particles, vars_to_plot) {
    newgraphs = list()
    for (i in vars_to_plot) {
      #print(qq("adding @{guide} @{S} @{i}"))
      ii = toString(i)
      newgraph = (graphs[[ii]] +
              stat_function(fun=dnorm, args = list(mean=mymean[i], sd = sqrt(mycovar[i,i])), 
                            aes(color=guide,
                                linetype=as.character(S),
                                size=as.character(particles)),
                                alpha=(particles)/3,
                            show.legend=TRUE) )
      newgraphs[[ii]] = newgraph
      
    }
    return(newgraphs)
  }
  return(ridiculous_closure(graphs, mymean, mycovar, guide, S, particles, vars_to_plot)) #I hate R sometimes...
}

#graph_coverages = function(samples, mymean, mycovar, guide, S) {
#  print(colnames(samples))
#  for (i in 1:4) {
#      print(ggplot(data.table(mcmc=c(samples[,i],
#                                     mymean[i])), aes(x=mcmc)) + #cheating to force axis
#              geom_histogram(aes(y=..density..)) +
#              scale_colour_manual(name="Which is which?",
#                                  values=c(red="red", blue="blue"),
#                                  labels=c(red="Truth (is out of style)", blue="Fiction (still imitating truth)")) +
#              stat_function(fun=dnorm, args = list(mean=mymean[i], sd = sqrt(mycovar[i,i])), aes(colour="red"), show.legend=TRUE) +
#              stat_function(fun=dnorm, args = list(mean=mymean[i] + 1, sd = sqrt(mycovar[i,i])), aes(colour="blue"), show.legend=TRUE) +
#              labs(title=qq("@{guide}; @{S} subsamples"),
#                   x = var_names[i])) +
#      scale_colour_identity(name="Which is which?", guide="legend",
#                          #values=c(red="red", blue="blue"),
#                          labels=c(red="Truth (is out of style)", blue="Fiction (still imitating truth)"))
#  }
#}







###########################################################################################
#Bring it all together!
###########################################################################################
###########################################################################################
get_metrics_for = function(params,guides = all_guides, dographs=all_guides, subsample_ns=SUBSAMPLE_NS) {
  amat = getMCMCfor(params)
  leftelbows = list()
  coverages = list()
  print(paste("guides:",guides))
  graphs = graph_mcmc(amat)
  for (guide in guides) {
    for (line in 1:dim(graph_combos)[1]) {
      graph_combo = graph_combos[line,]
      S = graph_combo[1]
      particles = graph_combo[2]
      print(paste("guide:",guide,"S",S,"particles",particles))
      tryCatch({
        
        meanhess = getRawFitFor(params,S,guide,particles)
        mymean = meanhess$mean
        myhess = meanhess$hess
        d = meanhess$d
        covar = solve(myhess)
        #print("mean")
        #print(mean)
        dens = dmvnorm(amat[,1:d],mymean,covar,log=TRUE)
        #print(rbind(amat[99:100,1:d],mean,sqrt(1/diag(hess))))
        #print(paste(guide,head(dens)))
        if (!(guide %in% names(leftelbows))) {
          leftelbows[[guide]] = list()
          coverages[[guide]] = list()
          
        }
        leftelbows[[guide]][[toString(S)]] = klOfLogdensities(amat[,48],dens)
        coverages[[guide]][[toString(S)]] = get_coverages(amat,mymean,covar)
        if (guide %in% dographs) {
          print(qq("adding @{guide} @{S} @{mymean[1]}"))
          #print(mymean)
          print(paste("mean[1] is",mymean[1],S,particles))
          graphs = add_coverages(graphs,mymean,covar,guide,S,particles)
        }
        #guide = "meanfield"
      }, error = function(e) {
        print(qq("FAILED to add @{guide} S@{S} part@{particles}"))
      })
    }
  }
  #print(names(graphs))
  g2 = list()
  for (i in 1:length(names(graphs))) {
    name = names(graphs)[i]
    #print(name)
    g2[[i]] = graphs[[name]]
    #print(graphs[[name]])
  }
  print("gonna print")
  #print(ggarrange(grobs=graphs,ncol=3, nrow=2, common.legend = TRUE, legend="bottom"))
  p = do.call(ggarrange,c(g2, list(ncol=3, nrow=2, common.legend = TRUE, legend="bottom")))
  print(length(p))
  print(p)
  print(" printed")
  return(list(leftelbows=leftelbows,coverages=coverages))
}


klOfLogdensities = function(a,b) {
  return(mean(a) - mean(b))
}

fat = TRUE
if (fat) {
  fat_metrics = get_metrics_for(ndom_fat_params,dographs=all_guides[c(1,3)])
} else {
  norm_metrics = get_metrics_for(ndom_norm_params)
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

