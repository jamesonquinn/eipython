
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
library(xtable)
rstan_options(auto_write = TRUE)

all.fits = list.files("../eiresults/", pattern="*.json")

fit.filename = all.fits[1]

fit = fromJSON(file=fit.filename)
R = 3
C = 3


BASE_DIRECTORY = "../testresults"


fitNameWithParams = function(filebase, trueparams, S=NA, N=DEFAULT_N){
  if (is.na(S)) {
    qq("@{filebase}_N@{N}_mu@{specify_decimal(trueparams$modal_effect)}_sigma@{specify_decimal(trueparams$t_scale)}_nu@{specify_decimal(trueparams$df)}.csv")
  } else {
    qq("@{filebase}_N@{N}_S@{S}_mu@{specify_decimal(trueparams$modal_effect)}_sigma@{specify_decimal(trueparams$t_scale)}_nu@{specify_decimal(trueparams$df)}.csv")
  }
}

getScenario = function(params,N) {
  fread(nameWithParams(qq("@{BASE_DIRECTORY}/scenario"),params,N=N))
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

getMCMCfor = function(params,N) {
  scenario = getScenario(params,N)
  print(paste("dims",dim(scenario),toString(names(scenario))))
  mymaxError = max(scenario[,s])
  print(paste(min(scenario[,s]),mymaxError,MIN_SIGMA_OVER_S,
              MIN_DF,SMEAN,DMEAN,DSCALE,SSCALE))

  afit = sampling(model, data = list(N=length(scenario[,s]),
                                     se=scenario[,s],
                                     x=scenario[,x],
                                     maxError=mymaxError,
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
getRawFitFor = function(params,S,guide ="amortized_laplace",particles=1,iter=0,N=DEFAULT_N){

  jsonName = nameWithParams(qq("@{BASE_DIRECTORY}/fit_@{guide}_@{iter}_parts@{particles}"),params,S,N=N)
  print(jsonName)
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
  return(list(mean=mean,hess=hess,d=d,fit=fittedGuide))
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
  print(paste("dimsamp",dim(samples)))
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
                          labels = particle_labels) +
    EXTRA_FORMATTING)
}

add_true_vals = function(graphs,params) {
  ridiculous_closure = function(graphs, params) {
    newgraphs = list()
    for (ii in names(graphs)) {
      if (ii=="1") {

        newgraph = (graphs[[ii]] +
                      geom_vline(xintercept=params$modal_effect))
      } else if (ii=="2") {

        newgraph = (graphs[[ii]] +
                      geom_vline(xintercept=log(params$t_scale-MIN_SIGMA_OVER_S )))
      } else if (ii=="3") {
        newgraph = (graphs[[ii]] +
                      geom_vline(xintercept=log(params$df-MIN_DF)))

      } else {
        newgraph = graphs[[ii]]
      }
      newgraphs[[ii]] = newgraph

    }
    return(newgraphs)
  }
  return(ridiculous_closure(graphs, params)) #I hate R sometimes...

}

add_coverages = function(graphs, mymean, mycovar, guide, S, particles, vars_to_plot=VARS_TO_PLOT) {
  print(paste("adding",toString(mymean[1:4]),toString(mycovar[1:4])))
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
                                #alpha=(particles)/3,
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

ELBOfrom = function(f) {

  tryCatch({
    ELBO = f$fit$mean_loss
    if (is.null(ELBO)) {
      return(NA)
    } else {
      return(ELBO)
    }
  }, error = function(e) {
    return(NA)
  })
}


get_metrics_for = function(params,N=DEFAULT_N,guides = all_guides, dographs=all_guides, subsample_ns=SUBSAMPLE_NS, graph_truth=TRUE) {
  amat = getMCMCfor(params,N)
  print(qq("Dimensions of @{toString(dim(amat))}, OK?"))
  metrics = data.table()
  print(paste("guides:",guides))
  graphs = graph_mcmc(amat)
  if (graph_truth) {
    graphs = add_true_vals(graphs,params)
  }
  for (iter in ITERS_TO_CHECK) {
    for (guide in guides) {
      for (line in 1:dim(graph_combos)[1]) {
        graph_combo = graph_combos[line,]
        S = graph_combo[1]
        particles = graph_combo[2]
        print(paste("guide:",guide,"S",S,"particles",particles))
        tryCatch({

          meanhess = getRawFitFor(params,S,guide,particles,iter=iter,N=N)
          mymean = meanhess$mean
          myhess = meanhess$hess
          d = meanhess$d
          covar = solve(myhess)
          #print("mean")
          #print(mean)
          dens = dmvnorm(amat[,1:d],mymean,covar,log=TRUE)
          #print(rbind(amat[99:100,1:d],mean,sqrt(1/diag(hess))))
          #print(paste(guide,head(dens)))
          EUBO = klOfLogdensities(amat[,d+1],dens)
          coverage = get_coverages(amat,mymean,covar)
          newmetrics = data.table(guide=guide,
                                  modal_effect=params$modal_effect,
                                     df=params$df,
                                     t_scale=params$t_scale,
                                     subsample_sites=S,
                                     particles=particles,
                                     iter=iter,
                                     EUBO=EUBO,
                                     coverage1=coverage[1,1],
                                     coverage2=coverage[2,1],
                                     coverage3=coverage[3,1],
                                     coverageT=coverage[4,1],
                                     ELBO = ELBOfrom(meanhess)
                                     )
          print(qq("dnm: @{dim(newmetrics)} dm: @{dim(metrics)}"))
          metrics = rbind(metrics,newmetrics,
                          fill=TRUE)
          if (iter==ITER_TO_GRAPH) {
            if (guide %in% dographs) {
              print(qq("adding @{guide} @{S} @{mymean[1]}"))
              #print(mymean)
              print(paste("mean[1] is",mymean[1],S,particles))
              graphs = add_coverages(graphs,mymean,covar,guide,S,particles)
            }
          }
          #guide = "meanfield"
        }, error = function(e) {
          print(qq("FAILED to add @{guide} N@{N} S@{S} part@{particles} iter@{iter}"))
          print(e)
        })
      }
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
  return(metrics)
}


klOfLogdensities = function(a,b) {
  return(mean(a) - mean(b))
}

fat = TRUE
if (fat) {
  fat_metrics = get_metrics_for(ndom_fat_params,dographs=all_guides[c(1,4,5,6,7)])
  output_table = fat_metrics[,list(EUBO=mean(EUBO),
                    ELBO=mean(ELBO,na.rm=TRUE),
                    #N=length(ELBO),
                    MuCover=mean(coverage1),
                    SigCover = mean(coverage2),
                    DFcover = mean(coverage3),
                    Tcover = mean(coverageT)
                    ),
              by=list(df,subsample_sites,particles,guide)][subsample_sites==100][order(particles),][order(guide),]
  print(xtable(output_table,digits=c(0,1,0,0,0,0,0,0,3,3,3,3)),include.rownames=FALSE)
} else {
  norm_metrics = get_metrics_for(ndom_norm_params)

  ITER_TO_GRAPH = 3
  graph_combo_nums =c(100,3
  )
  graph_combos=t(matrix(graph_combo_nums,2,length(graph_combo_nums)/2))
  norm_metrics_mini = get_metrics_for(ndom_norm_params)

  ouput_table2 = norm_metrics[,list(EUBO=mean(EUBO),
                                  ELBO=mean(ELBO,na.rm=TRUE),
                                  N=length(ELBO),
                                  MuCover=mean(coverage1),
                                  SigCover = mean(coverage2),
                                  DFcover = mean(coverage3),
                                  Tcover = mean(coverageT)
  ),
  by=list(df,subsample_sites,particles,guide)][subsample_sites==100][order(particles),][order(guide),]
  xtable(ouput_table2)
}
if (false) {
  ITER_TO_GRAPH = 10
  graph_combo_nums =c(44,3
  )
  graph_combos=t(matrix(graph_combo_nums,2,length(graph_combo_nums)/2))
  get_metrics_for(dummy_echs_params,graph_truth = FALSE,N=44)

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
