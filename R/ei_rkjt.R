
setwd(dirname(rstudioapi::getActieDocumentContext()$path))

library(ggplot2)
library(gridExtra)
library(ggpubr)
library(reshape2)
library(data.table)
library(eiPack)
library(tidyr)
library(rjson)
library(mvtnorm)
library(GetoptLong)
library(latex2exp)
library(xtable)

samp_eis = function(filename="../eiresults_penultimate/scenario_SIG0.02_0_N527.csv") {
    rawNC = fread(filename)
    
    nc_obs = rawNC[var %in% c("n","v"),]
    names(nc_obs)
    nc_obs[,valname := paste0(var,r,c)]
    nc_obs_only = nc_obs[,list(u,valname,val)]
    
    nc_wide = spread(nc_obs_only, valname, val)
    eibayes = ei.MD.bayes(cbind(vNA0,vNA1,vNA2)~cbind(n0NA,n1NA,n2NA), data=nc_wide)
    #cover.plot(eibayes,1,1)
    lamei = lambda.MD(eibayes,c("vNA0","vNA1","vNA2"))
    #densityplot(lamei)
    dim(lamei)
    #densityplot(lamei)
    
    totn = nc_obs[var=="n",sum(val),by=r][,V1]
    
    l2m1 = matrix(lamei,1000,9)
    
    l2m2 = t(matrix(t(l2m1) * totn,9,1000))
    return(l2m2)
}


data_dir = "../eiresults.penultimate/"
all.fits = list.files(data_dir, pattern="fit.*.json",full.names=T)
all.samps = list.files(data_dir, pattern="samps.*.json",full.names=T)

fit.filename = all.fits[1]
samp.filename = all.samps[1]

fit = fromJSON(file=fit.filename)
R = 3
C = 3

samp=fread(samp.filename)
sparts = strsplit(samp.filename,"_")[[1]]
scenario.filename = paste(paste0(data_dir,"scenario"),sparts[5],sparts[6],sparts[7],sep="_")
scenario.filename = paste0(scenario.filename,".csv")
scenario = fread(scenario.filename)
Y = c()
for (rr in 0:2) {
  for (cc in 0:2) {
    Y = c(Y, scenario[var=="y"&r==rr&c==cc,sum(val+1)])
  }
}
Ymat = matrix(Y,3,3)

sampei = samp_eis(scenario.filename)
summary(sampei)
summary(samp - 1)
