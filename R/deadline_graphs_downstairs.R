
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

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

filename="../eiresults/scenario_SIG0.02_0_N2774.csv"
samp_eis = function(filename="../eiresults/scenario_SIG0.02_0_N2774.csv") {
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
  totn <<- nc_obs[var=="n",sum(val),by=r][,V1]
  
  l2m1 = matrix(lamei,1000,9)
  
  l2m2 = t(matrix(t(l2m1) * totn,9,1000))
  return(l2m2)
}
samp_eis_Q = function(filename="../eiresults/scenario_SIG0.02_0_N2774.csv") {
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
  totn <<- nc_obs[var=="n",sum(val),by=r][,V1]
  
  l2m1 = matrix(lamei,1000,9)
  
  l2m2 = t(matrix(t(l2m1) * totn,9,1000))
  return(l2m1)
}

samp_ei_betas = function(filename="../eiresults/scenario_SIG0.02_0_N2774.csv") {
  rawNC = fread(filename)
  
  nc_obs = rawNC[var %in% c("n","v"),]
  names(nc_obs)
  nc_obs[,valname := paste0(var,r,c)]
  nc_obs_only = nc_obs[,list(u,valname,val)]
  
  nc_wide = spread(nc_obs_only, valname, val)
  eibayes = ei.MD.bayes(cbind(vNA0,vNA1,vNA2)~cbind(n0NA,n1NA,n2NA), 
                        data=nc_wide,ret.beta='r',ret.mcmc=F)
  #cover.plot(eibayes,1,1)
  lamei = lambda.MD(eibayes,c("vNA0","vNA1","vNA2"))
  #densityplot(lamei)
  alldims = dim(eibayes$draws$Beta)
  betas = eibayes$draws$Beta
  R = alldims[1]
  C = alldims[2]
  U = alldims[3]
  S = alldims[4]
  #densityplot(lamei)
  
  totn = nc_obs[var=="n",sum(val),by=r][,V1]
  totu = nc_obs[var=="n",sum(val),by=u][,V1]
  
  totu.betaform = array(totu,c(U,R,C,S))
  totu.betaform = aperm(totu.betaform,c(2,3,1,4))
  betavars = apply(betas, c(1,2,4), var)
  binomvars = betas * (1 - betas) / totu.betaform
  ns = array(nc_obs[var=="n",][order(r),val],c(U,R))
  vs = array(nc_obs[var=="v",][order(c),val],c(U,R))
  effdim_n = 1/(apply((ns/totu)**2, 1, sum))
  effdim_v = 1/(apply((vs/totu)**2, 1, sum))
  binomvar_squish = sqrt(1/effdim_n/effdim_v) #fuck, I don't know, dude. Something like that?
  binomvar_squish.betaform = array(binomvar_squish,c(U,R,C,S))
  binomvar_squish.betaform = aperm(binomvar_squish.betaform,c(2,3,1,4))
  mbinomvars = apply(binomvars*binomvar_squish.betaform, c(1,2,4), mean)
  qvars = betavars + mbinomvars
  
  
  l2m1 = matrix(lamei,1000,9)
  
  l2m2 = t(matrix(t(l2m1) * totn,9,1000))
  return(list(ytots=l2m2,
              betavars=betavars,
              mbinomvars=mbinomvars,
              binomvar_squish=binomvar_squish,
              qvars = qvars))
}

data_dir = "../eiresultsQ3"
all.fits = list.files(data_dir, pattern="fit.*.json",full.names=T)
all.samps = list.files(data_dir, pattern="dsamps.*.csv",full.names=T)


fit.filename = all.fits[1]
samp.filename = all.samps[1]

fit = fromJSON(file=fit.filename)
R = 3
C = 3

samp=fread(samp.filename)
msamp = as.matrix(samp)
psamp= msamp[,c(1,4,7,2,5,8,3,6,9)+7]
sparts = strsplit(samp.filename,"_")[[1]]
scenario.filename = paste(paste0(data_dir,"/scenario"),sparts[5],sparts[6],sparts[7],sep="_")
scenario.filename = paste0(scenario.filename,".csv")
scenario = fread(scenario.filename)
Y = c()
for (rr in 0:2) {
  for (cc in 0:2) {
    Y = c(Y, scenario[var=="y"&r==rr&c==cc,sum(val)])
  }
}
Ymat = matrix(Y,3,3)

scenario[var=="y",n:=sum(val),by=list(r,u)]

scenario[var=="y",Q:=val/n]
s_Q = matrix(scenario[var=="y",var(Q),by=list(r,c)][,V1],3,3)
m_Q = matrix(scenario[var=="y",mean(Q),by=list(r,c)][,V1],3,3)

sampei = samp_eis_Q(scenario.filename)
summary(sampei)
summary(psamp - 1)
Ymat


dim(msamp)
7+9+4+27
mysQ = matrix(NA,400,9)
mymQ = matrix(NA,400,9)
firstq = 21
for (ix in 0:8) {
  mysQ[,ix+1] = msamp[,firstq+ix*3+2] / msamp[,firstq+ix*3]
  mymQ[,ix+1] = msamp[,firstq+ix*3+1] 
}
head(mysQ)
head(sqrt(mysQ))
s_Q
head(mymQ)
m_Q

mstd = function(x) c("Mean"=round(mean(x)*100,1),
                     "Std"=round(sd(x)*100,3))
apply(sampei, 2, mstd)
totn
qsamp = psamp
for (rr in 0:2) {
  for (cc in 0:2) {
    qsamp[,1+rr+cc*3] = (qsamp[,1+rr+cc*3]-U) / totn[rr+1]
  }
}
colnames(qsamp) = paste0("vv",1:9)
colnames(sampei) = paste0("vv",1:9)
t(apply(sampei, 2, mstd))
t(apply(qsamp, 2, mstd))
mysum = t(apply(qsamp, 2, mstd))

t(round(m_Q*100, 1))
