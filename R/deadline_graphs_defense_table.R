
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
library(psych)
library(GetoptLong) #qq

samp_eis = function(filename="xxx"){#../eiresults/scenario_SIG0.02_0_N2774.csv") {
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


data_dir = "../ei_results_2020.3"
all.fits = list.files(data_dir, pattern="fit.*.json",full.names=T)
all.samps = list.files(data_dir, pattern="Dsamps.*.csv",full.names=T)
all.fits
all.samps

fit.filename = all.fits[7]#2]
samp.filename = all.samps[4]#1]

fit = fromJSON(file=fit.filename)
R = 3
C = 3

asPercentStr = function(val,ndigits=3) {
  return(paste0(formatC(signif(val,digits=ndigits), digits=ndigits,format="fg", flag="#"),"\\%"))
}

makeResultTable = function(samp.filename,fit.filename,dodescribe=F) {
    samp=fread(samp.filename)
    msamp = as.matrix(samp)
    VIsamp= msamp[,c(1,4,7,2,5,8,3,6,9)+7]
    sparts = strsplit(samp.filename,"_")[[1]]
    print(sparts[c(5,10,7,6,3)])
    scenario.filename = paste(paste0(data_dir,"/scenario"),sparts[7],0,sparts[9],sep="_")
    scenario.filename = paste0(scenario.filename,".csv")
    scenario = fread(scenario.filename)
    Y = c()
    for (rr in 0:2) {
      for (cc in 0:2) {
        Y = c(Y, scenario[var=="y"&r==rr&c==cc,sum(val+1)])
      }
    }
    Ymat = matrix(Y,3,3)
    
    Ksamp = samp_eis(scenario.filename)
    tot = sum(Ymat)
    rtots = rowSums(t(Ymat))
    
    
    
    
    if (dodescribe) {
      print(describe(Ksamp/tot*100))
      print(describe((VIsamp - 1)/tot*100))
      
    } else {
      summary(Ksamp)
      summary(VIsamp - 1)
      Ymat
    }
    round(t(Ymat)/rtots*100,1)
    cat(paste0(as.character(c(t(round(t(Ymat)/rtots*100,1)))),"\\%"))
    sparts[5]
    
    results = matrix("",9,8)
    races = c("White","Black","Other")
    
    truthstrs = paste0(as.character(c(t(round(t(Ymat)/rtots*100,1)))),"\\%")
    truthmat = t(Ymat)/rtots*100
    for (r in 1:3) {
      baserow = r * 3 - 2
      results[baserow,1] = qq("\\multirow{3}{*}{@{races[r]}}")
      results[baserow,2] = qq("Truth")
      results[baserow+1,2] = qq("RJKT")
      results[baserow+2,2] = qq("Laplace")
      
      for (c in 1:3) {
        basecol = c*2+1
        datacol = (c-1)*3 + r
        
        #Truth
        results[baserow,basecol] = asPercentStr(truthmat[r,c])
        
        #RJKT
        rjkts = Ksamp[,datacol]*100/rtots[r]
        results[baserow+1,basecol] = asPercentStr(mean(rjkts))
        results[baserow+1,basecol+1] = asPercentStr(sd(rjkts),2)
        
        #Us
        rjkts = VIsamp[,datacol]*100/rtots[r]
        results[baserow+2,basecol] = asPercentStr(mean(rjkts))
        results[baserow+2,basecol+1] = asPercentStr(sd(rjkts),2)
        
      }
    }
    return(results)
}
#0.3
if (FALSE) {
  t1 = makeResultTable(samp.filename,fit.filename)
  print(xtable(t1),include.rownames = F,sanitize.text.function = identity)
}

results = data.table()
for (i in 1:length(all.samps)) {
  samp.filename = all.samps[i]
  sparts = strsplit(samp.filename,"_")[[1]]
  t1 = makeResultTable(samp.filename,fit.filename)
  print(t1)
  rr = data.table(file=samp.filename,
                  parts=sparts[5],samps=sparts[10],sig=sparts[7],steps=sparts[6],grad=sparts[3],
                  btrump=t1[6,7],bvar=t1[6,8],
                  latex = paste0(capture.output(print(xtable(t1),include.rownames = F,sanitize.text.function = identity)),collapse='\n'))
  results = rbind(results,rr,fill=T)
}

# 
# #0.3 wgrad
# fit.filename = all.fits[2]
# samp.filename = all.samps[7]
# t1 = makeResultTable(samp.filename,fit.filename)
# t1 #God, that's crap
# 
# #0.3 wgrad 2
# fit.filename = all.fits[4]
# samp.filename = all.samps[8]
# t1 = makeResultTable(samp.filename,fit.filename)
# t1 #God, that's crap

#0.02
fit.filename = all.fits[11]
samp.filename = all.samps[5]
t1 = makeResultTable(samp.filename,fit.filename)
print(xtable(t1),include.rownames = F,sanitize.text.function = identity)

grad = fit$big_grad[1:7]
gg_raw_inv = solve(matrix(unlist(fit$big_arrow$gg_raw),7,7))
gg_raw_inv %*% grad
