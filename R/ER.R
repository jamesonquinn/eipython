
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
library(tidyr)

EIdata = fread("../eiresults/scenario_N527.csv")
eiun = EIdata %>% unite(vari,var,i)
eiwide = eiun %>% pivot_wider(id_cols=u,names_from=vari,values_from=val) %>% data.table
eiwide[,n:=v_0+v_1+v_2]
eiwide[,v0f:=v_0/n]
eiwide[,n0f:=n_0/n]
eiwide[,n1f:=n_1/n]
eiwide[,n2f:=n_2/n]
lm(v0f ~ n0f + n1f + n2f,eiwide)
lm(v0f ~ n0f + n1f + n2f,eiwide)
