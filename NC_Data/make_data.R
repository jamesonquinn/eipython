rm(list=ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dirname(rstudioapi::getActiveDocumentContext()$path)

library(data.table); library(plyr)

load("ProcessedData/NC_reg_data.Rdata")
load("ProcessedData/NC_election_data.Rdata")
load("ProcessedData/usable_elections.Rdata")
load("ProcessedData/usable_reg.Rdata")
load("ProcessedData/usable_counties.Rdata")
load("ProcessedData/standardized_precincts.Rdata")

elections <- setDT(elections)
reg <- setDT(reg)

years = c("2008","2012","2016")
data_for_analysis <- data.table(
  county = character(),
  precinct = character(),
  year = character(),
  contest = character(),
  total_reg = numeric(),
  black_reg = numeric(),
  white_reg = numeric(),
  other_reg = numeric(),
  pct_white_reg = numeric(),
  numvotes = numeric(),
  party = character(),
  num_usable_years = numeric()
)

cast_reg <- setDT(dcast(reg,county+precinct+year ~ race, value.var = "numreg"))[precinct!=""]


for (y in years){
  uy <- paste("usable",y, sep="")
  pey <- paste("p_e",y, sep="")
  pry <- paste("p_r",y, sep="")
  
  data = elections[year==y & precinct_type=="REAL"]
  data[,precinct_type:=NULL]
  setnames(data, "precinct", pey)
  data <- merge(data,standardized_precincts, by=c("county",pey))
  
  yreg = cast_reg[year==y]
  setnames(yreg, "precinct",pry)
  yreg[,year:=NULL]
  
  data <- merge(data,yreg, by=c("county",pry))
  data<-data[data[[uy]]==TRUE]
  data[is.na(W), W:=0]
  data[is.na(B), B:=0]
  data[is.na(other), other:=0]
  # combine data for a single precinct (because in some years it might have corresponded to multiple precincts)
  data <- data[, .(numvotes = sum(numvotes), white_reg = sum(W), black_reg = sum(B), other_reg = sum(other)), by = .(county,precinct,year,contest,party,num_usable_years)]
  # Make the "voted for neither" rows
  data[,total_reg:=white_reg+black_reg+other_reg]
  neither <- data[, .(allvotes = sum(numvotes)), by = .(county,precinct,year,contest,num_usable_years,total_reg,white_reg,black_reg,other_reg)]
  neither[, `:=`(numvotes = total_reg - allvotes, party = "NEITHER", allvotes = NULL)]
  data <- rbindlist(list(data,neither),use.names=TRUE)
  # compute precent white in each precinct
  data[,pct_white_reg:=white_reg/total_reg]    
  # order the variables the way we like them
  data<-data[,.(county, precinct, year, contest, party, numvotes,num_usable_years,total_reg,white_reg,black_reg,other_reg, pct_white_reg)][order(county,precinct,contest,party)]
  # copy this year's data to master data
  data_for_analysis <- rbindlist(list(data_for_analysis,data), use.names=TRUE) 
}  

save(data_for_analysis, file = "ProcessedData/data_for_analysis.Rdata")
rm(list=ls())
load("ProcessedData/data_for_analysis.Rdata")
