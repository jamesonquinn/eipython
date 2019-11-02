setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dirname(rstudioapi::getActiveDocumentContext()$path)

library(data.table); library(reshape); library(dplyr)

# Read in raw data into two data frames
#"elections" contains voting data by precinct
# "reg" contains voter registration demographics by precinct

years = c("2008", "2012", "2016")
election_dates = c("20081104", "20121106",  "20161108")
# values of variables that differ slightly by year
PRES <-c("PRESIDENT AND VICE PRESIDENT OF THE UNITED STATES",  "PRESIDENT AND VICE PRESIDENT OF THE UNITED STATES",  "US PRESIDENT")
GOV <- c("GOVERNOR", "NC GOVERNOR", "NC GOVERNOR")


# Initialize dataframes: one for election data, one for voter registration data
elections <- data.frame(matrix(ncol = 7, nrow = 0))
colnames(elections) <- c("county", "precinct", "precinct_type", "year", "contest", "party","numvotes")

reg <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(reg) <- c("county","precinct","year","race","numreg")


for (i in 1:length(years)){
#for (i in 1:1){
  # read in election data file
  election_data_file <-paste("RawData/results_pct_",election_dates[i],".txt", sep="")
  print(paste("Processing ", election_data_file))
  separator = ifelse(as.numeric(years[i]) <= 2012, ",", "\t")
  election_data <- read.table(election_data_file,header=TRUE,sep=separator,row.names=NULL,fill=TRUE,stringsAsFactors = FALSE, skipNul = TRUE)
  # rename variables as needed
  if(as.numeric(years[i]) >= 2014) {
    newvars = c("county","precinct","contest","party","numvotes")
    oldvars = c("County", "Precinct", "Contest.Name", "Choice.Party", "Total.Votes")
    setnames(election_data, oldvars, newvars)
  }
  else {
    setnames(election_data,c("total.votes"),c("numvotes"))
  }
  # restrict to elections we care about
  election_data$contest <- trimws(election_data$contest)
  election_data$contest[election_data$contest == "US SENATE"] <- "SEN"
  election_data$contest[election_data$contest == GOV[i]] <- "GOV"
  election_data$contest[election_data$contest == PRES[i]] <- "PRES"
  election_data <- subset(election_data,subset = (election_data$contest %in% c("SEN","GOV","ALL","PRES")))
  # delete all votes for other parties (we'll consider them the same as non-voters)
  election_data$party <- trimws(election_data$party)
  election_data <- subset(election_data,subset = (election_data$party %in% c("DEM","REP") & election_data$numvotes!=0))
  print(cast(election_data,contest~party,value = "numvotes", sum))
  # Make variable party_code, which is 1 for Democratic voters, 2 for Republican voters
  #election_data$party_code <- rep.int(2,dim(election_data)[1])
  #election_data$party_code[election_data$party=="DEM"] <- 1
  # add variable showing whether precinct is real
  election_data$precinct <- trimws(election_data$precinct)
  election_data$precinct_type <- rep(c("REAL"),dim(election_data)[1]) 
  election_data$precinct_type[substring(election_data$precinct,1,13) == "ABS/PROV/CURB"] <- "ABS/PROV/CURB"
  election_data$precinct_type[substring(election_data$precinct,1,13) == "ABS/CURB/PROV"] <- "ABS/PROV/CURB"
  election_data$precinct_type[substring(election_data$precinct,1,5) == "ABSEN"] <- "ABSENTEE"
  election_data$precinct_type[election_data$precinct == "MAIL ABSENTEE"] <- "ABSENTEE"
  election_data$precinct_type[substring(election_data$precinct,1,8) == "ONE STOP"] <- "ONESTOP"
  election_data$precinct_type[substring(election_data$precinct,1,8) == "ONE-STOP"] <- "ONESTOP"
  election_data$precinct_type[substring(election_data$precinct,1,2) == "OS"] <- "ONESTOP"
  election_data$precinct_type[substring(election_data$precinct,1,8) == "CURBSIDE"] <- "CURBSIDE"
  election_data$precinct_type[substring(election_data$precinct,1,5)  == "PROVI"] <- "PROVISIONAL"
  election_data$precinct_type[substring(election_data$precinct,1,8)  == "TRANSFER"] <- "TRANSFER"
  election_data$precinct_type[election_data$precinct == ""] <- "NONE"
  # add year
  election_data$year <- rep(c(years[i]),dim(election_data)[1]) 
  # add to master data frame
  elections <- rbind(elections,election_data[colnames(elections)])
  
  # read in registration data file   
  reg_data_file <-paste("RawData/voter_stats_",election_dates[i],".txt", sep="")
  print(paste("Processing ", reg_data_file))
  reg_data <- read.table(reg_data_file,header=TRUE,sep="\t",row.names=NULL, fill=TRUE,stringsAsFactors = FALSE, skipNul = TRUE)
  # rename variables
  vars = c("county","precinct","race","numreg")
  oldvars = c("county_desc", "precinct_abbrv", "race_code","total_voters")
  setnames(reg_data, oldvars,vars)
  # replace race with "H" for Hispanic where applicable, 
  #reg_data$ethnic_code <-trimws(reg_data$ethnic_code)
  #reg_data$race <-trimws(reg_data$race)
  #reg_data$race[(reg_data$ethnic_code=="HL")] <- "H"
  # Replace race with "other" for all non-{W,B,H}
  reg_data$race[((reg_data$race!="B") & (reg_data$race!= "H") & (reg_data$race!= "W")) | is.na(reg_data$race)] <- "other"  
  # collapse aross irrelevant variables (age, sex)
  reg_data$precinct <-trimws(reg_data$precinct)
  reg_data<-cast(reg_data,county+precinct+race~.,value="numreg",sum)
  setnames(reg_data, "(all)","numreg")
  #add year
  reg_data$year <- rep(c(years[i]),dim(reg_data)[1]) 
  #add to master data frame
  reg <- rbind(reg,reg_data[colnames(reg)])
}  








save(reg, file = "ProcessedData/NC_reg_data.Rdata")
save(elections, file = "ProcessedData/NC_election_data.Rdata")

#rm(list=ls())




  
