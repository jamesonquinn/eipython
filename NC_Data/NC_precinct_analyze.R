rm(list=ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dirname(rstudioapi::getActiveDocumentContext()$path)

library(data.table); library(reshape); library(plyr)

load("ProcessedData/NC_election_data.Rdata")
load("ProcessedData/NC_reg_data.Rdata")

# Find the usable counties, i.e. in counties where 
# all the votes are assigned to precincts
data <- cast(elections, year + county + contest + precinct_type ~ ., value = "numvotes", sum)
data <- cast(data, year + county + precinct_type ~ ., value = "(all)", mean)
TOTAL <- cast(data, year + county ~ ., value = "(all)", sum)
setnames(TOTAL, "(all)", "votes")
data <- cast(data, year + county ~ precinct_type, value = "(all)")
data <- merge(data, TOTAL, by = c("county", "year"))
data$usable <- (data$REAL == data$votes)
print("How many votes are in usable counties?")
print(cast(data, year ~ usable, value = "votes", sum))
usable_counties <- data[c("county", "year", "usable")]

# Figure out the number of years each county is usable   
usable_counties_years=cast(usable_counties,county ~ year, value = "usable")
usable_counties_years$num_usable = usable_counties_years[["2008"]]+usable_counties_years[["2012"]]+usable_counties_years[["2016"]]
usable_counties_num <- usable_counties_years[c("county","num_usable")] 
usable_counties <- merge(usable_counties, usable_counties_num, by = "county")
  

# # Compare % whites in usable and unusable counties
#   data <- cast(reg, county+year ~ race, value = "numreg", sum)
#   TOTAL <- cast(reg, county+year ~., value = "numreg", sum)
#   data <- merge(data, TOTAL, by=c("county", "year"))
#   data$pct_white <- data$W/data$"(all)"
#   pct_white <- data[c("county", "year", "pct_white")]
#   usable_pct_white <- merge(usable_counties,pct_white,by = c("county","year"))
# 
#   years = c("2008", "2012", "2016")
#   for (y in years) {
#     year_data <- subset (usable_pct_white, subset = (year==y))
#     linearMod <- lm(pct_white~usable, data=year_data)
#     print(summary(linearMod))
# }

# counties usable only in 2008 need to be retained for other years 
# so we can figure out their precinct codes to match them to r2008
usable2008 <- cast(usable_counties, county ~ year, value="usable")[c("county", "2008")]
setnames(usable2008, "2008", "usable2008")
usable_counties <- merge(usable_counties, usable2008, by="county")

# filter election data to just the usable (or usable2008) counties
temp <- merge(elections, usable_counties, by = c("county", "year"))
temp <- subset(temp, subset = ((temp$usable) | (temp$usable2008)))
temp <- subset(temp, subset = (precinct_type=="REAL"))
usable_elections <- temp[c("county", "precinct", "year","contest","usable","numvotes","num_usable")]

print("How many votes are in all_year usable counties?")
data = subset(usable_elections, subset = (num_usable==3 & contest == "PRES"))
print(cast(data, year ~ usable, value = "numvotes", sum))

# filter reg data to usable (or usable2008) counties and exclude rows with no precinct specified
temp <- merge(reg, usable_counties, by = c("county", "year"))
usable_reg <- subset(temp, subset = (((temp$usable) | (temp$usable2008)) & precinct!=""))
usable_reg <- usable_reg[c("county", "precinct", "year","race", "numreg", "num_usable")]




# save
save(usable_counties, file = "ProcessedData/usable_counties.Rdata")
save(usable_elections, file = "ProcessedData/usable_elections.Rdata")
save(usable_reg, file = "ProcessedData/usable_reg.Rdata")