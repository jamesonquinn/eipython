rm(list=ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dirname(rstudioapi::getActiveDocumentContext()$path)

library(data.table); library(plyr)

load("ProcessedData/usable_elections.Rdata")
load("ProcessedData/usable_reg.Rdata")
load("ProcessedData/usable_counties.Rdata")

# Make everything into data frames!
usable_elections <- setDT(usable_elections)
usable_reg <- setDT(usable_reg)
usable_counties <- setDT(usable_counties)


years = c("2008","2012","2016")

#reg_precincts_by_year <- arrange(cast(usable_reg,county+precinct~year, length, value="numreg"), county, precinct)
#election_precincts_by_year <- arrange(cast(usable_elections,county+precinct~year, length, value="numvotes"), county, precinct)

e = list()
r = list()
for (y in years){
  ey <- paste("e",y, sep="")
  uy <- paste("usable",y, sep="")
  pey <- paste("p_e",y, sep="")
  pry <- paste("p_r",y, sep="")
  data <- usable_elections[year==y & contest == "PRES",]
  e[[y]] <- setDT(dcast(data, county+precinct+usable~., fun.agg = sum,value.var="numvotes"))
  setnames(e[[y]],c("precinct",".", "usable"),c(pey,ey,uy))
  
  ry <- paste("r",y, sep="")
  data <- usable_reg[year==y,]
  r[[y]] <- setDT(dcast(data, county+precinct~., fun.agg = sum,value.var="numreg"))
  setnames(r[[y]],c("precinct","."), c(pry,ry))
}

#######################################################################3

# CLEAN UP p2008
# restrict to usable in 2008
p2008 <- e[["2008"]][usable2008==TRUE]

#e2008 has different formats for the variable "old_precinct", depending on county
# BURKE and UNION counties are special, so we separate them out
BURKE = p2008[county=="BURKE",]
UNION = p2008[county=="UNION",]
p2008 <- p2008[county !="BURKE" & county !="UNION",]

# The variable p_e2008 is different in different counties
# In most counties, the variable  is actually precinct_name
p2008[, precinct_name := p_e2008]

# EXCEPTIONS:
# (CABARRUS), COLUMBUS, (GUILFORD), (NEW HANOVER), SWAIN: precinct actually is the precinct code 
# PERQUIMANS: precinct code is first 6 characters of precinct_name, and precinct_name appears nowhere else
p2008[county=="PERQUIMANS", precinct:=substring(precinct_name,1,6)]
p2008[county %in% c("COLUMBUS","SWAIN"), precinct:=precinct_name]
p2008[county %in% c("COLUMBUS","SWAIN","PERQUIMANS"), precinct_name:=NA]

# (MECKLENBURG): uses "PCT <precinct_code>" in e2008 and just precinct_code in e2012 (but in e2008, without leading digits)
# also for some reason in all but e2008, .1 is added to precincts 78,107,139,204,223,238. Go figure!

# BURKE, (FORSYTH), UNION: in e2008, uses "<precinct_code> - <precinct_name>" instead of precinct_name (sep " - ")
BURKE[, c("precinct", "precinct_name") := tstrsplit(p_e2008, " - ")]
UNION[, c("precinct", "precinct_name") := tstrsplit(p_e2008, " - ")]


p2008 <- rbindlist(list(p2008, BURKE, UNION),use.names = TRUE)


######################################################################

# CLEAN UP e2012
p2012 <- e[["2012"]]
p2012[, c("precinct", "precinct_name") := tstrsplit(p_e2012, "_")]

# MERGE e2008 and e2012
e08e12 <- merge(p2008,p2012, by=c("county","precinct_name"), all=TRUE)
# but deal with COLUMBUS, PERQUIMANS, SWAIN, separately
sep_counties = c("COLUMBUS","PERQUIMANS","SWAIN")
e08e12 <- e08e12[!(county %in% sep_counties)]
e08e12 <- e08e12[,.(county,precinct2008 = precinct.x,precinct2012 = precinct.y,precinct_name, usable2008, usable2012, p_e2008, p_e2012)]



# Manually put in false in non-usable-2008 where appropriate
#usable_by_year <- cast(usable_counties,county ~ year, value = "usable")
#usable_2012_not_2008 <- subset(usable_by_year, subset = (usable_by_year["2008"]==FALSE & usable_by_year["2012"]==TRUE ))
e08e12[county %in% c("ALEXANDER", "CLEVELAND", "FORSYTH","GRANVILLE","IREDELL", "NEW HANOVER","ROWAN", "SCOTLAND", "VANCE"), usable2008:=FALSE] 

# Deal with COLUMBUS and PREQUIMANS and SWAIN, which get merged by precinct, not precinct name
EXTRA08 <- p2008[county %in% sep_counties]
EXTRA12 <- p2012[county %in% sep_counties]
EXTRA_merged <- merge(EXTRA08,EXTRA12,by = c("county", "precinct"),all=TRUE)
EXTRA_merged <- EXTRA_merged[,.(county, precinct2008 = precinct, precinct2012=precinct, precinct_name = precinct_name.y, usable2008,usable2012,p_e2008,p_e2012)]
e08e12 <- rbind(e08e12, EXTRA_merged)

# p_e2008 for UNION county contains both the precinct codes and the precinct names. We merged on the names (because some names have changed codes), 
# but the opposite is also true: some codes have changed names, and we want to match those too.
# so we drop all the 2008 UNION observations that did not match on name (i.e. precinct2012 = NA), and then remerge them based on precinct
UNIONunmerged <- e08e12[county=="UNION" & is.na(precinct2012), .(county,p_e2008_temp = p_e2008, precinct2012 = precinct2008)]
e08e12 <- e08e12[county!="UNION" | !is.na(precinct2012)]
e08e12 <- merge(e08e12,UNIONunmerged,by = c("county","precinct2012"),all=TRUE)
e08e12[county=="UNION" & !is.na(p_e2008_temp), `:=`(precinct2008 = precinct2012,usable2008=TRUE,p_e2008=p_e2008_temp)]
e08e12[,p_e2008_temp:=NULL]

# Copy precinct2012 to precinct 2008 where the latter does not already exist but should
e08e12[is.na(precinct2008) & usable2008, precinct2008 := precinct2012]

# For 2008 precincts that don't appear in 2012, get precinct codes from r2008
e08e12[county=="ALAMANCE" & precinct_name=="WEST BOONE SAT", `:=`(precinct2008 = "03WS", usable2008 = TRUE)]

e08e12[county=="CHEROKEE" & precinct_name=="BEAVERDAM Y", `:=`(precinct2008 = "BEAVY",usable2008=TRUE)]
e08e12[county=="CHEROKEE" & precinct_name=="BEAVERDAM Z", `:=`(precinct2008 = "BEAVZ",usable2008=TRUE)]
e08e12[county=="CHEROKEE" & precinct_name=="NOTLA Y", `:=`(precinct2008 = "NOTLY",usable2008=TRUE)]
e08e12[county=="CHEROKEE" & precinct_name=="NOTLA Z", `:=`(precinct2008 = "NOTLZ",usable2008=TRUE)]

e08e12[county=="ROCKINGHAM" & precinct_name=="AYERS VALLEY", `:=`(precinct2008 = "AV",usable2008=TRUE)]
e08e12[county=="ROCKINGHAM" & precinct_name=="COURTLAND", `:=`(precinct2008 = "CO",usable2008=TRUE)]
e08e12[county=="ROCKINGHAM" & precinct_name=="MADISON", `:=`(precinct2008 = "MD",usable2008=TRUE)]
e08e12[county=="ROCKINGHAM" & precinct_name=="VANCE", `:=`(precinct2008 = "VA",usable2008=TRUE)]

e08e12[county=="TRANSYLVANIA" & precinct_name=="BALSAM GROVE", `:=`(precinct2008 = "BG",usable2008=TRUE)]
e08e12[county=="TRANSYLVANIA" & precinct_name=="GLOUCESTER", `:=`(precinct2008 = "GL",usable2008=TRUE)]
e08e12[county=="TRANSYLVANIA" & precinct_name=="LAKE TOXAWAY", `:=`(precinct2008 = "LT",usable2008=TRUE)]
e08e12[county=="TRANSYLVANIA" & precinct_name=="QUEBEC", `:=`(precinct2008 = "QB",usable2008=TRUE)]
e08e12[county=="TRANSYLVANIA" & precinct_name=="SAPPHIRE WHITEWATER", `:=`(precinct2008 = "SW",usable2008=TRUE)]

#merge in r2008
r2008 <- setDT(r[["2008"]])
r2008[, precinct2008 := p_r2008]
e08r08e12<-merge(e08e12,r2008,by = c("county","precinct2008"), all=TRUE)
e08r08e12 <- e08r08e12[,r2008:=NULL]

#merge in r2012
r2012 <- setDT(r[["2012"]])
r2012[,precinct2012:=p_r2012]
e08r08e12r12<-merge(e08r08e12,r2012,by = c("county","precinct2012"), all=TRUE)
e08r08e12r12 <- e08r08e12r12[,r2012:=NULL]

# some checks:
# In each year, e and r data exists for the same precincts
stopifnot(nrow(e08r08e12r12[!is.na(p_e2008) & is.na(p_r2008)])==0)
stopifnot(nrow(e08r08e12r12[is.na(p_e2008) & !is.na(p_r2008)])==0)
stopifnot(nrow(e08r08e12r12[!is.na(p_e2012) & is.na(p_r2012)])==0)
stopifnot(nrow(e08r08e12r12[is.na(p_e2012) & !is.na(p_r2012)])==0)
# In each year, our variable precinct(year) is the same as p_r(year), but not p_e(year)
stopifnot(nrow(e08r08e12r12[!is.na(precinct2008) & is.na(p_r2008)])==0)
stopifnot(nrow(e08r08e12r12[is.na(precinct2008) & !is.na(p_r2008)])==0)
stopifnot(nrow(e08r08e12r12[precinct2008 != p_r2008])==0)
stopifnot(nrow(e08r08e12r12[!is.na(precinct2012) & is.na(p_r2012)])==0)
stopifnot(nrow(e08r08e12r12[is.na(precinct2012) & !is.na(p_r2012)])==0)
stopifnot(nrow(e08r08e12r12[precinct2012 != p_r2012])==0)

print("Which precincts have different codes between 2008 and 2012?")
print(e08r08e12r12[!is.na(precinct2012) & !is.na(precinct2008) & precinct2008 != precinct2012])
print("That's OK, their precinct name is the same, so it's all good")

# We now make our official "precinct" variable, which rules in favor of precinct2012 in case of conflict
# This is the variable on which we'll merge with 2016
e08r08e12r12[,precinct:=precinct2012]
e08r08e12r12[is.na(precinct), precinct := precinct2008]
stopifnot(!is.na(e08r08e12r12$precinct))
#e08r08e12r12[,`:=`(precinct2008=NULL, precinct2012=NULL)]
###########################################################################################

#Let's merge e2016 and r2016
r2016 <- setDT(r[["2016"]])
r2016[, precinct := p_r2016]

e2016 <- setDT(e[["2016"]])
e2016[, precinct := p_e2016]

e16r16 <- merge(e2016,r2016,by = c("county","precinct"), all=TRUE)[usable2016==TRUE]
stopifnot(!is.na(e16r16$p_e2016))
stopifnot(!is.na(e16r16$p_r2016))


# Before merging with 2016 data, we merge some 2012 precincts together, because this is how they occur in 2016
e08r08e12r12[county == "ALEXANDER" & precinct=="G1", precinct := "G1G2"]
e08r08e12r12[county == "ALEXANDER" & precinct=="G2", precinct := "G1G2"]
e08r08e12r12[county == "ALEXANDER" & precinct=="S1", precinct := "S1S2"]
e08r08e12r12[county == "ALEXANDER" & precinct=="S2", precinct := "S1S2"]
e08r08e12r12[county == "ALEXANDER" & precinct=="T1", precinct := "T1T4T5"]
e08r08e12r12[county == "ALEXANDER" & precinct=="T2", precinct := "T2T3"]
e08r08e12r12[county == "ALEXANDER" & precinct=="T3", precinct := "T2T3"]
e08r08e12r12[county == "ALEXANDER" & precinct=="T4", precinct := "T1T4T5"]
e08r08e12r12[county == "ALEXANDER" & precinct=="T5", precinct := "T1T4T5"]
e08r08e12r12[county == "ALEXANDER" & precinct=="L", precinct := "LRSL"]
e08r08e12r12[county == "ALEXANDER" & precinct=="SL", precinct := "LRSL"]
e08r08e12r12[county == "HOKE" & precinct=="06A", precinct := "06B"]

standardized_precincts <- merge(e08r08e12r12,e16r16, by=c("county","precinct"),all=TRUE)[,.(county,precinct,p_e2008,p_r2008,p_e2012,p_r2012,p_e2016,p_r2016)]
stopifnot(!is.na(standardized_precincts$precinct))

# precinct usability: make sure that we do not have precincts for non-usable counties 
usable_counties_by_year <- cast(usable_counties, county ~ year, value="usable")
setnames(usable_counties_by_year,"2008","usable2008")
setnames(usable_counties_by_year,"2012","usable2012")
setnames(usable_counties_by_year,"2016","usable2016")

standardized_precincts <- merge(standardized_precincts,usable_counties_by_year,by="county",all=TRUE)
stopifnot(nrow(standardized_precincts[!usable2008 & (!is.na(p_e2008) | !is.na(p_r2008))])==0)
stopifnot(nrow(standardized_precincts[!usable2016 & (!is.na(p_e2016) | !is.na(p_r2016))])==0)
standardized_precincts[usable2012==FALSE, `:=`(p_e2012 = NA, p_r2012 = NA)]

# conversely, precincts that we don't have for a given election are not usable
standardized_precincts[is.na(p_e2008), usable2008:= FALSE]
standardized_precincts[is.na(p_e2012), usable2012:= FALSE]
standardized_precincts[is.na(p_e2016), usable2016:= FALSE]

standardized_precincts[,num_usable_years:=usable2008+usable2012+usable2016]
stopifnot(!is.na(standardized_precincts$num_usable_years))

save(standardized_precincts, file = "ProcessedData/standardized_precincts.Rdata")
rm(list=ls())
load("ProcessedData/standardized_precincts.Rdata")