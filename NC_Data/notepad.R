








###########################################################################################

# RECONCILE 2012 WITH 2016


merged <- merge(e[["2012"]],e[["2016"]], by=c("county","precinct"), all=TRUE)[c("county", "precinct","usable2012","precinct_name","usable2016")]

# EXPORT FOR CLEANUP
#merged_bad <- subset(merged,subset=(is.na(merged$usable2012) | is.na(merged$usable2016)))
#write.csv(merged_bad, file = "Reconcile_2012_2016.csv")

# CLEANUP
setnames(merged,"precinct","precinct2012")
merged$precinct2016<-merged$precinct2012
merged$precinct2012[is.na(merged$usable2012)]=NA
merged$precinct2016[is.na(merged$usable2016)]=NA




# Not sure ALLEGHANY match is right
merged$precinct2016[merged$county == "ALLEGHANY" & merged$precinct2012=="02"] = "03A"
merged$precinct2016[merged$county == "ALLEGHANY" & merged$precinct2012=="03"] = "03A"
merged$precinct2016[merged$county == "ALLEGHANY" & merged$precinct2012=="05"] = "06A"
merged$precinct2016[merged$county == "ALLEGHANY" & merged$precinct2012=="06"] = "06A"
merged$precinct2016[merged$county == "ALLEGHANY" & merged$precinct2012=="07"] = "06A"
merged$usable2016[merged$county == "ALLEGHANY" & !is.na(merged$precinct2012)] = TRUE
merged <- subset(merged,subset=(merged$county!="ALLEGHANY" | (!is.na(merged$usable2012) & !is.na(merged$usable2016))))

#merged$precinct2016[merged$county == "BRUNSWICK" & merged$precinct2012=="04C"] = "04C1"
#merged$usable2016[merged$county == "BRUNSWICK" & !is.na(merged$precinct2012)] = TRUE
#merged <- subset(merged,subset=(merged$county!="BRUNSWICK" | (!is.na(merged$usable2012) & !is.na(merged$usable2016))))

#merged$precinct2016[merged$county == "CASWELL" & merged$precinct2012=="HIGH"] = "PH"
#merged$precinct2016[merged$county == "CASWELL" & merged$precinct2012=="PROS"] = "PH"
#merged$usable2016[merged$county == "CASWELL" & !is.na(merged$precinct2012)] = TRUE
#merged <- subset(merged,subset=(merged$county!="CASWELL" | (!is.na(merged$usable2012) & !is.na(merged$usable2016))))

# HOKE precinct 15 is missing in 2012; deleting for now
merged$precinct2016[merged$county == "HOKE" & merged$precinct2012=="06A"] = "06B"
merged$usable2016[merged$county == "HOKE" & !is.na(merged$precinct2012)] = TRUE
merged <- subset(merged,subset=(merged$county!="HOKE" | (!is.na(merged$usable2012) & !is.na(merged$usable2016))))

#merged$precinct2016[merged$county == "JACKSON" & merged$precinct2012=="DIL"] = "SND"
#merged$precinct2016[merged$county == "JACKSON" & merged$precinct2012=="SNW"] = "SND"
#merged$usable2016[merged$county == "JACKSON" & !is.na(merged$precinct2012)] = TRUE
#merged <- subset(merged,subset=(merged$county!="JACKSON" | (!is.na(merged$usable2012) & !is.na(merged$usable2016))))

# CLEVELAND and NEW HANOVER and PERSON are unmatchable

# Delete 2016 precincts that don't have a 2012 version and where usable2016 is FALSE
merged <- subset(merged,subset=(!is.na(merged$precinct2012) | merged$usable2016))







