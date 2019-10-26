Data files from https://dl.ncsbe.gov/?prefix=ENRS/

Metadata in layout_results_pct.txt and layout_voter_stats.txt
(although variable names in layout_results are not quite right.)
Also, what the Metadata for the election results don't tell you is that the values of the "contest" variable change: e.g. sometimes it's "GOVERNOR" and sometimes it's "NC GOVERNOR". Ugh.

Modifying the raw files:
   * In all of the files, had to replace precinct "SH#1" in Greene County with "SH1", since the # character was screwing up R. In 2008-2012, also removed other # characters. 
   * In 2014 election file also had to be redacted to get rid of weird characters in candidate names that were screwing everything up; deleted candidate name field. 


Info on straight party voting: 
          https://en.wikipedia.org/wiki/Straight-ticket_voting#North_Carolina
However, it appears that all "straight party" votes have already been added to the totals for non-presidential races in these elections. 
(We can see this by comparing these results to those in Ballotopedia; also, if it were not the case, the turnout for Governor and Senator would be 50% larger than for President, which can't be right.)