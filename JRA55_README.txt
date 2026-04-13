JRA-25 transition to JRA-55:
1st October 2013 JRA-55 monthly latitude/longitude gridded data (1.25deg, 37 pressure levels)
Beginning of November 2013 Same as above but for daily data (year 1996 to 2012)
Beginning of December 2013 Same as above but for daily data (year 1979 to 1995)
Beginning of January 2014 Same as above but for daily data (year 1958 to 1978)
Beginning of February 2014 Starting the distribution of near real-time version of the JRA-55
Early February 2014 Ceasing the distribution of the JRA-25/JCDAS
Beginning of March 2014 JRA-55 model grid data (TL319L60)



JRA-55
http://jra.kishou.go.jp/JRA-55/index_en.html
or
http://rda.ucar.edu/datasets/ds628.0/
http://rda.ucar.edu/datasets/ds628.1/


TIME:
1958-01-01 00:00 +0000 to 2013-01-01 00:00 +0000 (JRA-55 3-Hourly Model Resolution 2-Dimensional Average Diagnostic Fields)
1958-01-01 00:00 +0000 to 2012-12-31 21:00 +0000 (JRA-55 3-Hourly Model Resolution 2-Dimensional Instantaneous Diagnostic Fields)


GRID:
http://rda.ucar.edu/datasets/ds628.0/docs/JRA-55.TL319L60_glw.txt
http://www.ecmwf.int/publications/manuals/libraries/interpolation/n160FIS.html
0.562° x ~0.562° from 0E to 359.438E and 89.57N to 89.57S (640 x 320 Gaussian Longitude/Latitude) (JRA-55 3-Hourly Model Resolution 2-Dimensional Average Diagnostic Fields)
0.562° x ~0.562° from 0E to 359.438E and 89.57N to 89.57S (640 x 320 Gaussian Longitude/Latitude) (JRA-55 3-Hourly Model Resolution 2-Dimensional Instantaneous Diagnostic Fields)

FIELDS:
61	TPRAT	Total precipitation	mm day-1
(62	LPRAT	Large scale precipitation	mm day-1
 63	CPRAT	Convective precipitation	mm day-1)
204	DSWRF	Downward solar radiation flux	W m-2
205	DLWRF	Downward longwave radiation flux	W m-2
11	TMP	Temperature	K
51	SPF H	Specific humidity	kg kg-1
33	U GRD	u-component of wind	m s-1
34	V GRD	v-component of wind	m s-1

Questions:
1.	TPRAT = LPRAT+CPRAT ?
2.	similar issue as JRA-25? ie,
	"the correct valid time appears to be 3 hours EARLIER than the encod
	ed reference time for variables in which 6-hour averaging was applied."
