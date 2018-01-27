#Reviews.csv filtering script

import csv
import re


csv_in = csv.reader(open("Reviews.csv", "r"))
csv_good = csv.writer(open("Good_Samples.csv", "w"))
csv_bad = csv.writer(open("Bad_Samples.csv", "w"))


for line in csv_in:
	#print(line)
	if len(line) != 10:
		csv_bad.writerow(line)
	else:
		csv_good.writerow(line)
	




"""
break the dataset into 2 parts:
-one which has the format that i am expecting
-one that does not have the format
then check to see if the latter has any similarities

"""
