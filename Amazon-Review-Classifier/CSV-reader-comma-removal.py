import csv
import re

"""
bitchy_line = '3,B000LQOCH0,ABXLMWJIXXAIN,"Natalia Corres ""Natalia Corres""",1,1,4,1219017600,"""Delight"" says it all","This is a confection, that has been, around a few centuries.  It is a light, pillowy citrus gelatin with nuts - in this case Filberts. And it is cut into tiny squares and then liberally coated with powdered sugar.  And it is a tiny mouthful of heaven.  Not too chewy, and very flavorful.  I highly recommend this yummy treat.  If you are familiar with the story of C.S. Lewis ""The Lion, The Witch, and The Wardrobe"" - this is the treat that seduces Edmund into selling out his Brother and Sisters to the Witch.'
fine_line = '1,B001E4KFG0,A3SGXH7AUHU8GW,delmartian,1,1,5,1303862400,”Good Quality Dog Food”,”I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most”.'
lines = [fine_line,bitchy_line] 
"""
pattern = re.compile(r"((?:\"[^\"]*\"),|(?:[^,\"]+))")
	
reader = csv.reader(open("Reviews.csv","r"), dialect = 'excel')
for i,line in enumerate(reader):
#	parsed_csv_line = [element.replace(",", " ") for element in re.findall(pattern, line)] 
	for item in line:
		print (item)
	if i==5: break
