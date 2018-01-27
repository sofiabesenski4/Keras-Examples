#Regular expression testing to remove commas in double quotes

import re 
test_text =str(3018)+ ","+ str("B0025UALB6")+","+ str("AJD41FBJD9010")+","+ "\"N. Ferguson ""Two, Daisy, Hannah, and Kitten""\""+","+ str(0)+","+str(0)+","+str(5)+","+str(1313107200)+","+"\"convenient, filling, 70 calories and lots o protein = great work lunch item\""+"," +"\"I keep a stack of these tuna cups at work, along with lemon juice and teriyaki sauce for variety.  They are low calorie, satisfying, and nutritious.  I'm sold on them-- now buy in bulk and keep on hand all the time!\""
print(test_text)
pattern = re.compile(r'\".*?(,)?\"')
print(re.findall(pattern, test_text))		
	
