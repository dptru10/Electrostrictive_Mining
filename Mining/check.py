import pandas as pd 

data = pd.read_csv('compare_electrostrict.csv')
lists=['vrh','voight','reuss']

new=[]
for item in lists: 
	for i in range(len(data)):
		if data[item].iloc[i] in list(data['original_list']):
			pass
		else: 
			new.append(data[item].iloc[i])
new = list(set(new))
for item in new: 
	print(item)
