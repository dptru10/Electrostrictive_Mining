import numpy as np 
import pandas as pd 
import argparse

data = pd.read_csv('Promising19_compounds_epsilion_components_Mh.csv')

eig_list=[]
for i in range(len(data)): 
	matrix = [[data['e11'].iloc[i],data['e12'].iloc[i],data['e13'].iloc[i]],
	[data['e21'].iloc[i],data['e22'].iloc[i],data['e23'].iloc[i]],
	[data['e31'].iloc[i],data['e32'].iloc[i],data['e33'].iloc[i]]]	    
	eigs,vecs = np.linalg.eig(matrix)
	mean = np.mean(eigs)
	eig_list.append(mean)
data_out = pd.DataFrame(columns=['compound','ave_epsilon_eig']) 
data_out['compound']        = data['S.N']
data_out['ave_epsilon_eig'] = eig_list
data_out.to_csv('Promising19_compounds_epsilion_eigs.csv') 
