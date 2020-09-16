import numpy as np 
import pandas as pd 
import argparse


eig_list=[]
for i in range(len(data)): 
	matrix = [[data['e11'].iloc[i],data['e12'].iloc[i],data['e13'].iloc[i]],
	[data['e21'].iloc[i],data['e22'].iloc[i],data['e23'].iloc[i]],
	[data['e31'].iloc[i],data['e32'].iloc[i],data['e33'].iloc[i]]]	    
	eigs,vecs = np.linalg.eig(matrix)
	mean = np.mean(eigs)
	eig_list.append(mean)

