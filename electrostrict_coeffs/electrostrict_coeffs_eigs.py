import numpy as np 
import pandas as pd 

eps_0  = 8.854e-12

data_p = pd.read_csv('Promising19_compounds_epsilion_components_Mh_plus25kb.csv')
data_m = pd.read_csv('Promising19_compounds_epsilion_components_Mh_minus25kb.csv')
deltaP = pd.read_csv('Promising19_compounds_dP.csv') 
dP     = deltaP['deltaP'] 

#eig_p
eig_p=[]
for i in range(len(data_p)): 
	matrix = [[data_p['e11'].iloc[i],data_p['e12'].iloc[i],data_p['e13'].iloc[i]],
	[data_p['e21'].iloc[i],data_p['e22'].iloc[i],data_p['e23'].iloc[i]],
	[data_p['e31'].iloc[i],data_p['e32'].iloc[i],data_p['e33'].iloc[i]]]	    
	eigs,vecs = np.linalg.eig(matrix)
	eig_p.append(np.mean(eigs))

#eig_m 
eig_m=[]
for i in range(len(data_m)): 
	matrix = [[data_m['e11'].iloc[i],data_m['e12'].iloc[i],data_m['e13'].iloc[i]],
	[data_m['e21'].iloc[i],data_m['e22'].iloc[i],data_m['e23'].iloc[i]],
	[data_m['e31'].iloc[i],data_m['e32'].iloc[i],data_m['e33'].iloc[i]]]	    
	eigs,vecs = np.linalg.eig(matrix)
	eig_m.append(np.mean(eigs))

#chi_plus 
chi_ave_p = []
eps_ave_p = []
for i in range(len(dP)): 
	chi   = 1/eig_p[i]
	chi_ave_p.append(eps_0 * (chi))
	eps_ave_p.append(eps_0 * (eig_p[i]))#data_p['e11'].iloc[i] + data_p['e22'].iloc[i] + data_p['e33'].iloc[i])) 

#chi_minus 
chi_ave_m = []
eps_ave_m = []
for i in range(len(dP)): 
	chi   = 1/eig_m[i]
	chi_ave_m.append(eps_0 * (chi))
	eps_ave_m.append(eps_0 * (eig_m[i]))#data_m['e11'].iloc[i] + data_m['e22'].iloc[i] + data_m['e33'].iloc[i])) 

#Qh & Mh 
Qh=[]
Mh=[]
for i in range(len(dP)): 
	Qh.append(np.abs(chi_ave_p[i] - chi_ave_m[i])/dP[i]) 
	Mh.append(np.abs(eps_ave_p[i] - eps_ave_m[i])/dP[i])

data_out = pd.DataFrame(columns=['compound','Mh','Qh'])
data_out['compound'] = deltaP['Compounds']
data_out['Mh'] = Mh 
data_out['Qh'] = Qh 
data_out.to_csv('mh_qh_out_eig.csv')
