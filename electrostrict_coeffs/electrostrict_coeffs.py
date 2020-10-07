import numpy as np 
import pandas as pd 

eps_0 = 1#8.854e-12

data_p = pd.read_csv('Promising19_compounds_epsilion_components_Mh_plus25kb.csv')
data_m = pd.read_csv('Promising19_compounds_epsilion_components_Mh_minus25kb.csv')
deltaP = pd.read_csv('Promising19_compounds_dP.csv') 
dP     = deltaP['deltaP'] 


#chi_plus 
chi_ave_p = []
eps_ave_p = []
for i in range(len(dP)): 
	chi_11p   = 1/data_p['e11'].iloc[i]
	chi_22p   = 1/data_p['e22'].iloc[i] 
	chi_33p   = 1/data_p['e33'].iloc[i]
	chi_ave_p.append((1/3*eps_0) * (chi_11p + chi_22p + chi_33p))
	eps_ave_p.append((1/3*eps_0) * (data_p['e11'].iloc[i] + data_p['e22'].iloc[i] + data_p['e33'].iloc[i])) 

#chi_minus 
chi_ave_m = []
eps_ave_m = []
for i in range(len(dP)): 
	chi_11m   = 1/data_m['e11'].iloc[i]
	chi_22m   = 1/data_m['e22'].iloc[i]
	chi_33m   = 1/data_m['e33'].iloc[i]
	chi_ave_m.append((1/3*eps_0) * (chi_11m + chi_22m + chi_33m))
	eps_ave_m.append((1/3*eps_0) * (data_m['e11'].iloc[i] + data_m['e22'].iloc[i] + data_m['e33'].iloc[i])) 


#Qh & Mh 
Qh=[]
Mh=[]
for i in range(len(dP)): 
	Qh.append(np.abs(chi_ave_p[i] - chi_ave_m[i])/dP[i]) 
	Mh.append(np.abs(eps_ave_p[i] - eps_ave_m[i])/dP[i])
print(Mh)
data_out = pd.DataFrame(columns=['compound','Mh','Qh'])
data_out['compound'] = deltaP['Compounds']
data_out['Mh'] = Mh 
data_out['Qh'] = Qh 
data_out.to_csv('mh_qh_out.csv')
