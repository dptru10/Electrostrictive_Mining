import numpy as np 
import pandas as pd 

eps_0 = 8.854e-12
dP    = 0.50 

data_p = pd.read_csv('Promising19_compounds_epsilion_components_Mh_plus25kb.csv')
data_m = pd.read_csv('Promising19_compounds_epsilion_components_Mh_minus25kb.csv')

for i in range(len(data)): 
	#chi_plus 
	chi_11p   = 1/data_p['e11']
	chi_22p   = 1/data_p['e22']
	chi_33p   = 1/data_p['e33']
	chi_ave_p = (1/3*eps_0) * (chi_11p + chi_22p + chi_33p)
	eps_ave_p = (1/3*eps_0) * (data_p['e11'] + data_p['e22'] + data_p['e33'])  
	#chi_minus 
	chi_11m   = 1/data_m['e11']
	chi_22m   = 1/data_m['e22']
	chi_33m   = 1/data_m['e33']
	chi_ave_m = (1/3*eps_0) * (chi_11m + chi_22m + chi_33m)
	eps_ave_m = (1/3*eps_0) * (data_m['e11'] + data_m['e22'] + data_m['e33'])  
	#
	Qh = np.abs(chi_ave_p - chi_ave_m)/dP  
	Mh = np.abs(eps_ave_p - eps_ave_m)/dP
 

