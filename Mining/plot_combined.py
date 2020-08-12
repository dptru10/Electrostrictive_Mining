import pandas as pd
import numpy as np
from pymatgen import MPRester, Structure
from numpy.linalg import eig

mpr = MPRester()
centrosymm=["-1","2/m","mmm","4/m","4/mmm","-3","-3m","6/m","6/mmm","m-3","m-3m"]
non_centro=["1","2","m","222","mm2","4","-4","422","4mm","-42m","3","32","3m","6","-6","622","6mm","-6m2","23","432","-43m"]

#find centro task ids based on pt grps  
mp_ids=[]
i=0
for item in centrosymm: 
	mp_ids.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["task_id"]))
	i+=1
mp_ids = [item for sublist in mp_ids for item in sublist]

clean_tasks=[]
for item in mp_ids:
	obj=dict(item)
	clean_tasks.append(obj['task_id'])
print(len(clean_tasks))

#find centro pretty formulas based on pt grps  
names=[]
i=0
for item in centrosymm:
	names.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["pretty_formula"]))
	i+=1
names = [item for sublist in names for item in sublist]
print(len(names)) 

clean_labels=[]
for item in names:
	obj=dict(item)
	clean_labels.append(obj['pretty_formula'])
print(len(clean_labels))

#find centro elastic compliance tensor based on pt grps  
cen_data=[]
for item in centrosymm: 
	cen_data.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["elasticity.compliance_tensor"]))

i=0
print()
print()
print("centrosymmetric structures:")
for item in cen_data: 
	print("point group %s, len: %i" %(centrosymm[i],len(item)))
	i+=1 
j=0
centro_elastic_compliance=[]
for struct in cen_data: 
	for obj in struct: 
		centro_elastic_compliance.append(obj)
		j+=1
print("Found Elasticity tensors for %i total centrosymmetric structures" %j)

cen_data=[]
for item in centrosymm: 
	cen_data.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["diel.e_total"]))

i=0
print()
print()
print("centrosymmetric structures:")
for item in cen_data: 
	print("point group %s, len: %i" %(centrosymm[i],len(item)))
	i+=1 
j=0
centro_dielectric_tensor=[]
for struct in cen_data: 
	for obj in struct: 
		centro_dielectric_tensor.append(obj)
		j+=1
print("Found dielectric tensors for %i total centrosymmetric structures" %j)

ec_list    =[]
reuss_list =[]
voight_list=[]
vrh_list   =[]
for item in centro_elastic_compliance:
	obj=dict(item)
	if obj['elasticity.compliance_tensor'] != None:
                obj=obj['elasticity.compliance_tensor']
                inv=np.linalg.inv(obj)
                w, v  = eig(obj)
                w     = np.average(np.real(w))
                reuss = ((obj[0][0] + obj[1][1] + obj[2][2]) + 2 * (obj[0][1] + obj[1][2] + obj[2][0]))
                voight= 1/(((inv[0][0] + inv[1][1] + inv[2][2]) + 2 * (inv[0][1] + inv[1][2] + inv[2][0]))/9)
                vrh_av= reuss + voight / 2  
	else: 
                w     = np.nan
                reuss = np.nan
                voight= np.nan
                vrh_av=np.nan
	ec_list.append(w)
	reuss_list.append(reuss)
	voight_list.append(voight)
	vrh_list.append(vrh_av) 

dt_list=[]
for item in centro_dielectric_tensor:
	obj=dict(item)
	if obj['diel.e_total']!= None:
		obj=obj['diel.e_total']
		w, v = eig(obj)
		w    = np.average(np.real(w))
	else: 
		w    = np.nan
	dt_list.append(w)

s_vs_ep=[]
kv_vs_ep=[]
inv_reuss_dt=[]
inv_voight_dt=[]
inv_vrh_dt=[]

for i in range(len(dt_list)):
	s_vs_ep.append((ec_list[i]/dt_list[i])/8.85) 
	inv_reuss_dt.append((reuss_list[i]/dt_list[i])/8.85)
	inv_voight_dt.append((voight_list[i]/dt_list[i])/8.85)
	inv_vrh_dt.append((vrh_list[i]/dt_list[i])/8.85)

data=pd.DataFrame() 
data['1/reuss']  = reuss_list
data['1/voight'] = voight_list
data['1/vrh']    = vrh_list
data['elastic compliance']=ec_list
data['dielectric tensor']=dt_list
data['s/e']     = s_vs_ep 
data['reuss/e'] = inv_reuss_dt
data['voight/e']= inv_voight_dt
data['vrh/e']  = inv_vrh_dt
data['labels'] =clean_labels
data['task_id']=clean_tasks

data=data.loc[data['elastic compliance'] > 0.0]
data=data.loc[data['dielectric tensor'] > 0.0]
data=data.loc[data['1/reuss'] > 0.0]
data=data.loc[data['labels']!='CO2']
data.to_csv('centrosymmetric_data.csv',mode='w')
