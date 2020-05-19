import numpy as np
import pandas as pd 
from numpy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument("--plot_kv",   action='store_true')
parser.add_argument("--plot_eigs", action='store_true')
args  = parser.parse_args() 

centro_elastic_compliance=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/elasticity_compliance/centro_elasticity.npy',allow_pickle=True)
centro_dielectric_tensor=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/dielectric_total/centro_diel.npy',allow_pickle=True)
labels=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/data/centro_names.npy',allow_pickle=True)
mp_ids=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/data/centrosymmetric_task_ids.npy',allow_pickle=True)

clean_tasks=[]
for item in mp_ids:
	obj=dict(item)
	clean_tasks.append(obj['task_id'])
print(len(clean_tasks))

clean_labels=[]
for item in labels:
	obj=dict(item)
	clean_labels.append(obj['pretty_formula'])
print(len(clean_labels))

ec_list=[]
Kv     =[]
for item in centro_elastic_compliance:
	obj=dict(item)
	if obj['elasticity.compliance_tensor'] != None:
                obj=obj['elasticity.compliance_tensor']
                w, v = eig(obj)
                w    = np.average(np.real(w))
                k    = ((obj[0][0] + obj[1][1] + obj[2][2]) + 2 * (obj[0][1] + obj[1][2] + obj[2][0]))/9
	else: 
		w    = np.nan
		k    = np.nan 
	ec_list.append(w)
	Kv.append(k)

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
for i in range(len(dt_list)):
	s_vs_ep.append((ec_list[i]/dt_list[i])/8.85) 
	kv_vs_ep.append((Kv[i]/dt_list[i])/8.85)

data=pd.DataFrame() 
data['Bulk modulus'] = Kv     
data['elastic compliance']=ec_list
data['dielectric tensor']=dt_list
data['s/e']    = s_vs_ep 
data['kv/e']   = kv_vs_ep 
data['labels'] =clean_labels
data['task_id']=clean_tasks

data=data.loc[data['elastic compliance'] > 0.0]
data=data.loc[data['dielectric tensor'] > 0.0]
data=data.loc[data['Bulk modulus'] > 0.0]
data=data.loc[data['labels']!='CO2']



fig=plt.figure()
ax=fig.add_subplot(111)
if args.plot_eigs is True: 
	high_ratio = data.loc[data['s/e'] > 1]
	print("len high ratio: %i" %len(high_ratio))
	low_ratio  = data.loc[data['s/e'] < 1]
	print("len low ratio: %i" %len(low_ratio))
	plt.title('distribution of $s$ vs $\epsilon$ eigs.')
	plt.scatter(high_ratio['dielectric tensor'],high_ratio['elastic compliance'],marker='o',color='red')
	plt.scatter(low_ratio['dielectric tensor'],low_ratio['elastic compliance'],marker='o',color='blue')
	i=0
	#for pts in zip(high_ratio['dielectric tensor'],high_ratio['elastic compliance']):
	#	ax.annotate(str(high_ratio['labels'].iloc[i]),pts)
	#	i+=1


	plt.ylabel("elastic compliance average eigs. [$10^{-12}$ $pa^{-1}$]")
	plt.yscale('log')
	name = 'elastic_eigs'
if args.plot_kv is True:
	high_ratio = data.loc[data['kv/e'] > 1]
	print("len high ratio: %i" %len(high_ratio))
	low_ratio  = data.loc[data['kv/e'] < 1]
	print("len low ratio: %i" %len(low_ratio))
	plt.title('distribution of $k_{v}$ vs $\epsilon$ eigs.')
	plt.scatter(high_ratio['dielectric tensor'],high_ratio['Bulk modulus'],marker='o',color='red')
	plt.scatter(low_ratio['dielectric tensor'],low_ratio['Bulk modulus'],marker='o',color='blue')
	i=0
	#for pts in zip(high_ratio['dielectric tensor'],high_ratio['Bulk modulus']):
	#	ax.annotate(str(high_ratio['labels'].iloc[i]),pts)
	#	i+=1
	plt.ylim(1,350)#np.max(data['Bulk modulus']))
	plt.xlim(1,800)#np.max(data['Bulk modulus']))
	#plt.axis([1,400,1,100])
	plt.ylabel("Bulk modulus Voight ave. ($k_{v}$) [$10^{-12}$ $pa^{-1}$]")
	name = 'kv'
plt.xlabel("dielect. tensor average eigs.")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('%s_vs_diel.png' %name)
plt.show()

fig=plt.figure()
ax=fig.add_subplot(111)
if args.plot_eigs is True: 
	high_ratio = high_ratio.sort_values(by='s/e',ascending=False)
	top_ten    = high_ratio.head(n=10)
	plt.title('distribution of $s$ vs $\epsilon$ eigs.')
	plt.scatter(high_ratio['dielectric tensor'],high_ratio['elastic compliance'],marker='o',color='red')
	i=0
	for pts in zip(top_ten['dielectric tensor'],top_ten['elastic compliance']):
		ax.annotate(str(top_ten['labels'].iloc[i]),pts)
		i+=1
	plt.ylabel("Elastic compliance average eigs. [$10^{-12}$ $pa^{-1}$]")
	plt.yscale('log')
	name = 'elastic_eigs'
if args.plot_kv is True:
	high_ratio = high_ratio.sort_values(by='kv/e',ascending=False)
	top_ten    = high_ratio.head(n=10)
	plt.title('Distribution of $K_{v}$ vs $\epsilon$ eigs.')
	plt.scatter(high_ratio['dielectric tensor'],high_ratio['Bulk modulus'],marker='o',color='red')
	i=0
	for pts in zip(top_ten['dielectric tensor'],top_ten['Bulk modulus']):
		ax.annotate(str(top_ten['labels'].iloc[i]),pts)
		i+=1
	plt.ylabel("Bulk modulus Voight ave. ($K_{v}$) [$10^{-12}$ $pa^{-1}$]")
	name = 'kv'
plt.xlabel("dielect. tensor average eigs.")
plt.yscale('log')
plt.tight_layout()
plt.savefig('%s_vs_diel_high_ratio.png' %name)
plt.show()

#plot figures
plt.figure()
if args.plot_eigs is True: 
	plt.title('Distribution of $S$ vs $\epsilon$ Eigs.')
	plt.hist2d(x=data['dielectric tensor'],y=data['elastic compliance'],bins=np.int(np.sqrt(len(data))),norm=colors.LogNorm())   
	plt.ylabel("Elastic Compliance Average Eignevalues [$10^{-12}$ $Pa^{-1}$]")
	plt.axis([np.min(data['dielectric tensor']),np.max(data['dielectric tensor']),np.min(data['elastic compliance']),np.max(data['elastic compliance'])])
if args.plot_kv is True: 
	plt.hist2d(x=data['dielectric tensor'],y=data['Bulk modulus'],bins=np.int(np.sqrt(len(data))),norm=colors.LogNorm())   
	plt.ylabel("Bulk modulus Voight ave. [$10^{-12}$ $Pa^{-1}$]")
	plt.axis([np.min(data['dielectric tensor']),np.max(data['dielectric tensor']),np.min(data['Bulk modulus']),np.max(data['Bulk modulus'])])
plt.colorbar() 
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dielectric Tensor Average Eigenvalues")
plt.tight_layout() 
plt.savefig('%s_vs_epsilon_dist.png' %name)

select=pd.DataFrame(columns=data.columns)
select_materials=['MgTiO3','HfO2','MgO','SrTiO3'] #'Al2O3','TiO2','SiO2', 
for item in select_materials: 
	select=select.append(data.loc[data['labels']==item])
select.to_csv('select.csv')

#plot figures
plt.figure()
fig=plt.figure()
ax=fig.add_subplot(111)
if args.plot_eigs is True: 
	plt.title('Distribution of $S$ vs $\epsilon$ Eigs.')
	i=0
	for pts in zip(select['dielectric tensor'],select['elastic compliance']):
		ax.annotate(str(select['labels'].iloc[i]),pts)
		i+=1
	plt.scatter(select['dielectric tensor'],select['elastic compliance'],marker='*',color='red')
	plt.hist2d(x=data['dielectric tensor'],y=data['elastic compliance'],bins=100,norm=colors.LogNorm())   
	plt.ylabel("Elastic Compliance Average Eignevalues [$10^{-12}$ $Pa^{-1}$]")
	plt.axis([4,40,5,10])
if args.plot_kv is True:
	plt.title('Distribution of $K_{v}$ vs $\epsilon$ Eigs.')
	i=0
	for pts in zip(select['dielectric tensor'],select['Bulk modulus']):
		ax.annotate(str(select['labels'].iloc[i]),pts)
		i+=1
	plt.scatter(select['dielectric tensor'],select['Bulk modulus'],marker='*',color='red')
	plt.hist2d(x=data['dielectric tensor'],y=data['Bulk modulus'],bins=100,norm=colors.LogNorm())   
	plt.ylabel("Bulk modulus Voight [$10^{-12}$ $Pa^{-1}$]")
	plt.axis([np.min(select['dielectric tensor']),np.max(select['dielectric tensor']),np.min(select['Bulk modulus']),np.max(select['Bulk modulus'])])
plt.colorbar() 

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dielectric Tensor Average Eigenvalues")
#plt.tight_layout()
plt.savefig('%s_vs_epsilon_dist_reduced.png' %name)

data.to_csv('centrosymmetric_data.csv',mode='w')
