from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import pandas as pd 
from numpy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse 

newnham_data = pd.read_csv('../newnham_properties.csv')
our_data     = pd.read_csv('../top_electrostrictive_candidates.csv')

#our_data['ratio'] = our_data['kv/e'] / our_data['Qh']

#our_data=our_data.sort_values('ratio',ascending=False)
#our_data=our_data.loc[our_data['ratio']>=10]
#our_data=our_data.head(n=5)


combined     = newnham_data.append(our_data)

select=pd.DataFrame(columns=combined.columns)
select_materials=[]#['Polyurethane','PVDF','PVC','Spodumene glass','Fused SiO2','AlN','Pb(TiZr)O3','PLZT'] 
select_materials += list(combined['labels'])
for item in select_materials: 
	select=select.append(combined.loc[combined['labels']==str(item)])
 
fig=plt.figure()
ax=fig.add_subplot(111)
plt.title('$Q_{h}$ vs $\epsilon_{0}\epsilon_{r}$')
i=0
for pts in zip(select['Qh'],select['kv/e']):
	ax.annotate(str(select['labels'].iloc[i]),pts)
	i+=1
plt.scatter(combined['Qh'],combined['kv/e'],c=combined['Class'],cmap=plt.cm.jet)
plt.axis([1e-3,1e4,1e-1,1e7])
plt.yscale('log')
plt.xscale('log')
plt.xlabel("$Q_{h}$ [$m^{4}/C^{2}$]")
plt.ylabel("$s/\epsilon_{0}\epsilon_{r}$ ")
plt.show()#savefig('newnham_replot.png')
