from matminer.featurizers.structure import ElectronicRadialDistributionFunction
import numpy as np 

cen_structures=np.load('centrosymmetric_insulators.npy',allow_pickle=True)
cen_redf=[]

non_cen_structures=np.load('non_centrosymmetric_insulators.npy',allow_pickle=True) 
non_cen_redf=[]

#create redf representations
redf = ElectronicRadialDistributionFunction()
redf.set_n_jobs(28)

redf.fit(cen_structures)
for item in cen_structures: 
	cen_redf.append(redf.featurize(item))
np.save('centrosymmetric_redf_representation.npy',cen_redf)


redf.fit(non_cen_structures)
for item in non_cen_structures:
	non_cen_redf.append(redf.featurize(item))
np.save('non_centrosymmetric_redf_representation.npy',non_cen_redf)
