import numpy as np 
from matminer.featurizers.structure import RadialDistributionFunction

cen_structures=np.load('centrosymmetric_insulators.npy',allow_pickle=True)

#create rdf representations
rdf = RadialDistributionFunction()#r_cut=2.0,periodic=True,normalize=True)
cen_rdf=[]

for item in cen_structures: 
	cen_rdf.append(rdf.featurize(item))
np.save('centrosymmetric_rdf_representation.npy',cen_rdf)

non_cen_structures=np.load('non_centrosymmetric_insulators.npy',allow_pickle=True) 
non_cen_rdf=[]
for item in non_cen_structures:
	non_cen_rdf.append(rdf.featurize(item))
np.save('non_centrosymmetric_rdf_representation.npy',non_cen_rdf)
