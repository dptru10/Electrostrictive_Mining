import numpy as np 
from matminer.featurizers.structure import OrbitalFieldMatrix 

cen_structures=np.load('/home/dennis/Dropbox/mp_share_dt_ag/structs/centrosymmetric_insulators.npy',allow_pickle=True)
print(len(cen_structures)) 

#create ofm representations
ofm = OrbitalFieldMatrix()
cen_ofm=[]

for item in cen_structures: 
	cen_ofm.append(ofm.featurize(item))
np.save('centrosymmetric_ofm_representation.npy',cen_ofm)

non_cen_structures=np.load('/home/dennis/Dropbox/mp_share_dt_ag/structs/non_centrosymmetric_insulators.npy',allow_pickle=True) 
print(len(non_cen_structures)) 

non_cen_ofm=[]
for item in non_cen_structures:
	non_cen_ofm.append(ofm.featurize(item))
np.save('non_centrosymmetric_ofm_representation.npy',non_cen_ofm)
