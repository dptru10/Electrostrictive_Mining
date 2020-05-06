import numpy as np 
from matminer.featurizers.structure import CoulombMatrix 

cen_structures=np.load('centrosymmetric_insulators.npy',allow_pickle=True)

#create cm representations
cm = CoulombMatrix()
cen_cm=[]

for item in cen_structures: 
	cen_cm.append(cm.featurize(item))
np.save('centrosymmetric_cm_representation.npy',cen_cm)

non_cen_structures=np.load('non_centrosymmetric_insulators.npy',allow_pickle=True) 
non_cen_cm=[]
for item in non_cen_structures:
	non_cen_cm.append(cm.featurize(item))
np.save('non_centrosymmetric_cm_representation.npy',non_cen_cm)
