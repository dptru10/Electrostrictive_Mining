import numpy as np
from pymatgen import MPRester, Structure

mpr = MPRester()
centrosymm=["-1","2/m","mmm","4/m","4/mmm","-3","-3m","6/m","6/mmm","m-3","m-3m"]
non_centro=["1","2","m","222","mm2","4","-4","422","4mm","-42m","3","32","3m","6","-6","622","6mm","-6m2","23","432","-43m"]

#find centro and non-centro materials based on pt grps  
cen_data=[]
i=0
for item in centrosymm: 
	cen_data.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["elasticity"]))
	i+=1
i=0
print()
print()
print("centrosymmetric structures:")
for item in cen_data: 
	print("point group %s, len: %i" %(centrosymm[i],len(item)))
	i+=1 

j=0
non_cen_data=[]
for item in non_centro: 
	non_cen_data.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["elasticity"]))
	j+=1
i=0
print()
print()
print("non-centrosymmetric structures:")
for item in non_cen_data: 
	print("point group %s, len: %i" %(non_centro[i],len(item)))
	i+=1 

cen_structures=[]
k=0
for struct in cen_data:
     for obj in struct: 
          #print("centrosym struct %i of %i" %(k,i)) 
          cen_structures.append(obj)
          k+=1
print("Structurized %i total centrosymmetric structures" %k)

non_cen_structures=[]
l=0
for struct in non_cen_data:
    for obj in struct:  
         #print("non-centro struct %i of %i" %(l,j))
         non_cen_structures.append(obj)
         l+=1

np.save('centro_elastic_tensor.npy',cen_structures)
np.save('non_centro_elastic_tensor.npy',non_cen_structures)
