import numpy as np
from pymatgen import MPRester, Structure

mpr = MPRester()
centrosymm=["-1","2/m","mmm","4/m","4/mmm","-3","-3m","6/m","6/mmm","m-3","m-3m"]
non_centro=["1","2","m","222","mm2","4","-4","422","4mm","-42m","3","32","3m","6","-6","622","6mm","-6m2","23","432","-43m"]

#find centro and non-centro materials based on pt grps  
cen_data=[]
i=0
for item in centrosymm: 
	cen_data.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["cif"]))
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
	non_cen_data.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["cif"]))
	j+=1
i=0
print()
print()
print("non-centrosymmetric structures:")
for item in non_cen_data: 
	print("point group %s, len: %i" %(non_centro[i],len(item)))
	i+=1 

#get mp_ids 
cen_task_id=[]
i=0
for item in centrosymm: 
	cen_task_id.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["task_id"]))
	i+=1
np.save('centrosymmetric_task_ids.npy',cen_task_id,allow_pickle=True)

j=0
non_cen_task_id=[]
for item in non_centro: 
	non_cen_task_id.append(mpr.query(criteria={"has":{"$all":["diel","elasticity"]},"band_gap":{"$gt":0.5},"spacegroup.point_group":{"$all":[item]},"e_above_hull":{"$lt":0.2},"nelements":{"$gt":1}},properties=["task_id"]))
	j+=1
np.save('non_centrosymmetric_task_ids.npy',non_cen_task_id,allow_pickle=True)

#structurize materials
cen_structures=[]
k=0
for struct in cen_data:
     for obj in struct: 
          #print("centrosym struct %i of %i" %(k,i)) 
          cen_structures.append(Structure.from_str(obj["cif"], fmt="cif"))#.to(fmt="poscar"))
          k+=1
print("Structurized %i total centrosymmetric structures" %k)
np.save('centrosymmetric_insulators.npy',cen_structures,allow_pickle=True)

non_cen_structures=[]
l=0
for struct in non_cen_data:
    for obj in struct:  
         #print("non-centro struct %i of %i" %(l,j))
         non_cen_structures.append(Structure.from_str(obj["cif"], fmt="cif"))#.to(fmt="poscar"))
         l+=1
print("Structurized %i total non-centrosymmetric structures" %l)
np.save('non_centrosymmetric_insulators.npy',non_cen_structures,allow_pickle=True)
