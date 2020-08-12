from pymatgen.ext.matproj import MPRester 
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
import pandas as pd

data = pd.read_csv('new_materials.csv')

mpr = MPRester() 

user_incar_settings={}
hubbard_off = False 

i=0
for mp_id in data['task_id']: 
    struct = mpr.get_structure_by_material_id(str(mp_id))
    relaxed    = MPRelaxSet(struct)
    relaxed.write_input("/tmp/structs/%s_%s" %(mp_id,data['labels'].iloc[i]),make_dir_if_not_present=True, potcar_spec=True)
    i+=1 
