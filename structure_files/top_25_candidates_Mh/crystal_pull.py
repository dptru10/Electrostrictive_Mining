import pandas as pd 
import argparse 
import numpy as np
from pymatgen import MPRester, Structure

mpr = MPRester()

parser= argparse.ArgumentParser()
parser.add_argument("file", help="file to be read in", type=str)
args  = parser.parse_args() 

data    = pd.read_csv(args.file)

spacegroup_symbol = []
crystal_system    = []
for mp_id in data['task_id']:
	symbol  = mpr.query(criteria={"task_id":mp_id},properties=["spacegroup.symbol"])
	crystal = mpr.query(criteria={"task_id":mp_id},properties=["spacegroup.crystal_system"])
	spacegroup_symbol.append(symbol[0]['spacegroup.symbol'])
	crystal_system.append(crystal[0]['spacegroup.crystal_system'])

structs = pd.DataFrame(columns=['labels','task_id','spacegroup','crystal_structure'])
structs['labels']            = data['labels']
structs['task_id']           = data['task_id']
structs['spacegroup']        = spacegroup_symbol
structs['crystal_structure'] = crystal_system 
structs.to_csv('crystal_systems.csv')
