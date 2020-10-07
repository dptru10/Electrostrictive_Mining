import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

centro_structs  = np.load('/home/dennis/Dropbox/mp_share_dt_ag/structs/centrosymmetric_insulators_all_insulating.npy',allow_pickle=True)
centro_elastic  = np.load('/home/dennis/Dropbox/mp_share_dt_ag/structs/centro_elastic_tensor_all_insulating.npy',allow_pickle=True)
centro_dielect  = np.load('/home/dennis/Dropbox/mp_share_dt_ag/structs/centro_diel_tensor_all_insulating.npy',allow_pickle=True)



K_VRH=[]
for elastic in centro_elastic: 
    if elastic['elasticity'] != None: 
        K_VRH.append(np.log10(elastic['elasticity']['K_VRH']))
    else: 
        K_VRH.append(np.nan)

diel=[]
for dielectric in centro_dielect: 
    if dielectric['diel'] != None: 
        w,v=np.linalg.eig(dielectric['diel']['e_total'])
        diel.append(np.log10(np.mean(np.real(w)))) 
    else: 
        diel.append(np.nan)

mh=[]
for i in range(len(diel)): 
    mh.append(1/(K_VRH[i] * diel[i]))

df = pd.DataFrame(columns=['structure','Mh'])

df['structure'] = centro_structs
df['Mh']   = mh #df['diel']*df['K_VRH']
df=df.replace([np.inf,-np.inf],np.nan)
df=df.dropna() 
df.to_csv('Mh_test.csv')
print(df.describe())

target = 'Mh'
train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=1)
prediction_df = test_df.drop(target)#['Mh','K_VRH','diel'],axis=1)
print(prediction_df.columns) 

from automatminer import MatPipe
pipe = MatPipe.from_preset("debug",n_jobs=28)#,cache_src='Mh_cache.json')
pipe.fit(train_df, target)

prediction_df = pipe.predict(prediction_df)

from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor

# fit the dummy
dr = DummyRegressor()
dr.fit(train_df["structure"], train_df[target])
dummy_test = dr.predict(test_df["structure"])


# Score dummy and MatPipe
true = test_df[target]
matpipe_test = prediction_df[target + " predicted"]

mae_matpipe = mean_absolute_error(true, matpipe_test)
mae_dummy = mean_absolute_error(true, dummy_test)

print("Dummy MAE: {} eV".format(mae_dummy))
print("MatPipe MAE: {} eV".format(mae_matpipe))

import pprint

# Get a summary and save a copy to json
summary = pipe.summarize(filename="MatPipe_predict_experimental_gap_from_composition_summary.json")

pprint.pprint(summary)

# Access some attributes of MatPipe directly, instead of via a text digest
out_info = open('model_info.txt','w')
out_info.write(pipe.learner.best_pipeline)

out_info.write(pipe.autofeaturizer.featurizers["composition"])
out_info.close()

filename = "MatPipe_test_centro_KVRH.p"
pipe.save(filename)

