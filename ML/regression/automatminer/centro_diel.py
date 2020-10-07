import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.dummy import DummyRegressor
from automatminer import MatPipe

centro_structs  = np.load('/home/dennis/Dropbox/electrostrictive_mining_ml/structs/centrosymmetric_insulators_all_insulating.npy',allow_pickle=True)
centro_dielect  = np.load('/home/dennis/Dropbox/electrostrictive_mining_ml/structs/centro_diel_tensor_all_insulating.npy',allow_pickle=True)

print(len(centro_structs))
print(len(centro_dielect))


diel=[]
for dielectric in centro_dielect: 
    if dielectric['diel'] != None: 
        w,v=np.linalg.eig(dielectric['diel']['e_total'])
        diel.append(np.log10(np.mean(np.real(w))))
    else: 
        diel.append(np.nan)

df = pd.DataFrame(columns=['structure','dielectric'])
df['structure']  = centro_structs
df['dielectric'] = diel

df=df.dropna()
df.to_csv('centro_diel.csv')
print(df.describe())

train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=1)
target = "dielectric"
prediction_df = test_df.drop(columns=[target])

pipe = MatPipe.from_preset("express",n_jobs=28,cache_src="cache_diel.json")
pipe.fit(train_df, target)

prediction_df = pipe.predict(prediction_df)

# fit the dummy
dr = DummyRegressor()
dr.fit(train_df["structure"], train_df[target])
dummy_test = dr.predict(test_df["structure"])


# Score dummy and MatPipe
true = test_df[target]
matpipe_test = prediction_df[target + " predicted"]

mae_matpipe = mean_absolute_error(true, matpipe_test)
mse_matpipe = mean_squared_error(true,matpipe_test)
r2_matpipe =r2_score(true,matpipe_test)

mae_dummy = mean_absolute_error(true, dummy_test)
mse_dummy = mean_squared_error(true,dummy_test)
r2_dummy  = r2_score(true,dummy_test)

print("Dummy MAE: {} eV".format(mae_dummy))
print("Dummy MSE: {} eV".format(mse_dummy))
print("Dummy R2: {} eV".format(r2_dummy))
print("MatPipe MAE: {} eV".format(mae_matpipe))
print("MatPipe MSE: {} eV".format(mse_matpipe))
print("MatPipe R2: {} eV".format(r2_matpipe))

plt.figure()
plt.title('Dummy Test')
plt.hist2d(x=true,y=dummy_test,bins=100,norm=colors.LogNorm())
#plt.gca().set_aspect('equal', adjustable='box')
plt.axis([np.min(true),np.max(dummy_test),np.min(true),np.max(dummy_test)])
plt.colorbar() 
plt.xlabel('Reported $\varepsilon$')
plt.ylabel('Predicted $\varepsilon$')
plt.savefig('dummy_kvrh.png')


plt.figure()
plt.title('Model Test')
plt.hist2d(x=true,y=matpipe_test,bins=100,norm=colors.LogNorm())
#plt.gca().set_aspect('equal', adjustable='box')
plt.axis([np.min(true),np.max(matpipe_test),np.min(true),np.max(matpipe_test)])
plt.colorbar() 
plt.xlabel('Reported $\varepsilon$')
plt.ylabel('Predicted $\varepsilon$')
plt.savefig('automatminer_kvrh.png')


import pprint

# Get a summary and save a copy to json
summary = pipe.summarize(filename="MatPipe_predict_centro_diel.json")

pprint.pprint(summary)

# Access some attributes of MatPipe directly, instead of via a text digest
print(pipe.learner.best_pipeline)

print(pipe.autofeaturizer.featurizers["composition"])

filename = "MatPipe_test_centro_diel.p"
pipe.save(filename)
