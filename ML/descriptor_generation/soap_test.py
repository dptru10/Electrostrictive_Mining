import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd 

centrosymmetric_structures=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/centrosymmetric_insulators.npy',allow_pickle=True)
task_ids=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/data/centrosymmetric_task_ids.npy',allow_pickle=True)
centro_elastic_compliance=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/elasticity_compliance/centro_elasticity.npy',allow_pickle=True)
centro_dielectric_tensor=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/dielectric_total/centro_diel.npy',allow_pickle=True)

data = {'structures': centrosymmetric_structures, 'ids' : task_ids}
df = pd.DataFrame(data)


#soap representation
#from matminer.featurizers.structure import SOAP 
#soap = SOAP(periodic=True)
#soap=soap.fit(data['structures'])
#labels=soap.feature_labels()
#df = soap.featurize_dataframe(df,'structures')

from matminer.featurizers.structure import SineCoulombMatrix
sine_coulomb = SineCoulombMatrix()
sine_coulomb.set_n_jobs(28)
sine_coulomb.fit(centrosymmetric_structures)#data['structures'])
labels=sine_coulomb.feature_labels()
df  = sine_coulomb.featurize_dataframe(df, 'structures')#,ignore_errors=True)

#agni
#from matminer.featurizers.site import AGNIFingerprints 
#agni=AGNIFingerprints(directions=['x','y','z']) 
#agni.set_n_jobs(28)
#labels=agni.feature_labels()
#df = agni.featurize(df['structures'],0)
#df  = agni.featurize_dataframe(df, ['structures', 'site'])#,ignore_errors=True)

#get s_vs_ep
ec_list=[]
for item in centro_elastic_compliance:
	obj=dict(item)
	if obj['elasticity.compliance_tensor'] != None:
		obj=obj['elasticity.compliance_tensor'][0][0]
	else: 
		obj=np.nan
	ec_list.append(obj)

dt_list=[]
for item in centro_dielectric_tensor:
	obj=dict(item)
	if obj['diel.e_total']!= None:
		obj=obj['diel.e_total'][0][0]
	else: 
		obj=np.nan
	dt_list.append(obj)

ep_0=8.85 
s_vs_ep=[]
for i in range(len(dt_list)):
    s_vs_ep.append(ep_0*ec_list[i]/dt_list[i]) 

endpoint='dt_list'
df[endpoint]=dt_list

df=df.dropna()
#df=df.loc[df[endpoint]>0.0]
print('len(df)')
print(len(df))
X=df[labels]
Y=df[endpoint]

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)
tuned_parameters = [{'n_estimators':[int(1e1),int(1e2),int(1e3)]}]
scores = ['neg_mean_absolute_error']
for score in scores:
    forest = GridSearchCV(ExtraTreesRegressor(random_state=1),tuned_parameters,verbose=10,cv=5,n_jobs=-1,scoring='%s' %score)
    
    forest.fit(X_train, y_train)
    model_train=forest.predict(X_train)
    model_test=forest.predict(X_test)
    r2_score_train=r2_score(y_train,model_train)
    mse_score_train=mean_squared_error(y_train,model_train)
    mae_score_train=mean_absolute_error(y_train,model_train)
    rmse_score_train=np.sqrt(mse_score_train)
    r2_score_test=r2_score(y_test,model_test)
    mse_score_test=mean_squared_error(y_test,model_test)
    mae_score_test=mean_absolute_error(y_test,model_test)
    rmse_score_test=np.sqrt(mse_score_test)

    dump(forest,'rf_s_vs_e.pkl')
    
    f=open('hyperpameters.txt',mode='w')
    f.write("Best parameters set found on development set:")
    f.write('\n\n')
    f.write(str(forest.best_params_))
    f.write('\n\n')
    f.write('Score:')
    f.write(str(-forest.best_score_))
    f.write('\n\n')
    
    f.write('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
    f.close() 
    
    #plot figures
    plt.figure()
    plt.title('Histogram forest Train')
    plt.hist2d(x=y_train,y=model_train,bins=100,norm=colors.LogNorm())   
    plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
    plt.colorbar() 
    plt.xlabel('Reported S$_{11}$/$\epsilon_{11}$')
    plt.ylabel('Predicted S$_{11}$/$\epsilon_{11}$')
    plt.tight_layout()
    plt.savefig('forest_histogram_train.png')
    
    plt.figure()
    plt.title('Histogram forest Test')
    plt.hist2d(x=y_test,y=model_test,bins=100,norm=colors.LogNorm())
    plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
    plt.colorbar() 
    plt.xlabel('Reported S$_{11}$/$\epsilon_{11}$')
    plt.ylabel('Predicted S$_{11}$/$\epsilon_{11}$')
    plt.tight_layout()
    plt.savefig('forest_histogram_test.png')

