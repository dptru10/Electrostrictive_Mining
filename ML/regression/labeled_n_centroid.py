import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler 
from numpy.linalg import eig
from sklearn.decomposition import PCA 
from sklearn.ensemble import IsolationForest
import matplotlib.colors as mcolors
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump 
import argparse 

ofms = np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/descriptors/ofm/centrosymmetric_ofm_representation.npy',allow_pickle=True)

parser = argparse.ArgumentParser()
parser.add_argument("file1")
parser.add_argument("--drop_anomaly",action="store_true")
parser.add_argument("--cluster_regression",action="store_true")
parser.add_argument("--global_regression",action="store_true")
args = parser.parse_args()


data = pd.read_csv(args.file1) 
data['ofms'] = pd.Series(list(ofms))
features = ['elastic_compliance','dielectric_tensor']

X_fit = data[features]
X_fit['labels'] = data['labels'] 
X_fit['ofms']   = data['ofms']
X_fit.to_csv('classes_data.csv')

#select=pd.DataFrame(columns=X.columns)
#select_materials=['FeCl2','LiFeF4','Cs2Pd3S4'] 
#for item in select_materials: 
#	select=select.append(X.loc[X['pretty_formula']==item])

plt.figure()
fig=plt.figure()
ax=fig.add_subplot(111)
plt.scatter(X_fit['dielectric_tensor'],X_fit['elastic_compliance'], c=X_fit['labels'],cmap=plt.cm.jet)
i=0
#for pts in zip(select['dielectric_tensor'],select['elastic_compliance']):
#	ax.annotate(str(select['pretty_formula'].iloc[i]),pts)
#	i+=1 

plt.title('Distribution of $S$ vs $\epsilon$ eigenvalues')
plt.xlabel("Dielectric Tensor Average Eigenvalues")
plt.ylabel("Elastic Compliance Average Eignevalues")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout() 
plt.savefig('centro_dataset_labeled_clusters.png')

big_X = pd.DataFrame(columns=X_fit.columns)
for label in set(X_fit['labels']):
    X_new = X_fit.loc[X_fit['labels'] == int(label)]
    if len(X_new) > 3: 
        features=['dielectric_tensor','elastic_compliance']
        isolated_forest=IsolationForest(n_estimators=100,max_samples='auto',behaviour='new',contamination='auto',max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
        
        isolated_forest.fit(X_new[features])
        predicted=isolated_forest.predict(X_new[features]) 
        
        X_new['anomaly']=predicted
        outliers=X_new.loc[X_new['anomaly']==-1]
        
        print('Number of anomalies in cluster %i' %label)
        print(X_new['anomaly'].value_counts())
    else: 
        print("cluster %i has less than 5 elements, this cluster is an anomaly..." %label)
        X_new['anomaly'] = -1 
    big_X=big_X.append(X_new) 
    big_X = big_X.loc[big_X['anomaly']==1]
big_X.to_csv('anomaly.csv')

if args.drop_anomaly is True: 
	big_X = big_X.loc[big_X['anomaly']==1]

plt.figure()
fig=plt.figure()
ax=fig.add_subplot(111)
plt.scatter(big_X['dielectric_tensor'],big_X['elastic_compliance'], c=big_X['labels'],cmap=plt.cm.jet)
i=0
#for pts in zip(select['dielectric_tensor'],select['elastic_compliance']):
#	ax.annotate(str(select['pretty_formula'].iloc[i]),pts)
#	i+=1 

plt.title('Distribution of $S$ vs $\epsilon$ eigenvalues')
plt.xlabel("Dielectric Tensor Average Eigenvalues")
plt.ylabel("Elastic Compliance Average Eignevalues")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout() 
plt.savefig('centro_dataset_labeled_anomaly_removed_clusters.png')


if args.cluster_regression is True:
	cv_val = 5 
	for label in set(X_fit['labels']):
		print("training model for cluster %i" %label)
		data = big_X.loc[big_X['labels'] == label]

		if len(data) > cv_val:
			endpoints = ['dielectric_tensor','elastic_compliance']
			for endpoint in endpoints:  
				feature  = 'ofms' 

				X = np.array(data[feature])
				Y = np.array(data[endpoint])
				X_clean=np.zeros(shape=(np.shape(X)[0],1024))
				for i in range(len(X)): 
				    X_clean[i]=X[i].reshape(1,1024)
				X=X_clean
				X = X.reshape(np.shape(X)[0],1024)#np.ones(shape=np.shape(X)[0],1024))

				# make training and test set
				X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)
				
				tuned_parameters = [{'n_estimators':[100,200,500]}]
				scores = ['neg_mean_absolute_error']
				for score in scores:
				   forest = GridSearchCV(ExtraTreesRegressor(),tuned_parameters,verbose=10,cv=cv_val,n_jobs=-1,scoring='%s' %score)
				   
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
				
				   dump(forest,'%s_cluster_%i.pkl' %(endpoint,label))
				   
				   f=open('%s_hyperpameters_cluster_%i.txt' %(endpoint,label),mode='w')
				   f.write("Best parameters set found on development set:")
				   f.write('\n\n')
				   f.write(str(forest.best_params_))
				   f.write('\n\n')
				   f.write('Score:')
				   f.write(str(-forest.best_score_))
				   f.write('\n\n')
				   
				   f.write('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
				   f.close() 
				   
				   if endpoint == 'dielectric_tensor': 
				   	math_label = '$\epsilon$'
				   if endpoint == 'elastic_compliance': 
				   	math_label = '$S$'

				   #plot figures
				   plt.figure()
				   plt.title('Histogram forest Train')
				   plt.hist2d(x=y_train,y=model_train,bins=100,norm=colors.LogNorm())   
				   plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
				   plt.colorbar() 
				   plt.xlabel('Reported %s Eig.' %math_label)
				   plt.ylabel('Predicted %s Eig' %math_label)
				   plt.tight_layout()
				   plt.savefig('%s_forest_histogram_train_cluster_%i.png' %(endpoint,label))
				   
				   plt.figure()
				   plt.title('Histogram forest Test')
				   plt.hist2d(x=y_test,y=model_test,bins=100,norm=colors.LogNorm())
				   plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
				   plt.colorbar() 
				   plt.xlabel('Reported %s Eig.' %math_label)
				   plt.ylabel('Predicted %s Eig' %math_label)
				   plt.tight_layout()
				   plt.savefig('%s_forest_histogram_test_cluster_%i.png' %(endpoint,label))
	   
	   
if args.global_regression is True:
	label  = 'global' 
	cv_val = 5 
	data   = big_X
	data   = data.loc[data['anomaly'] == 1]

	if len(data) > cv_val:
		endpoints = ['dielectric_tensor','elastic_compliance']
		for endpoint in endpoints:  
			feature  = 'ofms' 

			X = np.array(data[feature])
			Y = np.array(data[endpoint])
			X_clean=np.zeros(shape=(np.shape(X)[0],1024))
			for i in range(len(X)): 
			    X_clean[i]=X[i].reshape(1,1024)
			X=X_clean
			X = X.reshape(np.shape(X)[0],1024)#np.ones(shape=np.shape(X)[0],1024))

			# make training and test set
			X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)
			
			tuned_parameters = [{'n_estimators':[10,20,50,100,200,500]}]
			scores = ['neg_mean_absolute_error']
			for score in scores:
			   forest = GridSearchCV(ExtraTreesRegressor(),tuned_parameters,verbose=10,cv=cv_val,n_jobs=-1,scoring='%s' %score)
			   
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
			
			   dump(forest,'%s_cluster_%s.pkl' %(endpoint,label))
			   
			   f=open('%s_hyperpameters_cluster_%s.txt' %(endpoint,label),mode='w')
			   f.write("Best parameters set found on development set:")
			   f.write('\n\n')
			   f.write(str(forest.best_params_))
			   f.write('\n\n')
			   f.write('Score:')
			   f.write(str(-forest.best_score_))
			   f.write('\n\n')
			   
			   f.write('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
			   f.close() 
			   
			   if endpoint == 'dielectric_tensor': 
			   	math_label = '$\epsilon$'
			   if endpoint == 'elastic_compliance': 
			   	math_label = '$S$'

			   #plot figures
			   plt.figure()
			   plt.title('Histogram forest Train')
			   plt.hist2d(x=y_train,y=model_train,bins=100,norm=colors.LogNorm())   
			   plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
			   plt.colorbar() 
			   plt.xlabel('Reported %s Eig.' %math_label)
			   plt.ylabel('Predicted %s Eig' %math_label)
			   plt.tight_layout()
			   plt.savefig('%s_forest_histogram_train_cluster_%s.png' %(endpoint,label))
			   
			   plt.figure()
			   plt.title('Histogram forest Test')
			   plt.hist2d(x=y_test,y=model_test,bins=100,norm=colors.LogNorm())
			   plt.axis([np.min(Y),np.max(Y),np.min(Y),np.max(Y)])
			   plt.colorbar() 
			   plt.xlabel('Reported %s Eig.' %math_label)
			   plt.ylabel('Predicted %s Eig' %math_label)
			   plt.tight_layout()
			   plt.savefig('%s_forest_histogram_test_cluster_%s.png' %(endpoint,label))

