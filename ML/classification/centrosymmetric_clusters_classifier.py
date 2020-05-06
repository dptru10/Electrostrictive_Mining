import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump 
import matplotlib.pyplot as plt 
from collections import Counter
from imblearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix 
from sklearn.svm import LinearSVC 
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, CondensedNearestNeighbour, NearMiss

ofms = np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/descriptors/ofm/centrosymmetric_ofm_representation.npy',allow_pickle=True)

cv_val =3 
data = pd.read_csv('anomaly.csv') 
data['ofms'] = pd.Series(list(ofms))
data = data.loc[data['anomaly']==1] 

endpoint = 'labels'
feature  = 'ofms' 
                                                                                                                     
X = np.array(data[feature])
Y = np.array(data[endpoint])
Y = Y.astype('int')
X_clean=np.zeros(shape=(np.shape(X)[0],1024))
for i in range(len(X)): 
    X_clean[i]=X[i].reshape(1,1024)
X=X_clean
X = X.reshape(np.shape(X)[0],1024)#np.ones(shape=np.shape(X)[0],1024))

print('Original dataset shape %s' % Counter(Y))

cnn           = CondensedNearestNeighbour(sampling_strategy='majority',random_state=0)
random_sample = RandomUnderSampler(random_state=0)
X_res, Y_res = random_sample.fit_resample(X, Y)                                                                                                                      
print('Resampled dataset shape %s' % Counter(Y_res))

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X_res, Y_res, test_size=0.1,random_state=1)

#pipeline attempt
#pipeline = make_pipeline(ClusterCentroids(random_state=0),
#                         ExtraTreesClassifier(n_estimators=50))
#
#pipeline.fit(X_train, y_train)
#predict=pipeline.predict(X_test)
#cm=confusion_matrix(y_test,predict)
#cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print(cm)
#print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))

tuned_parameters = [{'n_estimators':[50,100,200,500,700,1000]}]
score = 'balanced_accuracy'

#balanced forest 
balanced_forest = GridSearchCV(BalancedRandomForestClassifier(random_state=0),tuned_parameters,verbose=0,cv=cv_val,n_jobs=-1,scoring='%s' %score)

balanced_forest.fit(X_train, y_train)
model_train=balanced_forest.predict(X_train)
model_test=balanced_forest.predict(X_test)

dump(balanced_forest,'centro_cluster_classifier_brf.pkl')

f=open('centro_hyperpameters_cluster_classifier_brf.txt',mode='w')
f.write("Best parameters set found on development set:")
f.write('\n\n')
f.write(str(balanced_forest.best_params_))
f.write('\n\n')
f.write('Score:')
f.write(str(balanced_forest.best_score_))
f.write('\n\n') 

cm=confusion_matrix(np.array(y_test),np.array(model_test))#balanced_forest,X_test,y_test,cmap=plt.cm.Blues,normalize=None)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure() 
sns.heatmap(cm_normalized,cmap=plt.cm.Blues)
plt.title("Cluster Based Classification of Centrosymmetric Data")
plt.savefig('centro_confusion_matrix_brf.png') 


#random forest 
forest = GridSearchCV(RandomForestClassifier(random_state=0),tuned_parameters,verbose=0,cv=cv_val,n_jobs=-1,scoring='%s' %score)

forest.fit(X_train, y_train)
model_train=forest.predict(X_train)
model_test=forest.predict(X_test)

dump(forest,'centro_cluster_classifier_rf.pkl')

f=open('centro_hyperpameters_cluster_classifier_rf.txt',mode='w')
f.write("Best parameters set found on development set:")
f.write('\n\n')
f.write(str(forest.best_params_))
f.write('\n\n')
f.write('Score:')
f.write(str(forest.best_score_))
f.write('\n\n') 

cm = confusion_matrix(np.array(y_test),np.array(model_test))#forest,X_test,y_test,cmap=plt.cm.Blues,normalize=None)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized) 
plt.figure() 
sns.heatmap(cm_normalized,cmap=plt.cm.Blues)
plt.title("Cluster Based Classification of Centrosymmetric Data")
plt.savefig('centro_confusion_matrix_rf.png') 


#ExtraTrees 
ExtraTrees = GridSearchCV(ExtraTreesClassifier(random_state=0),tuned_parameters,verbose=0,cv=cv_val,n_jobs=-1,scoring='%s' %score)

ExtraTrees.fit(X_train, y_train)
model_train=ExtraTrees.predict(X_train)
model_test=ExtraTrees.predict(X_test)

dump(ExtraTrees,'centro_cluster_classifier_et.pkl')

f=open('centro_hyperpameters_cluster_classifier_et.txt',mode='w')
f.write("Best parameters set found on development set:")
f.write('\n\n')
f.write(str(ExtraTrees.best_params_))
f.write('\n\n')
f.write('Score:')
f.write(str(ExtraTrees.best_score_))
f.write('\n\n') 

cm = confusion_matrix(np.array(y_test),np.array(model_test))#ExtraTrees,X_test,y_test,cmap=plt.cm.Blues,normalize=None)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized) 
plt.figure() 
sns.heatmap(cm_normalized,cmap=plt.cm.Blues)
plt.title("Cluster Based Classification of Centrosymmetric Data")
plt.savefig('centro_confusion_matrix_et.png') 



