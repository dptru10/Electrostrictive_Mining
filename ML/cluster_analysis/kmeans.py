import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler 
from numpy.linalg import eig
from sklearn.decomposition import PCA 
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--normalize",action="store_true")
parser.add_argument("--drop_anomaly",action="store_true")
args = parser.parse_args()

centro_elastic_compliance=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/elasticity_compliance/centro_elasticity.npy',allow_pickle=True)
centro_dielectric_tensor=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/dielectric_total/centro_diel.npy',allow_pickle=True)
labels=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/data/centro_names.npy',allow_pickle=True)
mp_ids=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/data/centrosymmetric_task_ids.npy',allow_pickle=True)

outliers = []#'Cs2Pd3S4','LiFeF4','FeCl2']
ec_list=[]
Kv     =[]
for item in centro_elastic_compliance:
	obj=dict(item)
	if obj['elasticity.compliance_tensor'] != None:
                obj=obj['elasticity.compliance_tensor']
                w, v = eig(obj)
                w    = np.average(np.real(w))
                k    = ((obj[0][0] + obj[1][1] + obj[2][2]) + 2 * (obj[0][1] + obj[1][2] + obj[2][0]))/9
	else: 
		w    = np.nan
		k    = np.nan 
	ec_list.append(w)
	Kv.append(k)

dt_list=[]
for item in centro_dielectric_tensor:
	obj=dict(item)
	if obj['diel.e_total']!= None:
		obj=obj['diel.e_total']
		w, v = eig(obj)
		w    = np.average(np.real(w))
	else: 
		w    = np.nan
	dt_list.append(w)

s_vs_ep=[]
kv_vs_ep=[]
for i in range(len(dt_list)):
	s_vs_ep.append((ec_list[i]/dt_list[i])/8.85) 
	kv_vs_ep.append((Kv[i]/dt_list[i])/8.85)

X  = pd.DataFrame() 
X['elastic_compliance'] = ec_list 
X['dielectric_tensor']  = dt_list
X['mp_id']              = mp_ids
X['pretty_formula']     = labels
X['bulk modulus']       = Kv
X = X.loc[X['elastic_compliance'] > 0]
X = X.loc[X['dielectric_tensor'] > 0]

for label in outliers: 
	X = X.loc[X['pretty_formula'] != label]
X=X.dropna() 

features = ['bulk modulus','dielectric_tensor']
X_fit = X[features]

if args.normalize is True: 
	mms = MinMaxScaler()
	mms.fit(X_fit) 
	data_transform=mms.transform(X_fit) 

if args.drop_anomaly is True: 
	isolated_forest=IsolationForest(n_estimators=100,max_samples='auto',behaviour='new',contamination='auto',max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
	isolated_forest.fit(X_fit)#data_transform)
	predicted = isolated_forest.predict(X_fit)#data_transform)
	X_fit['anomaly'] = predicted 
	X_fit = X_fit.loc[X_fit['anomaly']==1]

# apply kmeans-silhouette
summed_square_distance=[]
calinski_score=[]
clusters=range(2,11)
for i in clusters:
    model = KMeans(n_clusters=i, random_state=0)
    if args.normalize is True: 
    	model.fit(data_transform)
    else: 
    	model.fit(X_fit)
    centroids = model.cluster_centers_
    #X_reduced = data_transform#PCA(n_components=2).fit_transform(data_transform)
    #plt.scatter(data_transform[:,0], data_transform[:,1], c=model.labels_,cmap=plt.cm.jet)
    plt.scatter(X_fit['dielectric_tensor'],X_fit['bulk modulus'], c=model.labels_,cmap=plt.cm.jet)
    plt.xlabel("Normalized Dielectric Tensor Average Eigenvalues")
    plt.ylabel("Bulk modulus Voight ave. ($k_{v}$) [$10^{-12}$ $pa^{-1}$]")
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout() 
    plt.savefig('centro_dataset_%i_clusters.png' %i)
    df=pd.DataFrame(columns=X.columns)
    j=0 
    summed_square_distance.append(model.inertia_)
    if args.normalize is True: 
    	calinski_score.append(calinski_harabasz_score(data_transform,model.labels_))
    	#calinski_score = np.array(calinski_score)
    	#calinski_score = calinski_score.reshape(-1,1)
    	#calinski_transform=mms.fit_transform(calinski_score)
    else:
    	calinski_score.append(calinski_harabasz_score(X_fit,model.labels_))
    X_fit['labels'] = model.labels_
    X_fit.to_csv('data_%i_clusters.csv' %i)
plt.figure() 
plt.plot(clusters, summed_square_distance, 'bx-')
#plt.plot(clusters, calinski_score,'rx-',label='calinski_harabasz')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum_of_squared_distances')
#plt.yscale('log')
plt.title('Elbow Method For Optimal k')
plt.tight_layout() 
plt.savefig('elbow_method_centro_dataset.png')

plt.figure() 
#plt.plot(clusters, summed_square_distance, 'bx-')
plt.plot(clusters, calinski_score,'rx-',label='calinski_harabasz')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum_of_squared_distances')
#plt.yscale('log')
plt.title('Elbow Method For Optimal k')
plt.tight_layout() 
plt.savefig('silhouette_method_centro_dataset.png')

