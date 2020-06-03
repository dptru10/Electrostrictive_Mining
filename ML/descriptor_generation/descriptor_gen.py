import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from numpy.linalg import eig
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd 
import itertools 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--read_dataframe",action="store_true")
args = parser.parse_args()

centrosymmetric_structures=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/centrosymmetric_insulators.npy',allow_pickle=True)
task_ids=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/data/centrosymmetric_task_ids.npy',allow_pickle=True)
centro_elastic_compliance=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/elasticity_compliance/centro_elasticity.npy',allow_pickle=True)
centro_dielectric_tensor=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/dielectric_total/centro_diel.npy',allow_pickle=True)

if args.read_dataframe is True:
	df = pd.read_csv('featurized_dataframe.csv')
 
else: 
	print(len(centrosymmetric_structures))
	print(len(task_ids))
	data = {'structures': centrosymmetric_structures, 'ids' : task_ids }
	df = pd.DataFrame(data)
	
	labels=[]
	##soap representation
	#from matminer.featurizers.structure import SOAP 
	#soap = SOAP()
	#soap.set_n_jobs(6)
	#soap=soap.fit(centrosymmetric_structures)
	#labels.append(soap.feature_labels())
	#df = soap.featurize_dataframe(df,'structures')
	
	##partial radial distribution function 
	#from matminer.featurizers.structure import PartialRadialDistributionFunction
	#partial_rdf = PartialRadialDistributionFunction()
	#partial_rdf.set_n_jobs(28)
	#partial_rdf.fit(centrosymmetric_structures)
	#labels.append(partial_rdf.feature_labels()) 
	#df  = partial_rdf.featurize_dataframe(df, 'structures',ignore_errors=False)
	
	#sine coulomb 
	from matminer.featurizers.structure import SineCoulombMatrix
	sine_coulomb = SineCoulombMatrix()
	sine_coulomb.set_n_jobs(28)
	sine_coulomb.fit(centrosymmetric_structures)
	labels.append(sine_coulomb.feature_labels()) 
	df  = sine_coulomb.featurize_dataframe(df, 'structures',ignore_errors=False)
	
	#orbital field matrix 
	from matminer.featurizers.structure import OrbitalFieldMatrix
	ofm = OrbitalFieldMatrix()
	ofm.set_n_jobs(28)
	ofm.fit(centrosymmetric_structures)
	labels.append(ofm.feature_labels()) 
	df  = ofm.featurize_dataframe(df, 'structures',ignore_errors=False)
	
	#chemical ordering
	from matminer.featurizers.structure import ChemicalOrdering 
	chemical_ordering = ChemicalOrdering()
	chemical_ordering.set_n_jobs(28)
	labels.append(chemical_ordering.feature_labels()) 
	df  = chemical_ordering.featurize_dataframe(df, 'structures',ignore_errors=False)
	
	#bond fraction
	from matminer.featurizers.structure import BondFractions 
	bond_fraction = BondFractions()
	bond_fraction.set_n_jobs(28)
	bond_fraction=bond_fraction.fit(centrosymmetric_structures)
	labels.append(bond_fraction.feature_labels()) 
	df  = bond_fraction.featurize_dataframe(df, 'structures',ignore_errors=False)
	
	#structural heterogeneity
	from matminer.featurizers.structure import StructuralHeterogeneity 
	structural_heterogeneity = StructuralHeterogeneity()
	structural_heterogeneity.set_n_jobs(28)
	labels.append(structural_heterogeneity.feature_labels()) 
	df  = structural_heterogeneity.featurize_dataframe(df, 'structures',ignore_errors=False)
	
	#convert structure to composition
	from matminer.featurizers.conversions import StructureToComposition 
	structures_to_compositions=StructureToComposition()
	labels.append(structures_to_compositions.feature_labels())
	df=structures_to_compositions.featurize_dataframe(df,'structures')
	
	#CohesiveEnergy
	from matminer.featurizers.composition import CohesiveEnergy 
	cohesive_energy = CohesiveEnergy()
	cohesive_energy.set_n_jobs(28)
	labels.append(cohesive_energy.feature_labels())
	df=cohesive_energy.featurize_dataframe(df,'composition',ignore_errors=True)
	
	#ValenceOrbital
	from matminer.featurizers.composition import ValenceOrbital
	valence_orbital = ValenceOrbital()
	valence_orbital.set_n_jobs(28)
	labels.append(valence_orbital.feature_labels())
	df=valence_orbital.featurize_dataframe(df,'composition',ignore_errors=True)
	
	#AtomicOrbital
	from matminer.featurizers.composition import AtomicOrbitals
	atomic_orbitals = AtomicOrbitals()
	atomic_orbitals.set_n_jobs(28)
	labels.append(atomic_orbitals.feature_labels())
	df=atomic_orbitals.featurize_dataframe(df,'composition',ignore_errors=True)
	
	df=df.dropna()
	
	#FunctionFeaturizer
	from matminer.featurizers.function import FunctionFeaturizer
	function_featurizer = FunctionFeaturizer(multi_feature_depth=2,combo_function=np.sum)
	function_featurizer.set_n_jobs(28)
	function_featurizer=function_featurizer.fit(df[labels])
	labels.append(function_featurizer.feature_labels())
	df=function_featurizer.featurize_dataframe(df,labels)
	
	df.to_csv('featurized_dataframe.csv')
	
	labels=list(itertools.chain(*labels))
	remove=['composition','HOMO_character','HOMO_element','LUMO_character','LUMO_element'] 
	for item in remove: 
		labels.remove(item)
	print(labels) 
	
	
	#get s_vs_ep
	ec_list=[]
	for item in centro_elastic_compliance:
		obj=dict(item)
		if obj['elasticity.compliance_tensor'] != None:
			obj=obj['elasticity.compliance_tensor']
			w, v = eig(obj)
			w    = np.average(np.real(w))
		else: 
			w=np.nan
		ec_list.append(w)
	
	dt_list=[]
	for item in centro_dielectric_tensor:
		obj=dict(item)
		if obj['diel.e_total']!= None:
			obj=obj['diel.e_total']
			w, v = eig(obj)
			w    = np.average(np.real(w))
		else: 
			w=np.nan
		dt_list.append(w)
	
	ep_0=8.85 
	s_vs_ep=[]
	for i in range(len(dt_list)):
	    s_vs_ep.append((ec_list[i]/dt_list[i])/ep_0) 
	
	endpoint='ec_list'
	print(len(ec_list))
	df[endpoint]=ec_list
	
df=df.loc[df[endpoint]>0.0]
df=df.loc[df[endpoint]<175.0]
print('len(df)')
print(len(df))
X=df[labels]
Y=df[endpoint]

forest = ExtraTreesRegressor(n_estimators=100,random_state=1)
forest.fit(X,Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


print("Feature ranking:")

ranked_features=[]
for f in range(X.shape[1]):
	if importances[indices[f]] != 0: 
		ranked_features.append(labels[indices[f]])
		print("%d. %s (%f)" % (f + 1, labels[indices[f]], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(ranked_features), ranked_features,
       color="r", yerr=std[indices], align="center")
plt.xticks(range(ranked_features),ranked_features,rotation=45,fontsize=10,fontweight='bold')
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png')

feature_rank = pd.DataFrame()
feature_rank['labels'] = labels[indices]
feature_rank['importances'] = importances[indices]
feature_rank.to_csv('feature_rank.csv')


isolated_forest=IsolationForest(n_estimators=1000,max_samples='auto',behaviour='new',contamination='auto',max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)

isolated_forest.fit(X[labels])
predicted=isolated_forest.predict(X_new[labels]) 
X['anomaly']  = predicted 
X['endpoint'] = Y 
X = X.loc[X['anomaly']==1]
Y = X['endpoint']
X = X.drop(columns='endpoint')

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)
#tuned_parameters = [{'n_estimators':[int(1e1),int(1e2),int(1e3)]}]
tuned_parameters = [{'n_estimators':[500,1000,2000]}]
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

