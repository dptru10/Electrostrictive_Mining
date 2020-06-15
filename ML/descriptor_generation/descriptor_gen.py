import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  
from numpy.linalg import eig
from matplotlib import colors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd 
import itertools 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--read_dataframe",action="store_true")
parser.add_argument("--write_dataframe",action="store_true")
parser.add_argument("--mercurial",action="store_true")
parser.add_argument("--dielectric",action="store_true")
parser.add_argument("--compliance",action="store_true")
parser.add_argument("--functionalize",action="store_true")
parser.add_argument("--importance",action="store_true")
parser.add_argument("--outlier_removal",action="store_true")
parser.add_argument("--ml",action="store_true")
args = parser.parse_args()

if args.mercurial is True: 
    centrosymmetric_structures=np.load('/home/dennis/Dropbox/mp_share_dt_ag/structs/centrosymmetric_insulators.npy',allow_pickle=True)
    task_ids=np.load('/home/dennis/Dropbox/mp_share_dt_ag/structs/data/centrosymmetric_task_ids.npy',allow_pickle=True)
    centro_elastic_compliance=np.load('/home/dennis/Dropbox/mp_share_dt_ag/elasticity_compliance/centro_elasticity.npy',allow_pickle=True)
    centro_dielectric_tensor=np.load('/home/dennis/Dropbox/mp_share_dt_ag/dielectric_total/centro_diel.npy',allow_pickle=True)

else: 
    centrosymmetric_structures=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/centrosymmetric_insulators.npy',allow_pickle=True)
    task_ids=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/data/centrosymmetric_task_ids.npy',allow_pickle=True)
    centro_elastic_compliance=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/elasticity_compliance/centro_elasticity.npy',allow_pickle=True)
    centro_dielectric_tensor=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/dielectric_total/centro_diel.npy',allow_pickle=True)

if args.read_dataframe is True:
    labels=[]
    df = pd.read_pickle('featurized_dataframe.pkl')
    df=df.drop(['structures','ids','composition','composition_oxid','HOMO_character','HOMO_element','LUMO_character','LUMO_element'],axis=1)
    if args.compliance is True: 
        df=df.drop('dt_list',axis=1)
    labels = list(df.columns) 

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
    
    #partial radial distribution function 
    from matminer.featurizers.structure import PartialRadialDistributionFunction
    partial_rdf = PartialRadialDistributionFunction()
    partial_rdf.set_n_jobs(28)
    partial_rdf.fit(centrosymmetric_structures)
    labels.append(partial_rdf.feature_labels()) 
    df  = partial_rdf.featurize_dataframe(df, 'structures',ignore_errors=False)
    
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
    df=structures_to_compositions.featurize_dataframe(df,'structures')
    
    #convert composition to oxidcomposition
    from matminer.featurizers.conversions import CompositionToOxidComposition 
    OxidCompositions=CompositionToOxidComposition()
    print(OxidCompositions.feature_labels())
    df=OxidCompositions.featurize_dataframe(df,'composition')

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
    
    #ElectronegativityDiff
    from matminer.featurizers.composition import ElectronegativityDiff 
    electronegativity_diff = ElectronegativityDiff()
    electronegativity_diff.set_n_jobs(28)
    labels.append(electronegativity_diff.feature_labels())
    df=electronegativity_diff.featurize_dataframe(df,'composition_oxid',ignore_errors=True)

    #ElectronAffinity
    from matminer.featurizers.composition import ElectronAffinity
    electron_affinity = ElectronAffinity()
    electron_affinity.set_n_jobs(28)
    labels.append(electron_affinity.feature_labels())
    df=electron_affinity.featurize_dataframe(df,'composition_oxid',ignore_errors=True)

if args.functionalize is True: 
    #FunctionFeaturizer
    new_labels=[]
    from matminer.featurizers.function import FunctionFeaturizer
    function_featurizer = FunctionFeaturizer(multi_feature_depth=2,expressions=["1/x","x**2"],combo_function=np.sum)
    function_featurizer.set_n_jobs(28)
    function_featurizer=function_featurizer.fit(df[labels])
    df=function_featurizer.featurize_dataframe(df[labels],labels)
    ff_labels=function_featurizer.feature_labels()
    new_labels=ff_labels
    labels.append(new_labels)
    
    
#labels=list(itertools.chain(*labels))
#remove=['HOMO_character','HOMO_element','LUMO_character','LUMO_element'] 
#for item in remove:
#	print(item) 
#	labels.remove(item)
#print(labels) 
	
	
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

if args.compliance is True: 
    endpoint='Kv'
    Y=Kv

if args.dielectric is True: 
    endpoint='dt_list'
    Y=dt_list

if args.write_dataframe is True: 
    df.to_pickle('featurized_dataframe.pkl')

df[endpoint] = Y
if args.read_dataframe is False: 
    df.to_pickle('featurized_dataframe.pkl')
df=df.replace([np.inf, -np.inf], np.nan)
df=df.dropna()
df=df.loc[df[endpoint]>0.0]
df=df.loc[df[endpoint]<3e2]


Y = df[endpoint]
df1=df.drop(endpoint,axis=1)
X = df1


if args.importance is True: 
	forest = ExtraTreesRegressor(n_estimators=1000,random_state=1)
	forest.fit(X,Y)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
	             axis=0)
	indices = np.argsort(importances)[::-1]
	
	
	print(len(indices))
	print(len(labels))
	print(len(importances))
	print("Feature ranking:")
	
	ranked_features=[]
	for f in range(X.shape[1]):
		print("%d. %s (%f)" % (f + 1, labels[indices[f]], importances[indices[f]]))

#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(len(ranked_features)), ranked_features,
#       color="r", align="center")
#plt.xticks(range(len(ranked_features)),ranked_features,rotation=45,fontsize=10,fontweight='bold')
#plt.xlim([-1, len(ranked_features)])
#plt.tight_layout()
#plt.savefig('random_forest_feature_importance.png')

#feature_rank = pd.DataFrame()
#feature_rank['labels'] = pd.Series(labels[indices])
#feature_rank['importances'] = pd.Series(importances[indices]) 
#feature_rank.to_csv('feature_rank.csv')

if args.outlier_removal is True: 
	isolated_forest=IsolationForest(n_estimators=2000,max_samples='auto',behaviour='new',contamination='auto',max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
	
	isolated_forest.fit(X)
	predicted=isolated_forest.predict(X) 
	X['anomaly']  = predicted 
	X['endpoint'] = Y
	print(X['anomaly'].value_counts())
	X = X.loc[X['anomaly']==1]
	Y = X['endpoint']
	X = X.drop(columns='endpoint')

if args.ml is True: 
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
        
	if args.dielectric is True: 
		if args.outlier_removal is True:
			f=open('hyperpameters_outlier_removal_dielectric.txt',mode='w')
		else: 
			f=open('hyperpameters_dielectric.txt',mode='w')
		f.write("Best parameters set found on development set:")
		f.write('\n\n')
		f.write(str(forest.best_params_))
		f.write('\n\n')
		f.write('Score:')
		f.write(str(-forest.best_score_))
		f.write('\n\n')
		f.write('Train:\nR2:%.3f \nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f\nTest:\nR2:%.3f\nMSE:%.3f\nRMSE:%.3f\nMAE:%.3f' %(r2_score_train,mse_score_train,rmse_score_train,mae_score_train,r2_score_test,mse_score_test,rmse_score_test,mae_score_test)) 
		f.close() 
    
	if args.compliance is True: 
		if args.outlier_removal is True: 
			f=open('hyperpameters_outlier_removal_compliance.txt',mode='w')
		else: 
			f=open('hyperpameters_compliance.txt',mode='w')
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
	#plt.gca().set_aspect('equal', adjustable='box')
	plt.axis([np.min(y_train),np.max(y_train),np.min(y_train),np.max(y_train)])
	plt.colorbar() 
	if args.dielectric is True: 
		plt.xlabel('Reported $\epsilon$')
		plt.ylabel('Predicted $\epsilon$')
		plt.tight_layout()
		if args.outlier_removal is True: 
			plt.savefig('forest_histogram_dielectric_outlier_removal_train.png')
		else: 
			plt.savefig('forest_histogram_dielectric_train.png')
	if args.compliance is True: 
		plt.xlabel('Reported $K_{v}$')
		plt.ylabel('Predicted $K_{v}$')
		plt.tight_layout()
		if args.outlier_removal is True: 
			plt.savefig('forest_histogram_compliance_outlier_removal_train.png')
		else: 
			plt.savefig('forest_histogram_compliance_train.png')
	plt.figure()
	plt.title('Histogram forest Test')
	plt.hist2d(x=y_test,y=model_test,bins=100,norm=colors.LogNorm())
	#plt.gca().set_aspect('equal', adjustable='box')
	plt.axis([np.min(y_test),np.max(y_test),np.min(y_test),np.max(y_test)])
	plt.colorbar() 
	if args.dielectric is True: 
		plt.xlabel('Reported $\epsilon$')
		plt.ylabel('Predicted $\epsilon$')
		plt.tight_layout()
		if args.outlier_removal is True: 
			plt.savefig('forest_histogram_dielectric_outlier_removal_test.png')
		else: 
			plt.savefig('forest_histogram_dielectric_test.png')
	if args.compliance is True: 
		plt.xlabel('Reported $K_{v}$')
		plt.ylabel('Predicted $K_{v}$')
		plt.tight_layout()
		if args.outlier_removal is True: 
			plt.savefig('forest_histogram_compliance_outlier_removal_test.png')
		else: 
			plt.savefig('forest_histogram_compliance_test.png')
