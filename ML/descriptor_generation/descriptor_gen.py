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
import itertools 

centrosymmetric_structures=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/centrosymmetric_insulators.npy',allow_pickle=True)
task_ids=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/structs/data/centrosymmetric_task_ids.npy',allow_pickle=True)
centro_elastic_compliance=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/elasticity_compliance/centro_elasticity.npy',allow_pickle=True)
centro_dielectric_tensor=np.load('/Users/dennistrujillo/Dropbox/mp_share_dt_ag/dielectric_total/centro_diel.npy',allow_pickle=True)

print(len(centrosymmetric_structures))
print(len(task_ids))
data = {'structures': centrosymmetric_structures, 'ids' : task_ids }
df = pd.DataFrame(data)

labels=[]
#soap representation
from matminer.featurizers.structure import SOAP 
soap = SOAP()
soap=soap.fit(centrosymmetric_structures)
labels.append(soap.feature_labels())
df = soap.featurize_dataframe(df,'structures')

##orbital field matrix 
#from matminer.featurizers.structure import OrbitalFieldMatrix
#ofm = OrbitalFieldMatrix()
#ofm.set_n_jobs(28)
#ofm.fit(centrosymmetric_structures)
#labels.append(ofm.feature_labels()) 
#df  = ofm.featurize_dataframe(df, 'structures',ignore_errors=False)

##chemical ordering
#from matminer.featurizers.structure import ChemicalOrdering 
#chemical_ordering = ChemicalOrdering()
#chemical_ordering.set_n_jobs(28)
#labels.append(chemical_ordering.feature_labels() 
#df1  = chemical_ordering.featurize_dataframe(df, 'structures',ignore_errors=False)
#df1.to_csv('chemical_ordering.csv')
#
#coulomb 
#from matminer.featurizers.structure import CoulombMatrix
#coulomb = CoulombMatrix()
#coulomb.set_n_jobs(28)
#coulomb.fit(centrosymmetric_structures)
#df  = coulomb.featurize_dataframe(df, 'structures',ignore_errors=False)
#labels.append(coulomb.feature_labels())

##convert structure to composition
#from matminer.featurizers.conversions import StructureToComposition 
#structures_to_compositions=StructureToComposition()
#labels.append(structures_to_compositions.feature_labels())
#df=structures_to_compositions.featurize_dataframe(df,'structures')

##AtomicOrbitals
#from matminer.featurizers.composition import AtomicOrbitals
#atomic_orbitals = AtomicOrbitals()
#atomic_orbitals.set_n_jobs(28)
#labels.append(atomic_orbitals.feature_labels()) 
#df=atomic_orbitals.featurize_dataframe(df,'composition')
#df.to_csv('atomic_orbitals.csv')

##AtomicPackingEfficiency
#from matminer.featurizers.composition import AtomicPackingEfficiency
#atomic_packing_efficiency = AtomicPackingEfficiency()
#atomic_packing_efficiency.set_n_jobs(28)
#labels.append(atomic_packing_efficiency.feature_labels())
#df=atomic_packing_efficiency.featurize_dataframe(df,'composition',ignore_errors=True)
#df.to_csv('atomic_packing_efficiency.csv')
#
##CohesiveEnergy
#from matminer.featurizers.composition import CohesiveEnergy 
#cohesive_energy = CohesiveEnergy()
#cohesive_energy.set_n_jobs(28)
#labels.append(cohesive_energy.feature_labels())
#df=cohesive_energy.featurize_dataframe(df,'composition',ignore_errors=True)
#df.to_csv('cohesive_energy.csv')
#
##ElectronAffinity
#from matminer.featurizers.composition import ElectronAffinity 
#electron_affinity = ElectronAffinity()
#electron_affinity.set_n_jobs(28)
#labels.append(electron_affinity.feature_labels())
#df=electron_affinity.featurize_dataframe(df,'composition',ignore_errors=True)
#df.to_csv('electron_affinity.csv')

##ElectronegativityDiff
#from matminer.featurizers.composition import ElectronegativityDiff
#electronegativity_diff = ElectronegativityDiff()
#electronegativity_diff.set_n_jobs(28)
#labels.append(electronegativity_diff.feature_labels())
#df=electronegativity_diff.featurize_dataframe(df,'composition',ignore_errors=True)
#df.to_csv('electronegativity_diff.csv')

##OxidationStates
#from matminer.featurizers.composition import OxidationStates 
#oxidation_states = OxidationStates()
#oxidation_states.set_n_jobs(28)
#labels.append(oxidation_states.feature_labels())
#df=oxidation_states.featurize_dataframe(df,'composition',ignore_errors=True)
#df.to_csv('oxidation_states.csv')
#
##ValenceOrbital
#from matminer.featurizers.composition import ValenceOrbital
#valence_orbital = ValenceOrbital()
#valence_orbital.set_n_jobs(28)
#labels.append(valence_orbital.feature_labels())
#df=valence_orbital.featurize_dataframe(df,'composition',ignore_errors=True)
#df.to_csv('valence_orbitals.csv')


labels=list(itertools.chain(*labels))
#remove=['composition','HOMO_character','HOMO_element','LUMO_character','LUMO_element'] 
#for item in remove: 
#	labels.remove(item)
print(labels) 



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

endpoint='ec_list'
print(len(ec_list))
df[endpoint]=ec_list

df=df.dropna()
df=df.loc[df[endpoint]>0.0]
print('len(df)')
print(len(df))
X=df[labels]
Y=df[endpoint]

# make training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1)
#tuned_parameters = [{'n_estimators':[int(1e1),int(1e2),int(1e3)]}]
tuned_parameters = [{'n_estimators':[50,100,200,500,1000]}]
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

switch = False 
if switch == True: 
	#holder=[]
	#dfdumb=pd.DataFrame()
	#i=0
	#for item in df_clean['orbital field matrix']:
	#    name = '%s' %ids[i]
	#    for obj in item:
	#        for thing in obj:
	#            holder.append(thing)
	#    dfdumb[name]=pd.Series(holder)
	#    holder=[]
	#    i+=1 
	#dfdumb.to_csv("ofm.csv",mode='w')
	
	#coulomb 
	from matminer.featurizers.structure import CoulombMatrix
	coulomb = CoulombMatrix()
	coulomb.set_n_jobs(28)
	coulomb.fit(centrosymmetric_structures)
	df  = coulomb.featurize_dataframe(df, 'structures',ignore_errors=False)
	labels=coulomb.feature_labels()
	df2  = chemical_ordering.featurize_dataframe(df, 'structures',ignore_errors=False)
	df2.to_csv('coulomb_representation.csv')
	#df_clean=pd.DataFrame(columns=[labels])
	#df_clean=df[labels] 
	#df_clean=df_clean.dropna() 
	
	#holder=[]
	#dfdumb=pd.DataFrame()
	#i=0
	#for item in df_clean['coulomb matrix']:
	#    name = '%s' %ids[i]
	#    for obj in item:
	#        for thing in obj:
	#            holder.append(thing)
	#    dfdumb[name]=pd.Series(holder)
	#    holder=[]
	#    i+=1 
	#dfdumb.to_csv("coulomb.csv",mode='w')
	
	#sine coulomb 
	from matminer.featurizers.structure import SineCoulombMatrix
	sine_coulomb = SineCoulombMatrix()
	sine_coulomb.set_n_jobs(28)
	sine_coulomb.fit(centrosymmetric_structures)
	labels=sine_coulomb.feature_labels()
	df  = sine_coulomb.featurize_dataframe(df, 'structures',ignore_errors=False)
	#df_clean=pd.DataFrame(columns=[labels])
	#df_clean=df[labels] 
	#df_clean=df_clean.dropna() 
	#
	#holder=[]
	#dfdumb=pd.DataFrame()
	#i=0
	#for item in df_clean['sine coulomb matrix']:
	#    name = '%s' %ids[i]
	#    for obj in item:
	#        for thing in obj:
	#            holder.append(thing)
	#    dfdumb[name]=pd.Series(holder)
	#    holder=[]
	#    i+=1 
	#dfdumb.to_csv("sine_coulomb.csv",mode='w')
	
	#bag of bonds 
	#from matminer.featurizers.structure import BagofBonds 
	#bag_of_bonds = BagofBonds()
	#bag_of_bonds.set_n_jobs(28)
	#bag_of_bonds=bag_of_bonds.fit(centrosymmetric_structures)
	#labels=bag_of_bonds.feature_labels() 
	#df  = bag_of_bonds.featurize_dataframe(df, 'structures',ignore_errors=False)
	#df_clean=pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('bag_of_bonds.csv',mode='w')
	
	#bond fraction
	from matminer.featurizers.structure import BondFractions 
	bond_fraction = BondFractions()
	bond_fraction.set_n_jobs(28)
	bond_fraction=bond_fraction.fit(centrosymmetric_structures)
	labels=bond_fraction.feature_labels() 
	df  = bond_fraction.featurize_dataframe(df, 'structures',ignore_errors=False)
	#df_clean=pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('bond_fraction.csv',mode='w')
	
	#chemical ordering
	from matminer.featurizers.structure import ChemicalOrdering 
	chemical_ordering = ChemicalOrdering()
	chemical_ordering.set_n_jobs(28)
	labels=chemical_ordering.feature_labels() 
	df  = chemical_ordering.featurize_dataframe(df, 'structures',ignore_errors=False)
	#df_clean=pd.DataFrame(columns=labels)
	#df_clean=df[labels]#.dropna() 
	#df_clean.to_csv('chemical_ordering.csv',mode='w')
	
	#structural heterogeneity
	from matminer.featurizers.structure import StructuralHeterogeneity 
	structural_heterogeneity = StructuralHeterogeneity()
	structural_heterogeneity.set_n_jobs(28)
	labels=structural_heterogeneity.feature_labels() 
	df  = structural_heterogeneity.featurize_dataframe(df, 'structures',ignore_errors=False)
	#df_clean=pd.DataFrame(columns=labels)
	#df_clean=df[labels]#.dropna() 
	#df_clean.to_csv('structural_heterogeneity.csv',mode='w')
	
	#convert structure to composition
	from matminer.featurizers.conversions import StructureToComposition 
	structures_to_compositions=StructureToComposition()
	labels=structures_to_compositions.feature_labels()
	df=structures_to_compositions.featurize_dataframe(df,'structures')
	#df_clean = pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('compositions.csv')
	
	#AtomicOrbitals
	from matminer.featurizers.composition import AtomicOrbitals
	atomic_orbitals = AtomicOrbitals()
	atomic_orbitals.set_n_jobs(28)
	labels=atomic_orbitals.feature_labels() 
	df=atomic_orbitals.featurize_dataframe(df,'composition')
	#df_clean = pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('atomic_orbitals.csv')
	
	#AtomicPackingEfficiency
	from matminer.featurizers.composition import AtomicPackingEfficiency
	atomic_packing_efficiency = AtomicPackingEfficiency()
	atomic_packing_efficiency.set_n_jobs(28)
	labels=atomic_packing_efficiency.feature_labels()
	df=atomic_packing_efficiency.featurize_dataframe(df,'composition',ignore_errors=False)
	#df_clean = pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('atomic_packing_efficiency.csv')
	
	#CohesiveEnergy
	from matminer.featurizers.composition import CohesiveEnergy 
	cohesive_energy = CohesiveEnergy()
	cohesive_energy.set_n_jobs(28)
	labels=cohesive_energy.feature_labels()
	df=cohesive_energy.featurize_dataframe(df,'composition',ignore_errors=False)
	#df_clean = pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('cohesive_energy.csv')
	
	#ElectronAffinity
	from matminer.featurizers.composition import ElectronAffinity 
	electron_affinity = ElectronAffinity()
	electron_affinity.set_n_jobs(28)
	labels=electron_affinity.feature_labels()
	df=electron_affinity.featurize_dataframe(df,'composition',ignore_errors=False)
	#df_clean = pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('electron_affinity.csv')
	
	#ElectronegativityDiff
	from matminer.featurizers.composition import ElectronegativityDiff
	electronegativity_diff = ElectronegativityDiff()
	electronegativity_diff.set_n_jobs(28)
	labels=electronegativity_diff.feature_labels()
	df=electronegativity_diff.featurize_dataframe(df,'composition',ignore_errors=False)
	#df_clean = pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('electronegativity_diff.csv')
	
	#OxidationStates
	from matminer.featurizers.composition import OxidationStates 
	oxidation_states = OxidationStates()
	oxidation_states.set_n_jobs(28)
	labels=oxidation_states.feature_labels()
	df=oxidation_states.featurize_dataframe(df,'composition',ignore_errors=False)
	#df_clean = pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('oxidation_states.csv')
	
	#ValenceOrbital
	from matminer.featurizers.composition import ValenceOrbital
	valence_orbital = ValenceOrbital()
	valence_orbital.set_n_jobs(28)
	labels=valence_orbital.feature_labels()
	df=valence_orbital.featurize_dataframe(df,'composition',ignore_errors=False)
	#df_clean = pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('valence_orbital.csv')
	
	#DOSFeaturizer
	from matminer.featurizers.dos import DOSFeaturizer
	DOSFeaturizer = DOSFeaturizer()
	DOSFeaturizer.set_n_jobs(28)
	labels=DOSFeaturizer.feature_labels()
	df=DOSFeaturizer.featurize_dataframe(df,'composition',ignore_errors=False)
	#df_clean = pd.DataFrame(columns=labels)
	#df_clean=df[labels]
	#df_clean.to_csv('valence_orbital.csv')

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
	
	endpoint='ec_list'
	df[endpoint]=ec_list
	
	#df=df.dropna()
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
