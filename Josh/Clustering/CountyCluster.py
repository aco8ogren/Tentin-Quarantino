#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import git
import sys
sys.path.append('Josh/Clustering')
from kMeansClustering import Kmeans
import time
import os
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)

File=open('Josh/Processing/Processed Data/GeoDict.pkl','rb')
GeoDict=pickle.load(File)
vals=[val for val in GeoDict.values()]
for i,val in enumerate(vals):
    if (np.isnan(val)).any():
        del vals[i]
vals=np.array(vals)
Output=[]
# ks=np.arange(30,100,5)
ks=np.arange(10,1000,10)
for k in ks:
# for k in [1000]:
    print(k)
    clust,cdist=Kmeans(k,vals)
    Output.append([k,len(np.unique(clust)),cdist])
Output=np.array(Output)
np.savetxt('Josh/Clustering/ClusterDat_refined.txt',Output)


# %%
