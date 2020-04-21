#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import git
import sys
sys.path.append('Josh/Clustering')
from kMeansClustering import Kmeans
import time
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
ks=np.arange(10,50,10)
for k in ks:
    clust,cdist=Kmeans(k,vals)
    Output.append([k,len(np.unique(clust)),cdist])
Output=np.array(Output)
np.savetxt('Josh/Clustering/ClusterDat.txt',Output)


# %%
