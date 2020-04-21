#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import git
from kMeansClustering import Kmeans
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)

File=open('Josh/Processing/Processed Data/GeoDict.pkl','rb')
GeoDict=pickle.load(File)

# %%
vals=[val for val in GeoDict.values()]
for i,val in enumerate(vals):
    if (np.isnan(val)).any():
        del vals[i]
vals=np.array(vals)
clust=Kmeans(100,vals)

# %%

