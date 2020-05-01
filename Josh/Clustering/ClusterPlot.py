#%%
import git
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)
Clusters=pd.read_csv('Josh/Clustering/FinalClusters.csv')
K=75
cols=['r','b','g','k','y','c','m']
colors=np.array([None]*len(Clusters))
for i in range(K):
    colors[Clusters.cluster==i]=cols[i%len(cols)]
plt.figure
plt.scatter(Clusters.long,Clusters.lat,color=colors)
plt.xlabel('Longitute [°]')
plt.ylabel('Latitude [°]')
plt.title('K-Means Clustering Results:\nK = 75')
plt.savefig('Josh/Clustering/FinalClusters.png')
plt.savefig('Josh/Clustering/FinalClusters.svg')
# %%