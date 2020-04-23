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
import pandas as pd
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)

File=open('Josh/Processing/Processed Data/GeoDict.pkl','rb')
GeoDict=pickle.load(File)
del GeoDict[0]
# vals=[val for val in GeoDict.values()]
# for i,val in enumerate(vals):
    # if (np.isnan(val)).any():
#         del vals[i]
GeoVals=np.array([[key]+[vall for vall in val] for key,val in GeoDict.items()])
vals=GeoVals[:,1:]
#%%

GeoDf=pd.DataFrame(GeoVals,columns=['fips','long','lat'])

K=75
clusters=Kmeans(K,vals)[0]
ClusterDf=pd.DataFrame(data={'fips':GeoVals[:,0],'cluster':clusters})
DF=GeoDf.merge(ClusterDf,how='left', on='fips')

        
cols=['r','b','g','k','y','c','m']
colors=np.array([None]*len(DF))
for i in range(K):
    colors[DF['cluster'].values==i]=cols[i%len(cols)]
plt.figure()
plt.scatter(DF['long'],DF['lat'],color=colors)
plt.title('K-Means Clustering:\nUS Counties')
plt.xlabel('Longitude [°]')
plt.ylabel('Latitude [°]')
plt.show()
# plt.savefig('ClusterResults.png')
# DF.to_csv('Josh/Clustering/FinalClusters.csv',index=False)


# %%
