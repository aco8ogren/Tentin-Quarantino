import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(df,fname=None):
        cols=['r','b','g','k','y','c','m']
        colors=np.array([None]*len(df))
        k=len(df.cluster.unique())
        for i in range(k):
            colors[df['cluster'].values==i]=cols[i%len(cols)]
        plt.figure()
        plt.scatter(df['long'],df['lat'],color=colors)
        plt.title('Josh Means Clustering:%i Clusters\nDeaths per Cluster ∈ [%i, %i]'%(k,df[df.long>-128].clusterDeaths.min(),df.clusterDeaths.max()))
        plt.xlabel('Longitude [°]')
        plt.ylabel('Latitude [°]')
        if fname==None:
            plt.show()
            return
        plt.savefig(fname)