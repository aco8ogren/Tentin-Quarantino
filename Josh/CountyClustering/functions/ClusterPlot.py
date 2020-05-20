import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(df,fname=None,title='Clustering'):
        cols=['r','b','g','k','y','c','m']
        colors=np.array([None]*len(df))
        k=len(df.cluster.unique())
        df.loc[:,'cluster']=df.cluster.astype(int)
        for i in np.sort(df.cluster.unique()):
            colors[df['cluster'].values==i]=cols[i%len(cols)]
        plt.figure()
        plt.scatter(df['long'],df['lat'],color=colors)
        plt.title('%s: %i Clusters\nDeaths per Cluster ∈ [%i, %i]'%(title,k,df[df.long>-128].clusterDeaths.min(),df.clusterDeaths.max()))
        plt.xlabel('Longitude [°]')
        plt.ylabel('Latitude [°]')
        if fname==None:
            plt.show()
            return
        plt.savefig(fname)