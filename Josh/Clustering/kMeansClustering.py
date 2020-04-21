def Kmeans(K,X):
    # K means algorithm for K clusters.
    # X is input data np array of shape (n,p)
    #   where n is number of data points and p is number of dimensions
    import numpy as np 
    import matplotlib.pyplot as plt
    (n,p)=X.shape
    mins=X.min(0)
    maxs=X.max(0)
    Clusters=[]
    for i in range(len(mins)):
        Clusters.append(np.random.uniform(size=(K),low=mins[i],high=maxs[i]))
    Clusters=np.array(Clusters).T
    ClusterAssignments=np.zeros(len(X))
    Continue=True
    while Continue:
        oldClusters=ClusterAssignments.copy()
        ClusterStack=np.zeros((X.shape+(K,)))
        for i in range(K):
            ClusterStack[:,:,i]=np.tile(Clusters[[i],:],(len(X),1))
        XStack=np.tile(X.reshape(X.shape+(1,)),K)
        ClusterAssignments=np.linalg.norm((XStack-ClusterStack),axis=1).argmin(1)
        ClustDist=np.linalg.norm((XStack-ClusterStack),axis=1).min(1).mean()
        if np.array_equal(oldClusters,ClusterAssignments):
            Continue=False
            break
        # cols=['r','b','g','k','y','c','m']
        # colors=np.array([None]*len(X))
        # for i in range(K):
        #     colors[ClusterAssignments==i]=cols[i%len(cols)]
        # plt.figure
        # plt.scatter(X[:,0],X[:,1],color=colors)
        # plt.show()
        oldC=Clusters.copy()
        for i in range(K):
            Clusters[i,:]=X[ClusterAssignments==i,:].mean(0)
            Clusters[np.isnan(Clusters)]=oldC[np.isnan(Clusters)]
    
    return ClusterAssignments,ClustDist


