#%%
import os
import pickle
from datetime import datetime
import git
import numpy as np
import pandas as pd

repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)

from Josh.Clustering.kMeansClustering import Kmeans

#%%
# FileRef=pd.DataFrame({'file':[],'fips':[]})
# FileRef.to_csv('Josh/CountyClustering/FileReference.csv',index=False)

# df=pd.read_csv('data/us/covid/nyt_us_counties.csv')
# fipsList=df.fips.unique()
# fipsList=fipsList[~np.isnan(fipsList)]

#%%
def CountyCluster(fipsList):
    MaxK=int(np.floor(len(fipsList))/5+5)
    fipsList=np.array(fipsList)
    fipsList.sort()
    fipstr=';'.join(['%i'%i for i in fipsList])
    # fipstr=''.join(np.array(fipsList).sort().astype(int).astype(str))
    fileRef=pd.read_csv('Josh/CountyClustering/FileReference.csv')
    if fipstr in fileRef.fips.values:
        fname=fileRef.file[fileRef.fips==fipstr].values[0]
        print('Clustering found for given fips list.\nPath: {}'.format(fname))
        return fname
    else:


        File=open('Josh/Processing/Processed Data/GeoDict.pkl','rb')
        GeoDict=pickle.load(File)
        GeoDict[46102]=np.array([-102.821318,43.352275])
        # GeoDict
        DF=pd.DataFrame({'fips':fipsList.astype(int),'long':[GeoDict[fips][0] for fips in fipsList],'lat':[GeoDict[fips][1] for fips in fipsList]}).sort_values(by='fips')



        # vals=[val for val in GeoDict.values()]
        # for i,val in enumerate(vals):
        #     if (np.isnan(val)).any():
        #         del vals[i]
        # vals=np.array(vals)
        vals=DF[['long','lat']].values

        #%%
        ks=np.arange(10,MaxK+5,5)
        Data=[]
        # ks=[10,20,30]

        for k in ks:
            print(k)
            clust,cdist=Kmeans(k,vals)
            Data.append([k,len(np.unique(clust)),cdist])
        Data=np.array(Data)


        p0=Data[0]
        Data=Data-p0
        p1=Data[-1]
        p=(p1/np.linalg.norm(p1))[np.newaxis].T
        DistV=Data-(Data@p)*p.T
        Dist=np.linalg.norm(DistV,axis=1)
        OptInd=Dist.argmax()
        kOpt=ks[OptInd]
        clust,cdist=Kmeans(kOpt,vals)
        ClusterDf=pd.DataFrame(data={'fips':DF.fips,'cluster':clust})


        fname='Josh/CountyClustering/ClusterFiles/{}.csv'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        with open('Josh/CountyClustering/FileReference.csv','a') as file:
            file.write('{},"{}"\n'.format(fname,fipstr))
        ClusterDf.to_csv('{}'.format(fname),index=False)



        # ('Josh/CountyClustering/Cluster Sets/fname.csv',Output)


        # %%
        import matplotlib.pyplot as plt
        DF=DF.merge(ClusterDf,how='left', on='fips')
        cols=['r','b','g','k','y','c','m']
        colors=np.array([None]*len(DF))
        for i in range(kOpt):
            colors[DF['cluster'].values==i]=cols[i%len(cols)]
        plt.figure()
        plt.scatter(DF['long'],DF['lat'],color=colors)
        plt.title('K-Means Clustering:\nUS Counties')
        plt.xlabel('Longitude [°]')
        plt.ylabel('Latitude [°]')
        plt.show
        print('Clustering complete, saved to file.\nPath: {}'.format(fname))
        return fname