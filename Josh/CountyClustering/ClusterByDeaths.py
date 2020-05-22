#%%
import numpy as np
import pandas as pd
import git
import os
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)


def JoshMeansClustering(fipsList,date,d_thres=50,cluster_radius=1e6,Fpath=None):
    covidDF=pd.read_csv('data/us/covid/nyt_us_counties.csv')
    covidDF['date']=pd.to_datetime(covidDF['date'])
    if date is not None:
        date=pd.to_datetime(date)#-np.timedelta64(1,'D')
    else:
        date=covidDF.date.max()
    
    # Get df that relates fips to lat/long
    geoDF=pd.read_csv(r'Josh\Processing\Processed Data\GeoData.csv')
    # Add missing Ogala entry
    Ogala={k:v for k,v in zip(['fips','long','lat'],[46102,-102.821318,43.352275])}
    geoDF=geoDF.append(Ogala,ignore_index=True)
    # Add lat/long to main df
    df=pd.merge(covidDF[['fips','deaths','date']],geoDF,how='left',on='fips')
    
    # take only df rows for corresponding to the cluster date (=train_til in main code)
    df=df[df.date==date].drop(columns='date')
    # take only rows for fips in the fips to cluster list
    df=df[df.fips.isin(fipsList)]
    
    # Seperate Alaska and Hawaii, as it does not make sense to cluster them geographically with continental counties
        # give them their own seperate clusters as they are usually below D_THRES
    AL_HA=df[df.long<-128]
    df=df[df.long>=-128]
    AL_HA['cluster']=np.nan*np.ones(len(AL_HA))
    AL_HA.loc[AL_HA.lat>40,'cluster']=-1
    AL_HA.loc[AL_HA.lat<=40,'cluster']=-2

    # Sort by long than lat and take the cumulative deaths along longitude
    df=df.sort_values(by=['long','lat']).reset_index(drop=True)
    df['LongDeaths']=df.deaths.cumsum()

    # estimate number of clusters
    TargetNumClusters=df.deaths.sum()/d_thres
    # estimate number of columnar clusters
    LongRes=((TargetNumClusters)**.5)
    # threshold for deaths per columnar cluster
        # i.e. the number of deaths each columnar cluster is guaranteed to match or exceed
    critLong=np.floor(df.deaths.sum()/LongRes)
    df['LongQDeaths']=df['LongDeaths']
    df['LongQ']=np.nan*np.ones(len(df))
    LongQ=0
    boundaries=[0]
    while df.iloc[-1].LongQDeaths>critLong:
        boundaries.append((df.LongQDeaths.values>=critLong).argmax()+1)
        df.iloc[boundaries[-2]:boundaries[-1],df.columns=='LongQ']=int(LongQ)
        df.LongQDeaths-=df.iloc[boundaries[-1]-1].LongQDeaths
        # print(df)
        LongQ+=1
    df.loc[df.LongQ.isna(),'LongQ']=LongQ-1
    DFs=[df[df.LongQ==q] for q in np.sort(df.LongQ.unique())]
    
    Q=0
    for i,DF in enumerate(DFs):
        # res=LatRes[i]
        DF= DF.sort_values(by=['lat'])
        # print(DF)
        DF.loc[:,'LatDeaths']=DF.deaths.cumsum()
        DF.loc[:,'LatQDeaths']=DF['LatDeaths']
        DF.loc[:,'cluster']=np.nan*np.ones(len(DF))
        boundaries=[0]
       
        while DF.iloc[-1].LatQDeaths>d_thres:
            boundaries.append((DF.LatQDeaths.values>=d_thres).argmax()+1)
            DF.iloc[boundaries[-2]:boundaries[-1],DF.columns=='cluster']=int(Q)
            DF.LatQDeaths-=DF.iloc[boundaries[-1]-1].LatQDeaths
            # print(DF)
            Q+=1
        DF.loc[DF.cluster.isna(),'cluster']=Q-1
        DFs[i]=DF

    ClusterDF=pd.concat(DFs)[['fips','cluster','long','lat','deaths']]
    AL_HA.loc[AL_HA.cluster==-1,'cluster']=ClusterDF.cluster.max()+1
    AL_HA.loc[AL_HA.cluster==-2,'cluster']=ClusterDF.cluster.max()+2
    ClusterDF=pd.concat((ClusterDF,AL_HA))[['fips','cluster','long','lat','deaths']]
    deaths=ClusterDF[['deaths','cluster']].groupby('cluster').sum().rename(columns={'deaths':'clusterDeaths'})
    ClusterDF=ClusterDF.merge(deaths,left_on='cluster',right_index=True)
    ClusterDF['state']=['ClusterState']*len(ClusterDF)
    ClusterDF['cluster']=ClusterDF['cluster'].astype(int)

    ClusterDF.loc[:,'cluster'] += 2
    ClusterDF['county']=['cluster%i'%i for i in ClusterDF.cluster.values]
    ClusterDF['centerWeights']=ClusterDF.deaths/ClusterDF.clusterDeaths
    ClusterDF['weightedLat']=ClusterDF.lat*ClusterDF.centerWeights
    ClusterDF['weightedLong']=ClusterDF.long*ClusterDF.centerWeights
    ClusterDF['clusterLat']=ClusterDF.groupby('cluster')['weightedLat'].transform(np.sum)
    ClusterDF['clusterLong']=ClusterDF.groupby('cluster')['weightedLong'].transform(np.sum)
    ClusterDF['distance']=((ClusterDF.clusterLat-ClusterDF.lat)**2+(ClusterDF.clusterLong-ClusterDF.long)**2)**(.5)
    fips_to_erf=np.sort(ClusterDF[ClusterDF.distance>cluster_radius].fips.unique())
    ClusterDF=ClusterDF[ClusterDF.distance<=cluster_radius]
    ClusterDF.drop(columns=['centerWeights','weightedLat','weightedLong'],inplace=True)


    if Fpath is not None:
        ClusterDF.to_csv(Fpath)
    return ClusterDF, fips_to_erf