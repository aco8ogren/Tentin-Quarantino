#%%
import numpy as np
import pandas as pd
import git
import os
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)


def JoshMeansClustering(fipsList,date,CritCases=50):
    date=pd.to_datetime(date)
    covidDF=pd.read_csv('data/us/covid/nyt_us_counties.csv')
    covidDF['date']=pd.to_datetime(covidDF['date'])
    geoDF=pd.read_csv(r'Josh\Processing\Processed Data\GeoData.csv')
    Ogala={k:v for k,v in zip(['fips','long','lat'],[46102,-102.821318,43.352275])}
    geoDF=geoDF.append(Ogala,ignore_index=True)
    df=pd.merge(covidDF[['fips','deaths','date']],geoDF,how='left',on='fips')

    df=df[df.date==date].drop(columns='date')
    df=df[df.fips.isin(fipsList)].sort_values(by=['long','lat']).reset_index(drop=True)
    df['LongDeaths']=df.deaths.cumsum()

    TargetNumClusters=df.deaths.sum()/CritCases
    LongRes=((TargetNumClusters)**.5)
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
    DFs=[df[df.LongQ==q] for q in np.arange(df.LongQ.max())]
    
    Q=0
    for i,DF in enumerate(DFs):
        # res=LatRes[i]
        DF.sort_values(by=['lat'],inplace=True)
        # print(DF)
        DF['LatDeaths']=DF.deaths.cumsum()
        DF['LatQDeaths']=DF['LatDeaths']
        DF['cluster']=np.nan*np.ones(len(DF))
        boundaries=[0]
       
        while DF.iloc[-1].LatQDeaths>CritCases:
            boundaries.append((DF.LatQDeaths.values>=CritCases).argmax()+1)
            DF.iloc[boundaries[-2]:boundaries[-1],DF.columns=='cluster']=int(Q)
            DF.LatQDeaths-=DF.iloc[boundaries[-1]-1].LatQDeaths
            # print(DF)
            Q+=1
        DF.loc[DF.cluster.isna(),'cluster']=Q-1
        DFs[i]=DF

    ClusterDF=pd.concat(DFs)[['fips','cluster','long','lat','deaths']]
    deaths=ClusterDF[['deaths','cluster']].groupby('cluster').sum().rename(columns={'deaths':'clusterDeaths'})
    ClusterDF=ClusterDF.merge(deaths,left_on='cluster',right_index=True)
    return ClusterDF