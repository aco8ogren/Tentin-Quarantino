#%%
import numpy as np
import pandas as pd
import git
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)

covidDF=pd.read_csv('data/us/covid/nyt_us_counties.csv')
covidDF=covidDF[~covidDF.fips.isna()]
#------------------------------------------------------------------------------------------
# Setting function Args
date=covidDF.date.max()
CDF=covidDF[covidDF.date==date]
fipsList=CDF[CDF.deaths<50].fips.values
CritCases=50
#------------------------------------------------------------------------------------------

geoDF=pd.read_csv('Josh\Processing\Processed Data\GeoData.csv')
Ogala={k:v for k,v in zip(['fips','long','lat'],[46102,-102.821318,43.352275])}
geoDF=geoDF.append(Ogala,ignore_index=True)


df=pd.merge(covidDF[['fips','deaths','date']],geoDF,how='left',on='fips')

df=df[df.date==date].drop(columns='date')
df=df[df.fips.isin(fipsList)].sort_values(by=['long','lat']).reset_index(drop=True)
df['LongDeaths']=df.deaths.cumsum()

#%%
# DF.sort_values(by=['lat'],inplace=True)
# print(DF)
df['LongQDeaths']=df['LongDeaths']
df['LongQ']=np.nan*np.ones(len(df))
LongQ=0
boundaries=[0]
critLong=
while df.iloc[-1].LongQDeaths>CritCases:
        boundaries.append((df.LongQDeaths.values>=CritCases).argmax()+1)
        df.iloc[boundaries[-2]:boundaries[-1],df.columns=='LongQ']=int(Q)
        df.LongQDeaths-=df.iloc[boundaries[-1]-1].LongQDeaths
        # print(df)
        LongQ+=1
    df.loc[df.LongQ.isna(),'LongQ']=LongQ-1
#%%

# TargetNumClusters=df.deaths.sum()/CritCases
# LongRes=np.floor((TargetNumClusters)**.5)
# numClusters=LongRes**2
# # quantiles=np.linspace(1/LongRes,1,int(LongRes))
# df['LongQ']=pd.qcut(df.LongDeaths,int(LongRes),labels=False)

# resLat=(df[['LongQ','deaths']].groupby('LongQ').sum()/50).rename(columns={'deaths':'numRows'})
# resLat.numRows=[np.floor(i) for i in resLat.numRows]
# LatRes=resLat.to_dict()['numRows']
# quantDFs=[df[df.LongQ==q] for q in range(df.LongQ.max())]
Q=0
for i,DF in enumerate(quantDFs):
    # res=LatRes[i]
    DF.sort_values(by=['lat'],inplace=True)
    # print(DF)
    DF['LatDeaths']=DF.deaths.cumsum()
    DF['LatQDeaths']=DF['LatDeaths']
    DF['LatQ']=np.nan*np.ones(len(DF))
    boundaries=[0]
#%%
    while DF.iloc[-1].LatQDeaths>CritCases:
        boundaries.append((DF.LatQDeaths.values>=CritCases).argmax()+1)
        DF.iloc[boundaries[-2]:boundaries[-1],DF.columns=='LatQ']=int(Q)
        DF.LatQDeaths-=DF.iloc[boundaries[-1]-1].LatQDeaths
        # print(DF)
        Q+=1
    DF.loc[DF.LatQ.isna(),'LatQ']=Q-1
        

    # DF['LatQ']=pd.qcut(DF.LatDeaths,int(res),labels=False)

    quantDFs[i]=DF




df.sort_values(by=['LongQ','lat'])
#%%
# cumLatDF=df[['lat','deaths','LongQ','fips']].set_index('fips').groupby('LongQ').cumsum().rename(columns={'deaths':'LatDeaths'}).reset_index().drop(columns='lat')
# df=df.merge(cumLatDF,how='inner',on='fips')
# # tmp=pd.merge(df[['LongQ','fips']],cumLatDF,how='inner',left_on='fips', right_index=True)
# quantilesLoc=df.LongDeaths.quantile(quantiles)
# # quantDF=



# # fipList=covidDF.fips.unique()
# # Date=covidDF.date.max()
# # df=pd.merge(covidDF,geoDF



# # %%
