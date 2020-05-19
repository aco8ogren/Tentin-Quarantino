#%%
import pandas as pd
import numpy as np
import git
import os
import sys
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)
sys.path.append(cwd)
from Josh.CountyClustering.ClusterByDeaths import JoshMeansClustering as JMC
from Josh.CountyClustering.functions.ClusterPlot import plot
from Josh.Clustering.kMeansClustering import Kmeans

geoDF=pd.read_csv(r'Josh\Processing\Processed Data\GeoData.csv')
Ogala={k:v for k,v in zip(['fips','long','lat'],[46102,-102.821318,43.352275])}
geoDF=geoDF.append(Ogala,ignore_index=True)


covidDF=pd.read_csv('data/us/covid/nyt_us_counties.csv')
covidDF=covidDF[~covidDF.fips.isna()]
date=covidDF.date.max()
kMeansDF=covidDF[covidDF.date==date]
kMeansDF=kMeansDF[kMeansDF.deaths<50]
fipsList=kMeansDF[kMeansDF.deaths<50].fips.unique()
kMeansDF=pd.merge(kMeansDF,geoDF,how='left', on='fips')

# AL_HA=kMeansDF[kMeansDF.long<-128]
# kMeansDF=kMeansDF[kMeansDF.long>=-128]
k=54
# AL_HA['cluster']=np.nan*np.ones(len(AL_HA))
kMeansDF['cluster']=Kmeans(k,kMeansDF[['long','lat']].values)[0]
kMeansDF['clusterDeaths']=kMeansDF.groupby('cluster')['deaths'].transform(np.sum)
# AL_HA.loc[AL_HA.lat>40,'cluster']=k
# AL_HA.loc[AL_HA.lat<=40,'cluster']=k+1
# AL_HA['clusterDeaths']=np.nan
# kMeansDF=pd.concat((kMeansDF,AL_HA))
plot(kMeansDF,fname='Josh/Kmeans3.svg',title='K-Means Clustering')


#%%





# X=kMeansDF[['long','lat']].values
# #%%





# jmcDF=JMC(fipsList,date)
# plot(jmcDF,fname='Josh/JMC.svg',title='Rolling Cluster')
# plot




# %%
