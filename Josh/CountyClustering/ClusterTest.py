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
from Josh.CountyClustering.ClusterPlot import plot

covidDF=pd.read_csv('data/us/covid/nyt_us_counties.csv')
covidDF=covidDF[~covidDF.fips.isna()]
date=covidDF.date.max()
CDF=covidDF[covidDF.date==date]
fipsList=CDF[CDF.deaths<50].fips.unique()

clusterDF=JMC(fipsList,date)
plot(clusterDF,'Josh/JMC.svg')

# import matplotlib.pyplot as plt
# alaskaHaw=clusterDF[clusterDF.long<-128]
# Continental=clusterDF[clusterDF.long>=-128]
# AL=alaskaHaw[alaskaHaw.lat>40]
# HW=alaskaHaw[alaskaHaw.lat<=40]
# plt.scatter(Continental.long,Continental.lat,c='k')
# plt.scatter(AL.long,AL.lat,c='r')
# plt.scatter(HW.long,HW.lat,c='b')
