# %%
import os
Direc='Tentin-Quarantino'
os.chdir(os.getcwd()[:os.getcwd().find(Direc)+len(Direc)])

#%%
import numpy as np
import pickle

# %% import GeoLoc data
GeoDat=np.loadtxt('data/us/geolocation/county_centers.csv',dtype=str,delimiter=',',skiprows=1)

# %%
# Using the last two columns of the GeoData, as these correspond to 2010 population weighted county center long/lat
# There are 4 values missing from each of the last two columns, so replacing these with the two columns before, i.e. 2000 population weighted county centers
GeoDat[GeoDat[:,-1]=='NA',-1]=GeoDat[GeoDat[:,-1]=='NA',-3]
GeoDat[GeoDat[:,-2]=='NA',-2]=GeoDat[GeoDat[:,-2]=='NA',-4]
LatLong=GeoDat[:,-2:].astype(float)
fips=GeoDat[:,0].astype(int)

# Create dictionary which relates fips integer code to shape (2,) np array of [long,lat]
GeoDict={fips[i]:LatLong[i] for i in range(len(fips))}
GeoDict[0]=np.nan
File=open('Josh/Processing/Processed Data/GeoDict.pkl','wb')
pickle.dump(GeoDict,File)
File.close()
