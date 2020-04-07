#%%
import numpy as np
# set directory to TQ root
DIR='Tentin-Quarantino'
direc=os.getcwd()
TQ_dir=direc[:direc.find(DIR)+len(DIR)]
os.chdir(TQ_dir)


#%%
#import GeoLoc data
# GeoDat=np.loadtxt('data/us/geolocation/county_centers.csv',dtype=str,delimiter=',',skiprows=1)
GeoDat=np.loadtxt('data/us/geolocation/county_centers.csv',delimiter=',',skiprows=1,converters={'NA':666})


# %%
