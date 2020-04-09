#%%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pickle
import sys
sys.path.append('Josh/Processing/')
from dateProcess import DateProcess
from matplotlib import pyplot as plt


# %% Change Directory to git root folder: Tentin-Quarantino
import os 
Direc='Tentin-Quarantino'
os.chdir(os.getcwd()[:os.getcwd().find(Direc)+len(Direc)])

#%%
File=open('Josh/Processing/Processed Data/GeoDict.pkl','rb')
GeoDict=pickle.load(File)
data=np.loadtxt('data/us/covid/nyt_us_counties.csv',dtype=str,delimiter=',')
MeanStd=np.loadtxt('Josh/RNN/NN 1/Coord_Mean_Std.txt')

#%% Try for Nassau County
fips=36059
coords=(GeoDict[fips]-MeanStd[0])/MeanStd[1]

# data=np.loadtxt('data/us/covid/nyt_us_counties.csv',dtype=str,delimiter=',')
# FipsCol=np.nonzero(data[0]=='fips')[0][0]
# data=data[data[:,FipsCol]!='']
# deaths=data[1:,data[0]=='deaths'][:,0]
# dates=data[1:,data[0]=='date'][:,0]
# days=np.array([DateProcess(date) for date in dates])
# latest=days.max()
# ProjecedDays=np.arange(latest+1,latest+15).reshape((14,1))
# Xpred=np.concatenate((ProjecedDays,np.ones((14,1))*coords[0],np.ones((14,1))*coords[1]),1)

#%% Load data and create dictionary with keys of column headers and values of column contents
data=np.loadtxt('data/us/covid/nyt_us_counties.csv',dtype=str,delimiter=',')
# First, some county info is missing, so remove data from unknown counties
FipsCol=np.nonzero(data[0]=='fips')[0][0]
data=data[data[:,FipsCol]!='']
DataDict={data[0][i]:data[1:,i] for i in range(data.shape[1])}

#%% Keys for variables. If changed an error is thrown and  must be updated manually
Keys=['date','county','state','fips','cases','deaths']
for key in Keys:
    if key not in DataDict:
        raise ValueError("Column Headers changed; update keys")


# %% Convert fips data from str to int then into coordinate pairs, then normailize coordinate data
DataDict['fips']=DataDict['fips'].astype(int)

#%% Convert Dates into day since January 1st
DataDict['day']=np.array([[DateProcess(DataDict['date'][i])] for i in range(len(DataDict['date']))])
latest=DataDict['day'].max()
ProjecedDays=np.arange(latest+1,latest+15).reshape((14,1))
Xpred=np.concatenate((ProjecedDays,np.ones((14,1))*coords[0],np.ones((14,1))*coords[1]),1)



model = Sequential()
model.add(Dense(200,input_shape=(3,),activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(200,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1))

filename='Josh/RNN/NN 1/Checkpoints/'+os.listdir('Josh/RNN/NN 1/Checkpoints/')[-1]
print(filename)
model.load_weights(filename)
model.compile(loss='mean_squared_error', optimizer='adam')

# %%
Ypred=model.predict(Xpred)[:,0]
# NassauData=data[(data[:,data[0]=='fips']=='36059')[:,0],:]
NassauInds=np.nonzero(DataDict['fips']==36059)
KnownDays=DataDict['day'][NassauInds]
KnownDeaths=DataDict['deaths'][NassauInds].reshape(KnownDays.shape).astype(int)
plt.figure(1)
plt.plot(KnownDays,KnownDeaths,color='r')
plt.plot(ProjecedDays,Ypred,color='b')
plt.show

plt.figure(2)
# plt.plot(





# %%


# %%
