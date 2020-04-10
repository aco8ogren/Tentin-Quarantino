#%%
import os 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pickle
import sys
import git
sys.path.append('Josh/Processing/')
from nyt_us_counties_Import2 import us_counties_Data2Dict
from matplotlib import pyplot as plt

# %% Change Directory to git root folder: Tentin-Quarantino
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)

#%%
TrainingVectorLength=10     #number of days to use for RNN input vector
DataDict=us_counties_Data2Dict(RemoveEmptyFips=True)



Fips=np.unique(DataDict['fips'])
N=len(DataDict['deaths'])
CountyData={}
for fip in Fips:
    countyInds=np.nonzero(DataDict['fips']==fip)
    countyN=len(DataDict['deaths'][countyInds])
    CountyData[fip]=np.concatenate((DataDict['day'][countyInds].reshape(countyN,1),DataDict['deaths'][countyInds].reshape(countyN,1)),1)

removeFips=[]
nonSeqDict={}
for fip, data in CountyData.items():
    days=data[:,0]
    if not np.array_equal(days,np.arange(days.min(),days.max()+1)):
        # nonSeqDict[fip]=data
        removeFips+=[fip]
        i+=1
    elif (data[:,1]==0).all():
        removeFips+=[fip]

for fip in removeFips:
    del CountyData[fip]   

#%%

lens=np.zeros((len(CountyData),2))
newLens=np.zeros((len(CountyData),2))
for (i,(fip, data)) in enumerate(CountyData.items()):
    lens[i]=[fip,len(data)]
    if lens[i][1]<TrainingVectorLength+1:
        addDays=int(TrainingVectorLength+1-lens[i][1])
        minDay=data[0,0]
        addData=np.concatenate((np.arange(minDay-addDays,minDay).reshape(addDays,1),np.zeros((addDays,1))),1)
        CountyData[fip]=np.concatenate((addData,CountyData[fip]),0)
    newLens[i]=[fip,len(CountyData[fip])]
#%%
X=[]
Y=[]
for fip,data in CountyData.items():
    deaths=data[:,1]
    for i in range(len(deaths)-TrainingVectorLength):
        X.append(deaths[i:i+TrainingVectorLength])
        Y.append(deaths[i+TrainingVectorLength])
X=np.array(X)
X=X.reshape(X.shape+(1,))
Y=np.array(Y)




model=Sequential()
model.add(LSTM(200,input_shape=(10,1),return_sequences=True,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(LSTM(100,activation='sigmoid',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

filedir="Josh/RNN/RNN 1/Checkpoints/"
filename=filedir+os.listdir(filedir)[-1]
model.load_weights(filename)
model.compile(loss='mean_squared_error', optimizer='adam')
XX=X.reshape(X.shape[:-1])
y=model.predict(X).reshape(Y.shape)

#%%
nyFips=36061
days,deaths=CountyData[nyFips][:,0],CountyData[nyFips][:,1]


# %%
x=np.zeros((14,10))
predDeaths=np.copy(deaths)
predDeaths[-14:]=0
for j in np.arange(-14,0):
  predDeaths[j]=model.predict(predDeaths[j-10:j].reshape(1,10,1))
plt.figure(1)
plt.plot(days,deaths,'b')
plt.plot(days[-14:],predDeaths[-14:],'r')
plt.title('New York Deaths')
plt.xlabel('Days since January 1st')
plt.ylabel('Deaths since January 1st')
plt.legend(('True','Prediction'))
plt.show()  

#%%







for i,j in enumerate(np.arange(-14,0)):
  x[i,:]=(deaths[j-10:j])
x=x.reshape(x.shape+(1,))
y=model.predict(x)


# %%
