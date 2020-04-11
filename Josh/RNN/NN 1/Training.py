# %% 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pickle
import sys
sys.path.append('Josh/Processing/')
from dateProcess import DateProcess


# %% Change Directory to git root folder: Tentin-Quarantino
import os 
FileDir=os.getcwd()
Direc='Tentin-Quarantino'
os.chdir(os.getcwd()[:os.getcwd().find(Direc)+len(Direc)])

#%% Import fips GeoLoc dictionary
File=open('Josh/Processing/Processed Data/GeoDict.pkl','rb')
GeoDict=pickle.load(File)

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
coords=np.array([GeoDict[DataDict['fips'][i]] for i in range(len(DataDict['fips']))])
means=coords.mean(0)
stds=coords.std(0)
coords=(coords-means)/stds
np.savetxt('Josh/RNN/NN 1/Coord_Mean_Std.txt',[means,stds])
DataDict['coords']=coords
# DataDict['coordinate

#%% Convert Dates into day since January 1st
DataDict['day']=np.array([[DateProcess(DataDict['date'][i])] for i in range(len(DataDict['date']))])

#%% Create X/Y input/output arrays
X=np.concatenate((DataDict['day'],DataDict['coords']),1)
Y=DataDict['deaths'].astype(int)

# %% define the model!
model = Sequential()
model.add(Dense(200,input_shape=X.shape[1:],activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(200,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

filepath="Josh/RNN/NN 1\\Checkpoints\\weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


model.fit(X, Y, epochs=20000, batch_size=32, callbacks=callbacks_list)




