import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
File = open("./data/AllSonnets.txt","r") 
DataLines = File.readlines()
File.close()
Data=''.join(DataLines)
chars=sorted(list(set(Data)))
char2num=dict((char, integer) for integer, char in enumerate(chars))

seqLength=40
Xdata=[]
Ydata=[]
for i in range(0,len(Data)-seqLength):
    x=Data[i:i+seqLength]
    y=Data[i+seqLength]
    Xdata.append([char2num[char] for char in x])
    Ydata.append(char2num[y])
    

X=np.reshape(Xdata,np.array(Xdata).shape+(1,))
X=X/float(X.max())

Y=np_utils.to_categorical(Ydata)

model=Sequential()
model.add(LSTM(200,input_shape=X.shape[1:]))
#model.add(LSTM(200,input_shape=X.shape[1:],return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(200))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


filepath="Checkpoints/weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


model.fit(X, Y, epochs=20000, batch_size=32, callbacks=callbacks_list)


