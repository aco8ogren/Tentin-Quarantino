import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Lambda
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
normalize=float(X.max())
X=X/normalize

Y=np_utils.to_categorical(Ydata)

model=Sequential()
#model.add(LSTM(200,input_shape=X.shape[1:]))
model.add(LSTM(200,input_shape=X.shape[1:],return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200))
model.add(Dropout(0.2))
model.add(Lambda(lambda x: x / 1e-10 ))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
#%% 
#filename='Checkpoints - Copy/'+os.listdir('Checkpoints - Copy')[-1]
filename='Liamm Checkpoint/'+os.listdir('Liamm Checkpoint')[-1]
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
num2char=dict((integer,char) for integer, char in enumerate(chars))

start = np.random.randint(0, len(Xdata)-1)
pattern=Xdata[start]
#pattern=list(np.random.randint(0,38,seqLength))
#pattern = [char2num[i] for i in " shall i compare thee to a summer's day\n"]
seed=np.copy(pattern)
print("Seed:",end='')
print("\"", ''.join([num2char[value] for value in pattern]), "\"")
output=''
counter=0
# generate characters
while counter<14:
#for i in range(200):
  x = np.reshape(pattern, (1, len(pattern), 1))/normalize
  
  prediction = model.predict(x, verbose=0)
  index=np.random.choice(len(prediction[0]),p=prediction[0])
#  index = prediction.argmax()
  result = num2char[index]
  if result=='\n':
    counter+=1
  seq_in = [num2char[value] for value in pattern]
  output+=result
#  print(result,end='')
  pattern.append(index)
  pattern = pattern[1:len(pattern)]
  
print(output)    
  