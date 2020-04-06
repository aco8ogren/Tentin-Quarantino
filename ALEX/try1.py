import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as krs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU
#import Keras
#from Keras.models import Sequential
#from Keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
#from scipy.io import loadmat
import time
import os

#script_path = os.path.dirname(os.path.realpath(__file__))
#os.chdir("../../training_data/graph222_6/point")
#os.chdir("../training_data/graph222/point")

data_df = pd.read_csv('C:/Users/alex/OneDrive - California Institute of Technology/Documents/GitHub/Tentin-Quarantino/data/us/covid/deaths.csv')
[m,n] = np.shape(data_df)
data_df = data_df.drop(m-1)

X = X_db.to_numpy()
Y = Y_db.to_numpy()

d = 20 # number of wavenumbers included in training data
[m,n] = np.shape(X)

## Creating the model
model = tf.keras.models.Sequential()

activ = 'sigmoid'
## The input layer (plus one hidden layer?)
model.add(Dense(2000,input_shape=(n,)))
model.add(Activation(activ))
#model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(.088))

## The hidden layers
model.add(Dense(100))
model.add(Activation(activ))
#model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(.088))

## The output layer
model.add(Dense(1))
model.add(Activation(activ))

## Printing a summary of the layers and weights in your model
model.summary()

## In the line below we have specified the loss function as 'rmse' (Root Mean Squared Error) because in the above code we did not one-hot encode the labels.
## In your implementation, since you are one-hot encoding the labels, you should use 'categorical_crossentropy' as your loss.
## You will likely have the best results with RMS prop or Adam as your optimizer.  In the line below we use Adadelta
model.compile(loss='mean_squared_error',optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])

X_shuff,Y_shuff = shuffle(X,Y)
fit = model.fit(X_shuff,Y_shuff, batch_size=4096*4, epochs=30, verbose=1)
#fit = model.fit(x_train, y_train,
#                    batch_size=64,
#                    epochs=3,
#                    # We pass some validation for
#                    # monitoring validation loss and metrics
#                    # at the end of each epoch
#                    validation_data=(x_val, y_val))

## Printing the accuracy of our model, according to the 'metrics' function specified in model.compile above
fit2 = model.fit(X_shuff,Y_shuff, batch_size=4096, epochs=70, verbose=1)
score = model.evaluate(X, Y, verbose=0)
print('Training score:', score[0])
print('Training accuracy:', score[1])

print(fit.history.keys())

plt.plot(fit2.history['loss'])  
#plt.plot(fit.history['val_acc'])  
plt.title('training curve')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train'], loc='upper right') 

#N_ex = 9
#randperm = np.random.permutation(int(m/d))
#structs = randperm[range(N_ex)]
#for i in range(N_ex):
#    struct = structs[i]
#    start_pointer = struct*d
#    end_pointer = start_pointer + d
#    X_temp = X[start_pointer:end_pointer,:]
#    Y_temp = Y[start_pointer:end_pointer]
#    Y_pred = model.predict(X_temp)
#    wavenumbers = X_temp[:,-1]
#    #fig = plt.figure(figsize=(20,10))
#    fig = plt.figure(figsize=(7.5,2.5))
#    #plt.subplot(3,3,i+1)
#    plt.plot(wavenumbers,Y_temp)
#    plt.plot(wavenumbers,Y_pred)
#    plt.title(str(np.round(X_temp[1,0:-1],3)))


RMSEs = np.zeros((int(m/d),)) 
errors_point = np.zeros((m,))
a = time.time()
Y_pred = model.predict(X,batch_size = 4096*4)
b = time.time()
print('predictions in total took ' + str(b-a))
a = time.time()
for i in range(int(m/d)):
    start_pointer = i*d
    end_pointer = start_pointer + d
    RMSE = np.sqrt(np.mean((Y_pred[start_pointer:end_pointer] - Y[start_pointer:end_pointer])**2))
#    print('evaluate took ' + str(b-a))
    RMSEs[i]=RMSE
    errors_point[start_pointer:end_pointer] = np.reshape(Y_pred[start_pointer:end_pointer] - Y[start_pointer:end_pointer],(20,))
#    print('add to array took ' + str(b-a))
#    print('control took ' + str(b-a))
    botthresh = 2e-2
    topthresh = 2e-2 + .0001
    if RMSE>botthresh and RMSE<topthresh:
        fig = plt.figure()
        X_temp = X[start_pointer:end_pointer]
        Y_temp = Y[start_pointer:end_pointer]
        wavenumbers = X_temp[:,-1]
        plt.plot(wavenumbers,Y_temp,'ko')
        plt.plot(wavenumbers,Y_pred[start_pointer:end_pointer],'rx')
b = time.time()
print('evaluation in total took ' + str(b-a))
fig = plt.figure()
plt.hist(RMSEs,bins = 200,range = (0.0,.6));
plt.title('Histogram of errors for full design space')
plt.xlabel('RMSE')
plt.ylabel('Occurrences')
plt.show

pd.DataFrame(errors_point).to_csv('errors_point.csv',header = False, index = False)





