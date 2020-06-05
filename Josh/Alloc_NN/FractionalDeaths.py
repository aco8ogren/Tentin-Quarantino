import numpy as np
import pandas as pd
import git
import os
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)


def Training(alloc_day,clust_fln,numDeaths=5,numCases=5,numMobility=5,lenOutput=5,remove_sparse=True,Patience=4, DropoutRate=.1,modelDir=None):
    # Funciton to train allocation neural net

    # The input length is the number of days used in the input. It is the maximum 
    # length of any of the input variables
    lenInput=np.max([numDeaths,numCases,numMobility])
    totLen=lenInput+lenOutput
    # Read in the cluster data frame from the training
    clusterDF=pd.read_csv(clust_fln)
    alloc_day=pd.to_datetime(alloc_day)
    
    # read in the NYT daily death data to be used in training
    df=pd.read_csv(r'data\us\covid\nyt_us_counties_daily.csv')
    # add cluster data to the main data frame
    df=df.merge(clusterDF[['fips','cluster']],how='right', on ='fips')
    # calculate the total cluster deaths and cases, and the fraction for
    # each county in the cluster
    tmp=df.groupby(['cluster','date'])[['deaths','cases']].sum()
    # df[['clusterDeaths','clusterCases']]=df.groupby(['cluster','date'])['deaths','cases'].transform(np.sum)
    df['clusterDeaths']=df.groupby(['cluster','date'])['deaths'].transform(np.sum)
    df['clusterCases']=df.groupby(['cluster','date'])['cases'].transform(np.sum)
    df['deathsFrac']=df['deaths']/df['clusterDeaths']
    df['casesFrac']=df['cases']/df['clusterCases']

    # read in mobility data and add it to the main dataframe
    if numMobility>0:
        mobilityDF=pd.read_csv(r'data\us\mobility\DL-us-mobility-daterow.csv')[['fips','date','m50_index']]
        df=df.merge(mobilityDF,how='inner',on=['date','fips'])
    else:
        df.insert(df.shape[1],'m50_index',np.nan*np.zeros(len(df)))
    df.loc[:,'date']=pd.to_datetime(df.date)
    df=df[df.date<=alloc_day]
    Inputs=[]
    Outputs=[]
    lens=[]
    FIPS=[]
    badFips=[]
    df=df.sort_values(by=['cluster','fips','date'])
    
    # gather data for each cluster
    for cluster in df.cluster.unique():
        clusterFips=df[df.cluster==cluster].fips.unique()
        FIPS.extend(clusterFips.tolist())
        # gather data for each county in the cluster
        for fip in clusterFips:
            # get the dataframe just for the county
            DF=df[df.fips==fip]
            DF.fillna(0,inplace=True)
            DF=DF.sort_values(by='date')
            # get the date of the first non-zero data
            firstDeath=np.argmax(DF.deaths.values)
            # if the remove_sparse parameter is true, get rid of all data before the first 
            # death that would produce all zero death data to aid in training
            if remove_sparse:
                if firstDeath-numDeaths-1>0:
                    DF=DF.iloc[firstDeath-numDeaths:]
            # Ensure that the data for this county has enough days to make at least
            # one input/output pair
            if len(DF)>=totLen:
                # for all totLen length sequence of days, create input/output pairs
                for i in range(len(DF)-totLen+1):
                    # create the input from fractional deaths, cases and m50 index values
                    Input=[ DF.iloc[i+lenInput-numDeaths:i+lenInput].deathsFrac.values, 
                            DF.iloc[i+lenInput-numCases:i+lenInput].casesFrac.values, 
                            DF.iloc[i+lenInput-numMobility:i+lenInput].m50_index.values] 
                    Input=[element for subList in Input for element in subList]
                    if len(Input)!=numDeaths+numCases+numMobility:
                        raise ValueError('Input lengths are wrong')
                    Inputs.append(Input)
                    # create output from the mean of lenOutput (totLen-numDeaths) days of 
                    # fractional death data
                    countyDeaths=DF.iloc[i+numDeaths:i+totLen].deaths.sum()
                    clusterDeaths=DF.iloc[i+numDeaths:i+totLen].clusterDeaths.sum()
                    if clusterDeaths==0:
                        output=0
                    else:
                        output=countyDeaths/clusterDeaths
                    Outputs.append(output)
    # Create input and output arrays
    X=np.array(Inputs)
    Y=np.array(Outputs)

    #%%
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint
    # from keras.utils import np_utils
    from datetime import datetime
    if modelDir is None:
        modelDir=r'Josh\Alloc_NN\ModelSaves\%i_%i_%i_%i_%s'%(numDeaths,numCases,numMobility,lenOutput,datetime.now().strftime("%m-%d_%H-%M"))
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)
    checkpointDir=os.path.join(modelDir,'Checkpoints')
    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)

    # remove any nan data
    XY=np.concatenate([X,Y[np.newaxis].T],1)
    inds=~np.isnan(XY).any(1)
    X=X[inds]
    Y=Y[inds]
    # calculate column means and standard deviations to normalize data for learning
    means=X.mean(0)
    stds=X.std(0)
    stds[stds==0]=1
    X=(X-means)/stds
    # save means and standard deviations for use in predictions
    np.savetxt(os.path.join(modelDir,'means.txt'),means)
    np.savetxt(os.path.join(modelDir,'stds.txt'),stds)
    # build model with two hidden layers of size input length and floor(input lenght/2)
    # two dropout layers
    model=keras.models.Sequential()
    model.add(Dense(X.shape[1],input_dim=X.shape[1],activation='sigmoid'))
    model.add(Dropout(DropoutRate))
    model.add(Dense(int(np.floor(X.shape[1]/2)),activation='sigmoid'))
    model.add(Dropout(DropoutRate))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # monitor validation loss and stop training when it does not decrease for Patience epochs
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=Patience)]
    model.fit(X, Y, epochs=200, batch_size=32, callbacks=callbacks_list,validation_split=0.1)

    # save model architechture and weights
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(modelDir,"model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(modelDir,"model.h5"))

    return modelDir
