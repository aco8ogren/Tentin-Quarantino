import numpy as np
import pandas as pd
import git
import os
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)

# -------------------------------------------------------------------------
# Function Inputs
# alloc_day='2020-05-10'
# clust_flnm='clustering_clusters.csv'
# numDeaths=5
# numCases=4
# numMobility=3
# lenOutput=5
# remove_sparse=True
# modelDir=None
# -------------------------------------------------------------------------



def Training(alloc_day,clust_fln,numDeaths=5,numCases=5,numMobility=5,lenOutput=5,remove_sparse=True,Patience=4, DropoutRate=.1,modelDir=None):
#%% Data formatting

    lenInput=np.max([numDeaths,numCases,numMobility])
    totLen=lenInput+lenOutput
    clusterDF=pd.read_csv(clust_fln)
    alloc_day=pd.to_datetime(alloc_day)
    

    df=pd.read_csv(r'data\us\covid\nyt_us_counties_daily.csv')
    df=df.merge(clusterDF[['fips','cluster']],how='right', on ='fips')
    tmp=df.groupby(['cluster','date'])[['deaths','cases']].sum()
    # df[['clusterDeaths','clusterCases']]=df.groupby(['cluster','date'])['deaths','cases'].transform(np.sum)
    df['clusterDeaths']=df.groupby(['cluster','date'])['deaths'].transform(np.sum)
    df['clusterCases']=df.groupby(['cluster','date'])['cases'].transform(np.sum)
    df['deathsFrac']=df['deaths']/df['clusterDeaths']
    df['casesFrac']=df['cases']/df['clusterCases']

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
    #---------------------------------------------------------
    # uncomment
    for cluster in df.cluster.unique():
        print(cluster)
        clusterFips=df[df.cluster==cluster].fips.unique()
        FIPS.extend(clusterFips.tolist())
        for fip in clusterFips:
            DF=df[df.fips==fip]
            DF.fillna(0,inplace=True)
            DF=DF.sort_values(by='date')
            firstDeath=np.argmax(DF.deaths.values)
            if remove_sparse:
                if firstDeath-numDeaths-1>0:
                    DF=DF.iloc[firstDeath-numDeaths:]
            if len(DF)>=totLen:
                for i in range(len(DF)-totLen+1):
                    # print(i)
                    Input=[ DF.iloc[i+lenInput-numDeaths:i+lenInput].deathsFrac.values, 
                            DF.iloc[i+lenInput-numCases:i+lenInput].casesFrac.values, 
                            DF.iloc[i+lenInput-numMobility:i+lenInput].m50_index.values] 
                    Input=[element for subList in Input for element in subList]
                    if len(Input)!=numDeaths+numCases+numMobility:
                        raise ValueError('Input lengths are wrong')
                    Inputs.append(Input)
                    countyDeaths=DF.iloc[i+numDeaths:i+totLen].deaths.sum()
                    clusterDeaths=DF.iloc[i+numDeaths:i+totLen].clusterDeaths.sum()
                    if clusterDeaths==0:
                        output=0
                    else:
                        output=countyDeaths/clusterDeaths
                    Outputs.append(output)
    X=np.array(Inputs)
    # X=X.reshape(X.shape+(1,))
    Y=np.array(Outputs)
    np.savetxt(r'Josh/X.txt.',X)
    np.savetxt(r'Josh/Y.txt.',Y)
    #---------------------------------------------------------
    # X=np.loadtxt(r'Josh/X.txt.')
    # Y=np.loadtxt(r'Josh/Y.txt.')


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


    XY=np.concatenate([X,Y[np.newaxis].T],1)
    inds=~np.isnan(XY).any(1)
    X=X[inds]
    Y=Y[inds]
    means=X.mean(0)
    stds=X.std(0)
    X=(X-means)/stds

    np.savetxt(os.path.join(modelDir,'means.txt'),means)
    np.savetxt(os.path.join(modelDir,'stds.txt'),stds)
    #%%
    model=keras.models.Sequential()
    model.add(Dense(X.shape[1],input_dim=X.shape[1],activation='sigmoid'))
    model.add(Dropout(DropoutRate))
    model.add(Dense(int(np.floor(X.shape[1]/2)),activation='sigmoid'))
    model.add(Dropout(DropoutRate))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #%%

    # filepath=os.path.join(checkpointDir,"model-{epoch:02d}-{loss:.4f}.hdf5")
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint,keras,keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=4)]
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=Patience)]
    model.fit(X, Y, epochs=200, batch_size=32, callbacks=callbacks_list,validation_split=0.1)


    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(modelDir,"model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(modelDir,"model.h5"))

    return modelDir
