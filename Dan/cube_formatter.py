import os
import numpy as np
import scipy.io as sio
import pandas as pd

# Library for performing basic operations on our data cubes
# Allows for consistency accross various scripts that perform the same operations


def read_cube(filename):
    """
    # Function to read in data cube (.npy OR .mat) and return a numpy cube
    # 
    # Arguments: 
    #   - filename  (string) filename to load
    #                 *Must be either .npy or .mat*
    #                 format for input cube:
    #                 row=sample, col=date, pane=county
    #
    # Output:
    #   - data      a numpy cube
    # 
    # *** First row should be beef (ie:)
    #   FIPS | 4022020 | 4032020 | ...
    """

    ##-- Read the file
    # Obtain file extension
    ext = os.path.splitext(filename)[1]

    # Extract data
    if ext == '.mat':
        # load data as dictionary
        data = sio.loadmat(filename)
        # get matlab matrix variable name
        vnm = sio.whosmat(filename)[0][0]
        # extract matrix
        data = data[vnm]
    elif ext == '.npy':
        data = np.load(filename)
    else:
        raise ValueError('The provided filename has an unsupported extension. Must be .mat or .npy')
    
    return data

def calc_daily(data):
    """
    # Function to convert a cube of data from cumulative deaths to daily deaths
    # 
    # Arguments: 
    #   - data      cube with cumulative death counts
    #                 format for input cube:
    #                 row=sample, col=date, pane=county
    #
    # Output:
    #   - data      a numpy cube with daily deaths
    # 
    # *** First row should be beef (ie:)
    #   FIPS | 4022020 | 4032020 | ...
    #
    # -----------------------------------------------------
    #
    # Method:   Create matrix with same data but shifted left (column-wise) by one.
    #           Append 0's to the first column.
    #           Subtract the time-shifted column from the original column.
    # Result:   A matrix of the original size but with the daily death count
    """
    
    # Create shifted matrix with 0's in left column
    datashift=np.concatenate((np.zeros((data.shape[0],1,data.shape[2])),data[:,:-1,:]),1)
    # Subtract shifted from original. Don't subtract 1st row; beef stays unchanged
    data[1:,:,:]-=datashift[1:,:,:]

    return data

def alloc_counties(data, county_ref_fln, alloc_day=None):
    """
    # Function to allocate state-wide deaths amongst counties
    # 
    # Arguments:
    #   - data      cube with daily death counts
    #                 format for input cube:
    #                 row=sample, col=date, pane=county
    #
    #   - county_ref_fln 
    #               filename of reference file to use for allocation
    #   
    #   - alloc_day Day to use to get proportions for allocations
    #               if None: uses most recent value in ref file
    #               else: use the date provide [format='YYYY-MM-DD']
    #
    # Output:
    #   - data      a numpy cube with daily deaths by county
    # 
    # *** First row should be beef (ie:)
    #   FIPS | 4022020 | 4032020 | ...
    #
    # -----------------------------------------------------
    #
    # Method:   * Allocates proportionally based on most recent deaths report*
    #           Import the latest NYT _DAILY_ county data
    #           Find the most recent date entries
    #           Determine fraction of deaths that each county holds for a given state
    #           Multiply the state-wide predicted deaths by these fractions
    #           Populate a new matrix with the results
    # 
    # Result:   A new matrix with deaths by state re-allocated to its counties
    #           This matrix is inherently larger than the original
    #           The beef is changed to county FIPS instead of state
    #           * Only counties present in NYT_daily file will be represented in new matrix
    #             - ie. only counties with at least 1 reported case
    #           Dimensionality is: [sample, day, county]
    """
    ##-- Separate input cube into state FIPS layer and data layer
    st_fips_data = ['%d' %int(x) for x in data[0,0,:]]    # cast to string so fips datatypes match
    st_deaths_data = data[1:,:,:].copy()

    ##-- Load the reference county file and extract pertinent data
    ref_data = pd.read_csv(county_ref_fln)

    if alloc_day is not None:
        lst_date = alloc_day
    else:
        # most recent date in set
        lst_date = max(np.array(ref_data['date']))
    # remove all data not from this date
    ref_data = ref_data[ref_data['date'] == lst_date]
    
    # Order by county code
    ref_data.sort_values(by=['fips'],inplace=True)

    # get fips codes (as list of strings to allow extraction of state code)
    ref_fips  = ['%d' %int(x) for x in np.array(ref_data['fips'])]
    ref_st_fips = []
    # get state code for all fips
    for i in range(0,len(ref_fips)):
        ref_st_fips.append(ref_fips[i][:-3])
    ref_data['st_fips'] = ref_st_fips
    
    # # get death count
    # deaths = np.array(ref_data['deaths'])

    ##-- Preallocate new data matrix (use nan to check if anything is unallocated later)
    data = np.zeros((data.shape[0],data.shape[1],len(ref_fips)))#*np.nan

    ##-- Iterate through states in input cube and allocate predcictions
    ind = 0
    # NOTE/TODO: if input data (st_fips_data) does not have a state that the ref_fips does,
        # the state will not be iterated over and thus we'll still have nans
    for i in range(0,len(st_fips_data)):
        # Get rows related to the given state
        st_rows = ref_data[ref_data['st_fips'] == st_fips_data[i]]
        # Get total deaths in state
        tot_deaths = sum(st_rows['deaths'])

        # Calculate proportion of deaths per county
        cnty_deaths = np.array(st_rows['deaths']/tot_deaths)

        #-- Allocate deaths by current proporitionality
        # Create matrix of predictions for later multiplication
        data_st = st_deaths_data[:,:,i].copy()
        data_st = data_st.reshape(data_st.shape+(1,))
        data_st = np.tile(data_st,(1,1,len(cnty_deaths)))
        # --> data_st is now: (samples,days,counties-in-state)
        # Create matrix of cnty_deaths for later mult.
        cnty_deaths = cnty_deaths.reshape((1,1,)+cnty_deaths.shape)
        cnty_deaths = np.tile(cnty_deaths,(data_st.shape[0],data_st.shape[1],1))
        # --> cnty_deaths is now: (samples,days,counties-in-state)
        # Multiply the two matrices together
        tmp = data_st*cnty_deaths

        #-- Create county FIPS beef
        beef = np.zeros((1,)+data_st.shape[1:])
        beef[0,:,:] = np.array(st_rows['fips'])

        #-- Place result into final matrix
        # Place predictions
        if tot_deaths == 0:
            # TODO: replace "pass" w/ below. This'll allow nan replacement.
            # data[1:,:,ine:ind+len(st_rows)] = tmp*0
            pass
        else:
            data[1:,:,ind:ind+len(st_rows)] = tmp
        # Add county FIPS beef
        data[[0],:,ind:ind+len(st_rows)] = beef


        #-- Increase index of current matrix allocation
        ind += len(st_rows)

    if np.sum(np.isnan(data)) != 0:
        raise ValueError('An element of the output matrix is unallocated')

    return data



def alloc_fromCluster(data, cluster_ref_fln,alloc_day=None,num_alloc_days=5):
    """
    # Function to allocate state-wide deaths amongst counties
    # 
    # Arguments:
    #   - data      cube with daily death counts
    #                 format for input cube:
    #                 row=sample, col=date, pane=cluster
    #                 * NOTE: can be mix of counties and clusters
    #   
    #   - cluster_ref_fln
    #               filename of cluster-to-county relationship file
    #                   should be .csv from Josh's clustering function
    #
    # Output:
    #   - data      a numpy cube with daily deaths by county
    # 
    # *** First row should be beef (ie:)
    #   FIPS | 4022020 | 4032020 | ...
    #
    # -----------------------------------------------------
    #
    # NOTE:     This currently uses the death values in the clustering file to
    #               allocate deaths. As such, it is allocating based on the
    #               cummulative deaths that a given county had on the day
    #               in which clustering was performed.
    #               We may want to change this so that we base it on the daily
    #               deaths since the input cube is daily deaths.
    #
    # Method:   * Allocates proportionally based on deaths report *
    #           Import the clustering file
    #           Determine fraction of deaths that each county in a cluster holds
    #           Multiply the cluster-wide predicted deaths by these fractions
    #           Populate a new matrix with the results
    # 
    # Result:   A new matrix with deaths by cluster re-allocated to its counties
    #           This matrix is inherently larger than the original
    #           The beef is changed to county FIPS instead of cluster id
    #
    #           Dimensionality is: [sample, day, county]
    """
    ref_data = pd.read_csv(r'data\us\covid\nyt_us_counties_daily.csv')
    ref_data=ref_data[['fips','date','deaths']]
    ref_data.loc[:,'date']=pd.to_datetime(ref_data.date)
    if alloc_day is not None:
        lst_date = pd.to_datetime(alloc_day)#-np.timedelta64(1,'D')
    else:
        # most recent date in set
        lst_date = ref_data.date.max()
    ##-- Separate panes that are already counties from panes that are clusters
        # Assumes cluster fips are <1000 and counties are >1000
    data_cnty = data[:,:,data[0,0,:] >= 1000]
    data = data[:,:,data[0,0,:] < 1000]

    ##-- Separate input cube into FIPS layer and data layer
    cube_cluster_data = data[0,0,:].astype('int').tolist()    # cast to int
    cube_deaths_data = data[1:,:,:].copy()


    ##-- Load and process cluster data
    cluster_data = pd.read_csv(cluster_ref_fln)

    # Extract useful columns
    cluster_data = cluster_data[['fips', 'cluster', 'deaths', 'clusterDeaths']]

    # Cast fips and cluster values to int
    cluster_data['fips'] = cluster_data['fips'].astype('int')
    cluster_data['cluster'] = cluster_data['cluster'].astype('int')

    # Extract clusters that are in the data cube
    cluster_data = cluster_data[cluster_data.cluster.isin(cube_cluster_data)]


    ##-- Preallocate new data matrix (use nan to check if anything is unallocated later)
        # Add as many frames as there are fips in the clustering data
        # NOTE: don't include fips from counties in the original data since 
        #  that data gets appended at the end
    data = np.zeros((data.shape[0],data.shape[1],len(cluster_data.index)))*np.nan
    ##-- Iterate through clusters in input cube and allocate predictions
    ind = 0     
    noDataCounter=0
    for i,clust in enumerate(cube_cluster_data):
        # Get rows related to the given state
        clust_rows = cluster_data[cluster_data['cluster'] == clust]
        fips_list=clust_rows.fips.unique()
        fips_data=ref_data[ref_data.fips.isin(fips_list)]
        first_date=lst_date-np.timedelta64(num_alloc_days-1,'D')
        
        fips_data=fips_data[(first_date <= fips_data.date) &  (fips_data.date<= lst_date)].groupby('fips').mean()
        tot_deaths=fips_data.deaths.sum()
        if tot_deaths != 0:
            cnty_deaths=fips_data.deaths.values/tot_deaths
        else:
            cnty_deaths = fips_data.deaths.values*tot_deaths

        noDataCounter+=len(fips_list)-len(fips_data)
        # Calculate proportion of deaths per county

        

        # cnty_deaths2 = (clust_rows['deaths']/clust_rows['clusterDeaths'].iloc[0]).values

        #-- Allocate deaths by current proporitionality
        # Create matrix of predictions for later multiplication
        data_clust = cube_deaths_data[:,:,i].copy()             # Extract this cluster's prediction frame
        data_clust = data_clust.reshape(data_clust.shape+(1,))  # Add 3rd dimension (to allow tiling)
        data_clust = np.tile(data_clust,(1,1,len(cnty_deaths)))    # Repeat frame for every county in cluster
        # --> data_clust is now: (samples,days,counties-in-cluster)
        # Rotate the cnty_deaths vector into 3rd dimension
        cnty_deaths = cnty_deaths.reshape((1,1,)+cnty_deaths.shape)
        # Multiply the proportion of deaths vector with the total deaths to allocate
            # np.multiply broadcasts the vector to the correct dimensionality
        cnty_deaths = np.multiply(data_clust, cnty_deaths)
        # --> cnty_deaths is now: (samples,days,counties-in-cluster)

        #-- Create county FIPS beef (will be top layer of cube)
        # Preallocate matrix
        beef = np.zeros((1,)+data_clust.shape[1:])
        # Populate beef with fips codes 
        # beef[0,:,:] = clust_rows['fips'].values
        beef[0,:,:] = fips_data.index.values


        #-- Place result into final matrix
        # Place predictions
        data[1:,:,ind:ind+len(fips_data.index)] = cnty_deaths
        # Add county FIPS beef
        data[0,:,ind:ind+len(fips_data.index)] = beef

        #-- Increase index of current matrix allocation
        ind += len(fips_data.index)

    # Find number of unallocated panes by counting all nans in cube and dividing by pane size
    numNanPanes=np.count_nonzero(np.isnan(data))/data.shape[0]/data.shape[1]
    # If there are some counties with no data:
        # 
    if noDataCounter>0:
        if numNanPanes!=noDataCounter:
            raise ValueError('An element of the output matrix is unallocated')
        else:
            data=data[:,:,:-(noDataCounter)]
    else:
        if np.sum(np.isnan(data)) != 0:
            raise ValueError('An element of the output matrix is unallocated')


    ##-- Append original data that was already county-based 
    data = np.dstack((data_cnty, data))

    return data


def alloc_NN(data,retrain=True,alloc_day=None,cluster_ref_fln=None,numDeaths=5,numCases=5,numMobility=5,lenOutput=5,remove_sparse=True,Patience=4, DropoutRate=.1, modelDir=None):


    """
    # Function to allocate cluster deaths amongst counties using a neural net
    # 
    # Arguments:
    #   - data      cube with daily death counts
    #                 format for input cube:
    #                 row=sample, col=date, pane=cluster
    #                 * NOTE: can be mix of counties and clusters
    #   
    #   -retrain=True 
    #               flag to retrain model. If true retrains and saves
    #               to modelDir given in 
    #
    #   - alloc_day
    #               day up until which model was trained
    #
    #   - cluster_ref_fln
    #               filename of cluster-to-county relationship file
    #                   should be .csv from Josh's clustering function
    #
    #   -numDeaths = 5    
    #               number of days of death data as input 
    #
    #   -numCases = 5
    #               number of days of cases data as input 
    #
    #
    #   -modelDir = None
    #               directory to save or load neurel net architecture and weights
    #
    #                   numMobility   = 5:    number of days of mobility data as input    
    #                   lenOutput     = 5:    number of days of death data to average    
    #                                           over for output
    #                   remove_sparse = True: boolean flag to remove data points with 
    #                                           zero deaths for inputs and output
    #
    # Output:
    #   - data      a numpy cube with daily deaths by county
    # 
    # *** First row should be beef (ie:)
    #   FIPS | 4022020 | 4032020 | ...
    #
    # -----------------------------------------------------
    #
    #
    # Method:   * Allocates proportionally based on training a neural net *
    #           Import the clustering file
    #           Determine fraction of deaths that each county in a cluster holds
    #           Multiply the cluster-wide predicted deaths by these fractions
    #           Populate a new matrix with the results
    # 
    # Result:   A new matrix with deaths by cluster re-allocated to its counties
    #           This matrix is inherently larger than the original
    #           The beef is changed to county FIPS instead of cluster id
    #
    #           Dimensionality is: [sample, day, county]
    """
    from Josh.Alloc_NN.FractionalDeaths import Training
    import keras
    ref_data = pd.read_csv(r'data\us\covid\nyt_us_counties_daily.csv')
    ref_data=ref_data[['fips','date','deaths']]
    ref_data.loc[:,'date']=pd.to_datetime(ref_data.date)
    if alloc_day is not None:
        lst_date = pd.to_datetime(alloc_day)#-np.timedelta64(1,'D')
    else:
        # most recent date in set
        lst_date = ref_data.date.max()
    ##-- Separate panes that are already counties from panes that are clusters
        # Assumes cluster fips are <1000 and counties are >1000
    data_cnty = data[:,:,data[0,0,:] >= 1000]
    data = data[:,:,data[0,0,:] < 1000]

    ##-- Separate input cube into FIPS layer and data layer
    cube_cluster_data = data[0,0,:].astype('int').tolist()    # cast to int
    cube_deaths_data = data[1:,:,:].copy()


    ##-- Load and process cluster data
    cluster_data = pd.read_csv(cluster_ref_fln)

    # Train model if retrain=True
    if retrain:
        modelDir=Training(alloc_day=alloc_day,clust_fln=cluster_ref_fln,numDeaths=numDeaths,numCases=numCases,numMobility=numMobility,lenOutput=lenOutput,remove_sparse=remove_sparse,Patience=Patience,DropoutRate=DropoutRate,modelDir=modelDir)
        # modelDir=Training(alloc_day=alloc_day,clust_fln=cluster_ref_fln,modelDir=modelDir,**kwargs)
    

    
    # load json and create model
    json_file = open(os.path.join(modelDir,"model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.path.join(modelDir,"model.h5"))



    # operate on data to be useful for prediciton
    clusterDF = pd.read_csv(cluster_ref_fln)
    df=pd.read_csv(r'data\us\covid\nyt_us_counties_daily.csv')
    df=df.merge(clusterDF[['fips','cluster']],how='right', on ='fips')
    tmp=df.groupby(['cluster','date'])[['deaths','cases']].sum()
    # df[['clusterDeaths','clusterCases']]=df.groupby(['cluster','date'])['deaths','cases'].transform(np.sum)
    df['clusterDeaths']=df.groupby(['cluster','date'])['deaths'].transform(np.sum)
    df['clusterCases']=df.groupby(['cluster','date'])['cases'].transform(np.sum)
    df['deathsFrac']=df['deaths']/df['clusterDeaths']
    df['casesFrac']=df['cases']/df['clusterCases']
    mobilityDF=pd.read_csv(r'data\us\mobility\DL-us-mobility-daterow.csv')[['fips','date','m50_index']]
    if numMobility>0:
        mobilityDF=pd.read_csv(r'data\us\mobility\DL-us-mobility-daterow.csv')[['fips','date','m50_index']]
        df=df.merge(mobilityDF,how='inner',on=['date','fips'])
    else:
        df.insert(-1,'m50_index',np.nan*np.zeros(len(df)))
    df.loc[:,'date']=pd.to_datetime(df.date)
    df=df[df.date<=lst_date]
    
    lenInput=np.max([numDeaths,numCases,numMobility])
    totLen=lenInput+lenOutput


    # Extract useful columns
    cluster_data = cluster_data[['fips', 'cluster', 'deaths', 'clusterDeaths']]

    # Cast fips and cluster values to int
    cluster_data['fips'] = cluster_data['fips'].astype('int')
    cluster_data['cluster'] = cluster_data['cluster'].astype('int')

    # Extract clusters that are in the data cube
    cluster_data = cluster_data[cluster_data.cluster.isin(cube_cluster_data)]


    ##-- Preallocate new data matrix (use nan to check if anything is unallocated later)
        # Add as many frames as there are fips in the clustering data
        # NOTE: don't include fips from counties in the original data since 
        #  that data gets appended at the end
    data = np.zeros((data.shape[0],data.shape[1],len(cluster_data.index)))*np.nan
    ##-- Iterate through clusters in input cube and allocate predictions
    ind = 0     
    noDataCounter=0


    notTrainedFips=[]
    for i,clust in enumerate(cube_cluster_data):
        # Get rows related to the given state
        clust_rows = cluster_data[cluster_data['cluster'] == clust]
        if len(clust_rows)==0:
            print('Empty Cluster')
            continue
        fips_list=clust_rows.fips.unique()
        Xpred=[]
        trainedFips=[]
        for fip in fips_list:
            DF=df[df.fips==fip]
            DF.fillna(0,inplace=True)
            DF=DF.sort_values(by='date')
            L=len(DF)
            if L<lenInput and L!=0:
                # raise ValueError('not enough data')
                notTrainedFips.append(fip)
            elif  L==0:
                notTrainedFips.append(fip)
            else:
                trainedFips.append(fip)
                input=[ DF.iloc[L-numDeaths:L].deathsFrac.values, 
                            DF.iloc[L-numCases:L].casesFrac.values, 
                            DF.iloc[L-numMobility:L].m50_index.values] 
                Input=[element for subList in input for element in subList]
                Xpred.append(Input)
                # Xpred.extend(Inputs)
        Xpred=np.array(Xpred)
        means=np.loadtxt(os.path.join(modelDir,'means.txt'))
        stds=np.loadtxt(os.path.join(modelDir,'stds.txt'))
        Xpred=(Xpred-means)/stds
        cnty_deaths=model.predict(Xpred).reshape((1,1,)+(len(Xpred),))




        # fips_data=ref_data[ref_data.fips.isin(fips_list)]
        # first_date=lst_date-np.timedelta64(4,'D')
        # fips_data=fips_data[(first_date <= fips_data.date) &  (fips_data.date<= lst_date)].groupby('fips').mean()
        # fips_data['tot_deaths']=fips_data.deaths.transsum()
        # cnty_deaths=fips_data.deaths.values/tot_deaths

        

        
        # Calculate proportion of deaths per county

        

        # cnty_deaths2 = (clust_rows['deaths']/clust_rows['clusterDeaths'].iloc[0]).values

        #-- Allocate deaths by current proporitionality
        # Create matrix of predictions for later multiplication
        data_clust = cube_deaths_data[:,:,i].copy()             # Extract this cluster's prediction frame
        data_clust = data_clust.reshape(data_clust.shape+(1,))  # Add 3rd dimension (to allow tiling)
        data_clust = np.tile(data_clust,(1,1,cnty_deaths.shape[2]))    # Repeat frame for every county in cluster
        # --> data_clust is now: (samples,days,counties-in-cluster)
        # Rotate the cnty_deaths vector into 3rd dimension
        # cnty_deaths = cnty_deaths.reshape((1,1,)+cnty_deaths.shape)
        # Multiply the proportion of deaths vector with the total deaths to allocate
            # np.multiply broadcasts the vector to the correct dimensionality
        cnty_deaths = np.multiply(data_clust, cnty_deaths)
        # --> cnty_deaths is now: (samples,days,counties-in-cluster)

        #-- Create county FIPS beef (will be top layer of cube)
        # Preallocate matrix
        beef = np.zeros((1,)+data_clust.shape[1:])
        # Populate beef with fips codes 
        # beef[0,:,:] = clust_rows['fips'].values
        # beef[0,:,:] = fips_data.index.values
        beef[0,:,:] = trainedFips


        #-- Place result into final matrix
        data[1:,:,ind:ind+len(trainedFips)] = cnty_deaths
        # Add county FIPS beef
        data[0,:,ind:ind+len(trainedFips)] = beef

        #-- Increase index of current matrix allocation
        ind += len(trainedFips)

    # Find number of unallocated panes by counting all nans in cube and dividing by pane size
    numNanPanes=np.count_nonzero(np.isnan(data))/data.shape[0]/data.shape[1]
    # If there are some counties with no data:
        # 
    noDataCounter=len(notTrainedFips)
    if noDataCounter>0:
        if numNanPanes!=noDataCounter:
            raise ValueError('An element of the output matrix is unallocated')
        else:
            data=data[:,:,:-(noDataCounter)]
    else:
        if np.sum(np.isnan(data)) != 0:
            raise ValueError('An element of the output matrix is unallocated')


    ##-- Append original data that was already county-based 
    data = np.dstack((data_cnty, data))

    return data