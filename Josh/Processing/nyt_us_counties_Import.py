def us_counties_Data2Dict(RemoveEmptyFips=False,RemoveUnknownCounties=False):
    # Function to import nyt_us_counties.csv data into a dictionary
    import numpy as np 
    import sys
    import pickle
    
    sys.path.append('Josh/Processing/')
    from dateProcess import DateProcess
    File=open('Josh/Processing/Processed Data/GeoDict.pkl','rb')
    GeoDict=pickle.load(File)
    File.close()
    

    data=np.loadtxt('data/us/covid/nyt_us_counties.csv',dtype=str,delimiter=',')
    # Remove data from unknown counties
    FipsCol=np.nonzero(data[0]=='fips')[0][0]
    if RemoveEmptyFips:
        data=data[data[:,FipsCol]!='']
    else:
        data[data[:,FipsCol]=='',FipsCol]='0'

    if RemoveUnknownCounties:
        CountyCol=np.nonzero(data[0]=='county')[0][0]
        data=data[data[:,CountyCol]!='Unknown']
    
    DataDict={data[0][i]:data[1:,i] for i in range(data.shape[1])}

    # Keys for variables. If changed an error is thrown and must be updated manually
    Keys=['date','county','state','fips','cases','deaths']
    for key in Keys:
        if key not in DataDict:
            raise ValueError("Column Headers changed; update keys in Josh/Processing/nyt_us_counties_Import.py")

    # Convert fips data from str to int then into coordinate pairs, then normailize coordinate data
    DataDict['fips']=DataDict['fips'].astype(int)
    for fip in DataDict['fips']:
        if fip not in GeoDict:
            GeoDict[fip]=np.array([np.nan,np.nan])
    coords=np.array([GeoDict[DataDict['fips'][i]] for i in range(len(DataDict['fips']))])
        # means=coords.mean(0)
        # stds=coords.std(0)
        # coords=(coords-means)/stds
        # np.savetxt(CoordSavePath+'Coord_Mean_Std.txt',[means,stds])
    DataDict['coords']=coords
    # Convert Dates into day since January 1st
    DataDict['day']=np.array([[DateProcess(DataDict['date'][i])] for i in range(len(DataDict['date']))])
    DataDict['cases']=DataDict['cases'].astype(int)
    DataDict['deaths']=DataDict['deaths'].astype(int)
    return DataDict
