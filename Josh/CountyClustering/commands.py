from Josh.CountyClustering.CountyCluster import CountyCluster
import pandas as pd
import numpy as np
df=pd.read_csv('data/us/covid/nyt_us_counties.csv')
fipsList=df.fips.unique()
fipsList=fipsList[~np.isnan(fipsList)]

CountyCluster(fipsList)

fipsList=np.array(fipsList)
fipsList.sort()
fipstr='.'.join(['%i'%i for i in fipsList])
# fipstr=''.join(np.array(fipsList).sort().astype(int).astype(str))
fileRef=pd.read_csv('Josh/CountyClustering/FileReference.csv')



FileRef=pd.DataFrame({'file':[],'fips':[]})
FileRef.to_csv('Josh/CountyClustering/FileReference.csv',index=False)

from Josh.CountyClustering.CountyCluster import CountyCluster
import pandas as pd
import numpy as np
df=pd.read_csv('data/us/covid/nyt_us_counties.csv')
fipsList=df.fips.unique()
fipsList=fipsList[~np.isnan(fipsList)]

f=CountyCluster(fipsList[:400])