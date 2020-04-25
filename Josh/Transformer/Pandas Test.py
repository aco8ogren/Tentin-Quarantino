#%% 
import pandas as pd 
import numpy as np 
mob=pd.DataFrame({'fips':np.arange(1,5),'m50':np.random.rand(4)})
mobDate=pd.DataFrame({'fips':np.tile(np.arange(1,5),3),'date':np.sort(np.tile(np.arange(5,8),4))})
# %%
MobDf=pd.merge(mob,mobDate,how='outer',on='fips')

# %%
Deathdf=pd.DataFrame({'death':np.tile(np.arange(1,8),2),'fips':np.sort(np.tile(np.arange(1,3),7))})
Deathdf['date']=Deathdf['death']
Deathdf=Deathdf[['fips','date','death']]
# %%
# df=pd.merge(MobDf,Deathdf, how='left', on='date', how ='right',on='fips')
df=pd.merge(MobDf,Deathdf, how=['left','right'], on=['date','fips'])

# %%
m1=pd.merge(d