#%% Dan ignore this. This is for my testing.
import numpy as np
datas=np.array([range(0,i*5,i) for i in np.arange(1,11)]).astype(float)
datas=datas.reshape(datas.shape+(1,))
data=np.concatenate((datas,datas,datas,datas),2)


datashift=np.concatenate((np.zeros((data.shape[0],1,data.shape[2])),data[:,:-1,:]),1)
data-=datashift




# %%
