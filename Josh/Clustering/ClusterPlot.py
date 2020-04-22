#%%
import git
import numpy as np
import matplotlib.pyplot as plt
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)
CD=np.loadtxt('Josh/Clustering/ClusterDat_refined.txt')
inds=CD[:,1].argsort()
plt.figure(1)
plt.plot(CD[inds,1],CD[inds,2])
plt.show()



# %%