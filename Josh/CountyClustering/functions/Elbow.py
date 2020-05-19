#%%
import numpy as np 
import git
import matplotlib.pyplot as plt 

repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)

Data=np.loadtxt('Josh/Clustering/ClusterDataSqrd.txt')[:,1:]
data=Data.copy
p0=Data[0]
Data=Data-p0
p1=Data[-1]
p=(p1/np.linalg.norm(p1))[np.newaxis].T


DistV=Data-(Data@p)*p.T
Dist=np.linalg.norm(DistV,axis=1)
OptInd=Dist.argmax()

DistLine=np.array([Data[OptInd,:],Data[OptInd,:]-DistV[OptInd,:]])
#%%
Data+=p0
# p1+=p0
DistLine+=p0
plt.figure(1)
plt.plot(Data[:,0],Data[:,1],'k')
# plt.plot([p0[0],p1[0]],[p0[1],p1[1]],'b')
# plt.plot(DistLine[:,0],DistLine[:,1],'r')
plt.plot(Data[OptInd,0],Data[OptInd,1],'ro',label='Optimal K = %i'%Data[OptInd,0])
plt.legend()
plt.title('K-Means Clustering:\nMean Square Distance To Cluster Center vs. K')
plt.xlabel('K')
plt.ylabel('Square Distance')
# plt.show()
# plt.gca().set_aspect('equal')
plt.savefig('Josh/Clustering/ElbowPlotNoLines.svg')
# plt.savefig('Josh/Clustering/ElbowPlot.png')







# %%
