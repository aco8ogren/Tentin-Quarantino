# Script for analyzing the "data/us/geolocation/neighborcounties.csv" file

# %%
import numpy as np
import git

# %% Setup paths
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

os.chdir(homedir)

# %% 

# Load data
dat = np.genfromtxt(f"{homedir}/data/us/geolocation/neighborcounties.csv",delimiter=',',skip_header=1)

# Use unique_counts flag to see how many times a county is repeated in the first column
    # Assuming the first column (orgfips) is sorted, this will tell us how many times 
    # a county had to be repeated, ie. how many neighbors it had
_, cts = np.unique(dat[:,1], return_counts=True)

print('Mean number of adjacent counties: %f'%np.mean(cts))


print('Distribution_____:')
print('Adj. counties | Number of instances')
dist = np.unique(np.sort(cts),return_counts=True)
for i in range(len(dist[0])):
    print('%2d | %5d'%(dist[0][i],dist[1][i]))


# %%
