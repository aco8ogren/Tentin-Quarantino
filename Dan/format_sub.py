# %%
import scipy.io as sio
import numpy as np
import os
import datetime as dt
import pandas as pd
import git

# %% Setup paths
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

os.chdir(homedir)

# %% 
# Constants for analysis
input_fln  = f"{homedir}/Dan/test.mat"             # input filename  (file to be reformatted)
output_fln = f"{homedir}/Dan/outtest.csv"           # output filename (file to be created)
# sample file (TA-provided reference file)
sampfile = f"{homedir}/sample_submission.csv"
# current county deaths reference file (used when isAllocCounties=True)
county_ref_fln = f"{homedir}/data/us/covid/nyt_us_counties_daily.csv"

# Control flags
isAllocCounties = True          # Flag to distribue state deaths amongst counties
isComputeDaily = True           # Flag to translate cummulative data to daily counts

# %%
def read_file(filename):
    # Function to format results into a submittable csv file
    # 
    # Arguments: 
    #   - filename  (string) filename to load
    #                 *Must be either .npy or .mat*
    #                 format for input cube:
    #                 row=sample, col=date, pane=county
    #
    # Output:
    #   - a csv file with the appropriate format for class submission
    # 
    # *** First row should be:
    #   FIPS | 4022020 | 4032020 | ...

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

# %%
# LOAD DATA

##-- Read files
data = read_file(input_fln)
samp = pd.read_csv(sampfile)

# %%
# (OPTIONAL) Translate a cummulative data set to daily death count
if isComputeDaily:
    datashift=np.concatenate((np.zeros((data.shape[0],1,data.shape[2])),data[:,:-1,:]),1)
    data-=datashift

# %%
# (OPTIONAL) ALLOCATE STATE PREDICTIONS TO COUNTIES
if isAllocCounties:
    ##-- Separate input cube into state FIPS layer and data layer
    st_fips_data = ['%d' %int(x) for x in data[0,0,:]]    # cast to string so fips datatypes match
    st_deaths_data = data[1:,:,:].copy()

    ##-- Load the reference county file and extract pertinent data
    ref_data = pd.read_csv(county_ref_fln)

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
    data = np.zeros((data.shape[0],date.shape[1],len(ref_fips)))*np.nan

    ##-- Iterate through states in input cube and allocate predcictions
    ind = 0
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
        beef[0,0,:] = np.array(st_rows['fips'])

        #-- Place result into final matrix
        # Place predictions
        data[1:,:,ind:ind+len(st_rows)] = tmp
        # Add county FIPS beef
        data[[0],:,:] = beef


        #-- Increase index of current matrix allocation
        ind += len(st_rows)

    if np.sum(np.isnan(data)) != 0:
        raise ValueError('An element of the output matrix is unallocated')





# %%
# PERFORM ANALYSIS
##-- Extract pertinent information from sample file
perc_list = np.array(samp.columns[1:]).astype(int)
dates = [dat[:10] for dat in np.array(samp['id'])]
fips  = [dat[11:] for dat in np.array(samp['id'])]

##-- Check that enough dates were provided
Ndays = len(np.unique(np.array((dates))))   # use np.unique to remove repeated date instances
if Ndays != data.shape[1]:
    raise ValueError('Data provided does not have enough dates (need more columns for each county)')

##-- Separate cube into FIPS layer and data layer
data_fips = ['%d' %int(x) for x in data[0,0,:]]    # cast to string so fips datatypes match
data = data[1:,:,:]

##-- Calculate quantiles
quantiles = np.percentile(data,perc_list,0)

##-- Iterate through sample file re-populating row by row
dcounter = -1        # Counter for what day we are on
fip1     = fips[0]  # first fips; used to determine when day changes
for i in range(len(fips)):
    if fip1 == fips[i]:
        # increase counter since day has increased
        dcounter += 1

    # Find the fips instance in the input data
    try:
        fips_ind = data_fips.index(fips[i])
    except ValueError:
        continue

    samp.at[i,'10':'90'] = quantiles[:,dcounter,fips_ind]

# %%
# SAVE RESULTS
samp.to_csv(output_fln,index=False, float_format='%0.3f')


# %%
