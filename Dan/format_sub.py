# %%
import scipy.io as sio
import numpy as np
import os
import pandas as pd
import git

# %% Setup paths
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

os.chdir(homedir)

# %% 
# Constants for analysis
input_fln  = f"{homedir}/ALEX/wkspc.mat"             # input filename  (file to be reformatted)
output_fln = f"{homedir}/Tentin_Qurantino_submit1.csv"           # output filename (file to be created)
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
# Method:   Create matrix with same data but shifted left (column-wise) by one.
#           Append 0's to the first column.
#           Subtract the time-shifted column from the original column.
# Result:   A matrix of the original size but with the daily death count

if isComputeDaily:
    # Create shifted matrix with 0's in left column
    datashift=np.concatenate((np.zeros((data.shape[0],1,data.shape[2])),data[:,:-1,:]),1)
    # Subtract shifted from original. Don't subtract 1st row; beef stays unchanged
    data[1:,:,:]-=datashift[1:,:,:]

# %%
# (OPTIONAL) ALLOCATE STATE PREDICTIONS TO COUNTIES
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
    data = np.zeros((data.shape[0],data.shape[1],len(ref_fips)))#*np.nan

    ##-- Iterate through states in input cube and allocate predcictions
    ind = 0
    # NOTE/TODO: if input data (st_fips_data) does not have a state that the ref_fips does,
        # the stat will not be iterated over and thus we'll still have nans
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


# %%
# CALCULATE QUANTILES and extract data chunks

##-- Extract pertinent information from sample file
perc_list = np.array(samp.columns[1:]).astype(int)
dates = [dat[:10] for dat in np.array(samp['id'])]
fips  = [dat[11:] for dat in np.array(samp['id'])]

##-- Check that enough dates were provided
Ndays = len(np.unique(np.array(dates)))   # use np.unique to remove repeated date instances
if Ndays < data.shape[1]:
    # Input has too many days; assume both end on same day and thus crop input
    data = data[:,-Ndays:,:]
elif Ndays > data.shape[1]:
    raise ValueError('Data provided does not have enough dates (need more columns for each county')

##-- Separate cube into FIPS layer and data layer
data_fips = np.array(['%d' %int(x) for x in data[0,0,:]])    # cast to string so fips datatypes match
data = data[1:,:,:]

##-- Calculate quantiles
quantiles = np.percentile(data,perc_list,0)

# %%
# PERORM ANALYSIS (FASTER method for output file)
#   - Places the data into a temporary numpy matrix row-by-row
#   - Converts the matrix into a pandas dataframe afterwards
#   - This takes much less time than the method in the next cell
#   - Still not ideal though

##-- Re-order data by FIPS for output
# Order FIPS
fips_ind = np.argsort(data_fips)
data_fips = list(data_fips[fips_ind])
# Reorder matrix
data = data[:,:,fips_ind]
quantiles = quantiles[:,:,fips_ind]

##-- Create Final output matrix (with 0s)
output_data = np.zeros((len(fips),len(perc_list)))

##-- Iterate through matrix rows populating them
dcounter = -1        # Counter for what day we are on
fip1     = fips[0]  # first fips; used to determine when day changes
for i in range(0,output_data.shape[0]):
    if fip1 == fips[i]:
        # increase counter since day has changed
        dcounter += 1
    
    # Find the fips instance in the input data
    try:
        fips_ind = data_fips.index(fips[i])
    except ValueError:
        # Skip rows where we have no prediction for the given county
        continue
    
    # Populate ouptput matrix
    output_data[i,:] = quantiles[:,dcounter,fips_ind]
    
# Create pandas dataframe from this matrix (easier for csv saving)
output_df = pd.DataFrame(output_data, columns=samp.columns[1:])
output_df.insert(0,column='id',value=samp['id'])

# %%
# PERFORM ANALYSIS (SLOW method for output file):
#   - Places the data into the sample dataframe row-by-row
#   - Takes ~30ms per row which makes it extremely slow
#   - I'm sure there's a better way to use pandas for this though 
#       so I kept it in for reference.

# ##-- Iterate through sample file re-populating row by row
# dcounter = -1        # Counter for what day we are on
# fip1     = fips[0]  # first fips; used to determine when day changes
# for i in range(len(fips)):
#     if fip1 == fips[i]:
#         # increase counter since day has increased
#         dcounter += 1

#     # Find the fips instance in the input data
#     try:
#         fips_ind = data_fips.index(fips[i])
#     except ValueError:
#         continue

#     samp.at[i,'10':'90'] = quantiles[:,dcounter,fips_ind]

# Alias samp to ouput_df for saving in next cell
# output_df = samp

# %%
# SAVE RESULTS
output_df.to_csv(output_fln,index=False, float_format='%0.3f')


# %%
