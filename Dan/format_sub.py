# %%
import scipy.io as sio
import numpy as np
import os
import pandas as pd
# import git
import Dan.cube_formatter as cf

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

# %% Define formatting function
def format_file_for_evaluation(input_fln,output_fln,isAllocCounties = True,isComputeDaily = True, alloc_day=None, num_alloc_days=5):
    # if alloc_day is not None:
    #     alloc_day=
# %-% Setup paths
    HomeDIR='Tentin-Quarantino'
    wd=os.path.dirname(os.path.realpath(__file__))
    DIR=wd[:wd.find(HomeDIR)+len(HomeDIR)]
    os.chdir(DIR)


    homedir = DIR

    os.chdir(homedir)

    # %-% 
    # Constants for analysis
    # input_fln  = f"{homedir}/Dan/train_til_4_22.mat"             # input filename  (file to be reformatted)
    # output_fln = f"{homedir}/Dan/train_til_4_22.csv"           # output filename (file to be created)
    # sample file (TA-provided reference file)
    sampfile = f"{homedir}/sample_submission.csv"
    # current county deaths reference file (used when isAllocCounties=True)
    county_ref_fln = f"{homedir}/data/us/covid/nyt_us_counties_daily.csv"


    # NOTE: This assumes that the clustering file's name is the same as the input file but with _cluster.csv
    cluster_ref_fln = os.path.splitext(input_fln)[0] + '_clusters.csv'


    # Control flags
    # isAllocCounties = True          # Flag to distribue cluster deaths amongst counties
    # isComputeDaily = True           # Flag to translate cummulative data to daily counts



    # %%
    # LOAD DATA

    ##-- Read files
    data = cf.read_cube(input_fln)
    samp = pd.read_csv(sampfile)

    # %%
    # (OPTIONAL) Translate a cummulative data set to daily death count

    if isComputeDaily:
        data = cf.calc_daily(data)

    # %%
    # (OPTIONAL) ALLOCATE STATE PREDICTIONS TO COUNTIES


    if isAllocCounties:
        data = cf.alloc_fromCluster(data, cluster_ref_fln,alloc_day=alloc_day, num_alloc_days=num_alloc_days)

    # %%
    # CALCULATE QUANTILES and extract data chunks (split into fips and percentiles)

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
