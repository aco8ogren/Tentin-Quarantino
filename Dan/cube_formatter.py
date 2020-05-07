import os
import numpy as np
import scipy.io as sio
import pandas as pd

# Library for performing basic operations on our data cubes
# Allows for consistency accross various scripts that perform the same operations


def read_cube(filename):
    """
    # Function to read in data cube (.npy OR .mat) and return a numpy cube
    # 
    # Arguments: 
    #   - filename  (string) filename to load
    #                 *Must be either .npy or .mat*
    #                 format for input cube:
    #                 row=sample, col=date, pane=county
    #
    # Output:
    #   - data      a numpy cube
    # 
    # *** First row should be beef (ie:)
    #   FIPS | 4022020 | 4032020 | ...
    """

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

def calc_daily(data):
    """
    # Function to convert a cube of data from cumulative deaths to daily deaths
    # 
    # Arguments: 
    #   - data      cube with cumulative death counts
    #                 format for input cube:
    #                 row=sample, col=date, pane=county
    #
    # Output:
    #   - data      a numpy cube with daily deaths
    # 
    # *** First row should be beef (ie:)
    #   FIPS | 4022020 | 4032020 | ...
    #
    # -----------------------------------------------------
    #
    # Method:   Create matrix with same data but shifted left (column-wise) by one.
    #           Append 0's to the first column.
    #           Subtract the time-shifted column from the original column.
    # Result:   A matrix of the original size but with the daily death count
    """
    
    # Create shifted matrix with 0's in left column
    datashift=np.concatenate((np.zeros((data.shape[0],1,data.shape[2])),data[:,:-1,:]),1)
    # Subtract shifted from original. Don't subtract 1st row; beef stays unchanged
    data[1:,:,:]-=datashift[1:,:,:]

    return data

def alloc_counties(data, county_ref_fln, alloc_day=None):
    """
    # Function to allocate state-wide deaths amongst counties
    # 
    # Arguments:
    #   - data      cube with daily death counts
    #                 format for input cube:
    #                 row=sample, col=date, pane=county
    #
    #   - county_ref_fln 
    #               filename of reference file to use for allocation
    #   
    #   - alloc_day Day to use to get proportions for allocations
    #               if None: uses most recent value in ref file
    #               else: use the date provide [format='YYYY-MM-DD']
    #
    # Output:
    #   - data      a numpy cube with daily deaths by county
    # 
    # *** First row should be beef (ie:)
    #   FIPS | 4022020 | 4032020 | ...
    #
    # -----------------------------------------------------
    #
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
    """
    ##-- Separate input cube into state FIPS layer and data layer
    st_fips_data = ['%d' %int(x) for x in data[0,0,:]]    # cast to string so fips datatypes match
    st_deaths_data = data[1:,:,:].copy()

    ##-- Load the reference county file and extract pertinent data
    ref_data = pd.read_csv(county_ref_fln)

    if alloc_day is not None:
        lst_date = alloc_day
    else:
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
        # the state will not be iterated over and thus we'll still have nans
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

    return data

