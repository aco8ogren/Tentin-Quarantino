# %% Imports
import os
import sys

import pandas as pd
import numpy as np

# %% Setup paths
HomeDIR='Tentin-Quarantino'
wd=os.path.dirname(os.path.realpath(__file__))
DIR=wd[:wd.find(HomeDIR)+len(HomeDIR)]
os.chdir(DIR)

homedir = DIR
datadir = f"{homedir}/data/us/"

sys.path.append(os.getcwd())

# %% load mobility data
mobility_df = pd.read_csv(datadir + 'mobility/DL-us-m50_index.csv')

# %%
#-- Gather necessary counties from mobility data
# cast fips to integers
mobility_df = mobility_df[mobility_df['fips'].notna()]      # remove entries without fips (us aggregate)
mobility_df['fips'] = mobility_df['fips'].astype(int)       # cast to int

# Deal with New York City
nyc_fips = ['36061', '36005', '36047', '36081', '36085']
# Average mobility data for these counties
nyc_avg = mobility_df.loc[mobility_df.fips.isin(nyc_fips),'2020-03-01':].mean()
# Put in as values for 36061
mobility_df.loc[mobility_df['fips'] == 36061,'2020-03-01':] = nyc_avg.values

# Keep only relavent counties
#mobility_df = mobility_df[mobility_df.fips.isin([36061, 1073, 56035, 6037])]

# Drop fips < 1000 (ie. non-county id's)
mobility_df = mobility_df[mobility_df['fips'] > 1000]

# %% Convert mobility data column headers to date_processed format

global_dayzero = pd.to_datetime('2020 Jan 21')

date_dict = dict()
for col in list(mobility_df.columns):
    col_dt = pd.to_datetime(col,errors = 'coerce')
    if not (isinstance(col_dt,pd._libs.tslibs.nattype.NaTType)): # check if col_dt could be converted. NaT means "Not a Timestamp"    
        # date_dict[col] = (col_dt - day_zero) / np.timedelta64(1, 'D')
        date_dict[col] = (col_dt - global_dayzero) / np.timedelta64(1, 'D')

temp_dict = dict()
temp_dict['admin1'] = 'state'

mobility_df = mobility_df.rename(columns = temp_dict)
mobility_df = mobility_df.rename(columns = date_dict)


# %% Check how many missing elements there are

# Total number of counties in the df
np.unique(mobility_df['fips']).shape

# Number of rows with missing elements
mobility_df[mobility_df.isnull().any(axis=1)].shape

# Number of rows without any missing elements
mobility_df[~mobility_df.isnull().any(axis=1)].shape

# Add column with number of missing elements in each 
mobility_df.insert(5, 'nulls', mobility_df.isnull().sum(axis=1))