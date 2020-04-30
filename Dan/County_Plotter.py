import pandas as pd
import numpy as np 
import git 
import matplotlib.pyplot as plt

#-- Setup paths
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

os.chdir(homedir)

#-- Control parameters
# Top N counties to plot with the most deaths
plotN = 20 

#-- Files to utilize
# Submission-style file to plot (model)
csv_model   = 'Tentin_Quarantino_submit2.csv'#'Dan\\train_til_today.csv'
# Reference file to treat as "true" death counts
csv_true    = 'data\\us\\covid\\nyt_us_counties_daily.csv'


#-- Define functions
def get_date(x):
    # Get the date from the id column of a submission-style file
    return '-'.join(x.split('-')[:3])
def get_fips(x):
    # Get the fips from the id column of a submission-style file
    return x.split('-')[-1]

#-- Read and format data
# Pre-process "true" data
true_df = pd.read_csv(csv_true)
true_df['fips'] = true_df['fips'].astype(int)
true_df['id'] = true_df['date'] + '-' + true_df['fips'].astype(str)

# Pre-process model data
model_df = pd.read_csv(csv_model)
model_df['date'] = model_df['id'].apply(get_date)
model_df['fips'] = model_df['id'].apply(get_fips).astype('int')
# Sort by county then day to get how each county changes day by day
model_df.sort_values(by=['fips','date'],inplace=True)
# Find peak deaths/day for each county and sort by ascending order
peaks = model_df.sort_values('90', ascending=False).drop_duplicates(['fips'])
            # OLD METHOD OF DOING THE ABOVE LINE
            # # Find peak deaths/day for each county
            # peaks = model_df.groupby('fips').max()
            # # Sort peaks in descending order 
            # peaks.sort_values(by='90',inplace=True, ascending=False)


#-- Extract counties to plot and further format
# Get fips of counties to consider
counties_to_plot = list(peaks['fips'][:plotN])
# Pull relevant data from the df's
true_df = true_df[true_df.fips.isin(counties_to_plot)]
model_df = model_df[model_df.fips.isin(counties_to_plot)]

# Add column of dates in datetime format
true_df['dateDT'] = pd.to_datetime(true_df['date'].values)
model_df['dateDT'] = pd.to_datetime(model_df['date'].values)

#-- Plot 
for ind, cnty in enumerate(counties_to_plot):
    # Pull just relevant county
    cnty_true_df = true_df[true_df['fips'] == cnty]
    cnty_model_df = model_df[model_df['fips'] == cnty]
    # Ensure they are chronolically sorted
    cnty_true_df.sort_values(by=['dateDT'],inplace=True)
    cnty_model_df.sort_values(by=['dateDT'],inplace=True)
        
    # Keep only data after first date existing in both datesets
    # day0 = max(cnty_model_df['dateDT'].iloc[0], cnty_true_df['dateDT'].iloc[0])
    # cnty_model_df = cnty_model_df[cnty_model_df['dateDT'] >= day0]
    # cnty_true_df = cnty_true_df[cnty_true_df['dateDT'] >= day0]

    # Create column with days since first datapoint
    day0 = min(cnty_model_df['dateDT'].iloc[0], cnty_true_df['dateDT'].iloc[0])
    cnty_true_df['rel_date'] = (cnty_true_df['dateDT'] - day0)/np.timedelta64(1,'D')
    cnty_model_df['rel_date'] = (cnty_model_df['dateDT'] - day0)/np.timedelta64(1,'D')

    # Create new figure for the plot
    plt.figure(ind)

    # Plot
    # plt.scatter(np.arange(len(cnty_true_df)), 'deaths', data=cnty_true_df)
    # plt.scatter(np.arange(len(cnty_model_df)), '50', data=cnty_model_df)
    plt.scatter('rel_date', 'deaths', data=cnty_true_df)
    plt.scatter('rel_date', '50', data=cnty_model_df)








