import pandas as pd
import numpy as np 
import git 
import matplotlib.pyplot as plt
import os
import sys 
import bokeh.io
import bokeh.application
import bokeh.application.handlers
import bokeh.models
import bokeh.plotting as bkp
from bokeh.models import Span
import holoviews as hv


#-- Setup paths
# Get parent directory using git
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
# Change working directory to parent directory
os.chdir(homedir)
# Add 'Dan' directory to the search path for imports
sys.path.append('Dan')
# Import our custom cube managing functions
import cube_formatter as cf


#-- Setup bokeh
bokeh.io.output_notebook()
hv.extension('bokeh')


#-- Control parameters
# Top N counties to plot with the most deaths
plotN = 10 
# Control flags (should match those used in creating submission file)
isAllocCounties = True          # Flag to distribue state deaths amongst counties
isComputeDaily = False           # Flag to translate cummulative data to daily counts
# Key days (should match those used in creating the cube)
global_dayzero = pd.to_datetime('2020 Jan 21')
# Day until which model was trained (train_til in epid model)
    # Leave as None to not display a boundary
boundary = '2020 April 24'


#-- Files to utilize
# Filename for cube of model data
    # should be (row=sample, col=day, pane=state) with state FIPS as beef in row1
mat_model   = 'Dan/train_til_4_24.mat'#'Dan\\train_til_today.csv'
# Reference file to treat as "true" death counts (also used for allocating deaths when isAllocCounties=True)
csv_true    = 'data\\us\\covid\\nyt_us_counties_daily.csv'


#-- Read and format true data to have correct columns
true_df = pd.read_csv(csv_true)
true_df['fips'] = true_df['fips'].astype(int)
true_df['id'] = true_df['date'] + '-' + true_df['fips'].astype(str)


#-- Read and format model data to county-based
# read raw cube from epid. code
model_cube = cf.read_cube(mat_model)
# format to county-based in same way as format_sub
if isComputeDaily:
    model_cube = cf.calc_daily(model_cube)
if isAllocCounties:
    model_cube = cf.alloc_counties(model_cube, csv_true)


#-- Calculate quantiles for all modeled counties
# Quantiles to consider
perc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# Calculate along each column ignoring the first row of beef
model_quants = np.percentile(model_cube[1:,:,:],perc_list,0)
    # model_quants now has 9 rows, one for each of the quantiles requested
    #   The cols and panes are the same format as model_cube


#-- Order model counties by peak deaths/day predicted AND extract counties for plotting from the cube
# Get maximum deaths/day ever hit by each county
    # Use 4th row of model_quants to use the 50th percentile (ie. the central prediction)
peak_daily_deaths = np.max(model_quants[4,:,:],0)
# Get indices of sorted (descending) vector 
    # NOTE: argsort only works in ascdending order so use [::-1] to reverse
peak_inds = np.argsort(peak_daily_deaths)[::-1]
# Take the largest plotN counties (since these are the only ones requested by the user)
peak_inds = peak_inds[:plotN]
# Extract the resulting counties
    # results will be implicitly sorted due to use of argsort
model_quants = model_quants[:,:,peak_inds]      # Get quantiles
model_fips   = model_cube[0,0,peak_inds]        # Get fips ID's


#-- Extract the same counties from the true data and add column with datetime date
# Pull desired counties from true_df
true_df = true_df[true_df.fips.isin(model_fips)]
# Add column of dates in datetime format
true_df['dateDT'] = pd.to_datetime(true_df['date'].values)

for ind, cnty in enumerate(model_fips):
    # Pull just the relevant county
    cnty_true_df = true_df[true_df['fips'] == cnty]
    cnty_model   = model_quants[:,:,ind]
    # Ensure true_df is chronolically sorted
    cnty_true_df.sort_values(by=['dateDT'],inplace=True)
        
    # Create column with days since global_dayzero (to have same reference point for both datasets)
    cnty_true_df['rel_date'] = (cnty_true_df['dateDT'] - global_dayzero)/np.timedelta64(1,'D')

    # Create time axes
    t_true = cnty_true_df['rel_date'].values
    t_model = np.arange(cnty_model.shape[1])

    # Create figure for the plot
    p = bkp.figure( plot_width=600,
                    plot_height=400,
                    title = 'SEIIRD+Q Model: %s, %s (%d)'%(cnty_true_df['county'].iloc[0],cnty_true_df['state'].iloc[0], cnty) ,
                    x_axis_label = 't (days since %s)'%global_dayzero.date(),
                    y_axis_label = '# deaths/day')

    # CONSIDER FLIPPING THE ORDER OF QUANTILES TO SEE IF IT FIXES THE PLOTTING
    # Plot uncertainty regions
    for i in range(4):
        p.varea(x=t_model, y1=cnty_model[i,:], y2=cnty_model[-i-1,:], color='black', fill_alpha=perc_list[i]/100)  

    # Plot 50th percentile line
    p.line(t_model, cnty_model[4,:], color = 'black', line_width = 1)

    # Plot true deaths
    p.circle(t_true, cnty_true_df['deaths'], color ='black')

    # Apply training boundary if desired
    if boundary is not None:
        bd_day = (pd.to_datetime(boundary)-global_dayzero)/np.timedelta64(1, 'D')
        vline = Span(location=bd_day, dimension='height', line_color='black', line_width=2)
        p.renderers.extend([vline])

    # Show plot
    bokeh.io.show(p)