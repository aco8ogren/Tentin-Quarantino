import pandas as pd
import numpy as np 
import git 
import os
import sys 
import bokeh.io
import bokeh.application
import bokeh.application.handlers
import bokeh.models
import bokeh.plotting as bkp
from bokeh.models import Span
import holoviews as hv
from pathlib import Path
# from bokeh.io import export_png

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
    # Set to -1 to plot all
plotN = 20
shift = 20
# Data Manipulation flags (should match those used in creating submission file)
isAllocCounties = True          # Flag to distribue state deaths amongst counties
isComputeDaily = False           # Flag to translate cummulative data to daily counts
#- Plot-type control flags
isStateWide = False          # Flag to plot state-wise data (will use nyt_states file for true_df)
                            #   The raw cube won't be affected so make sure it is also state-wise data
                            #   AND cumulative since there is only cumulative nyt_us_states data
isCumul     = True          # Flag to denote that the plot should be cumulative, not daily deaths
                            # ** Only affects county-level data since state-wide is implicitly cumulative
                            #   This sets which county-wide nyt file is used and sets the plot y-axis label
# Key days (should match those used in creating the cube)
global_dayzero = pd.to_datetime('2020 Jan 21')
# Day until which model was trained (train_til in epid model)
    # Leave as None to not display a boundary
boundary = '2020 May 10'
# Day to use for allocating to counties
    # Leave as None to use most recent date
    # OR use '2020-04-23' format to allocate based on proportions from that day
alloc_day = '2020-05-10'
# Flag to choose whether to save .svg of figures
is_saveSVG = False
# Filename (including path) for saving .svg files when is_saveSVG=True
    # county, state, and fips will be appended to the name to differentiate plots
svg_flm = 'Dan/MidtermFigs/CountyWideDaily2/'


#-- Files to utilize
# Filename for cube of model data
    # should be (row=sample, col=day, pane=state) with state FIPS as beef in row1
mat_model   = 'Alex\\PracticeOutputs\\fresh.mat'#'Dan\\train_til_today.csv'
# Reference file to treat as "true" death counts 
csv_true    = 'data\\us\\covid\\nyt_us_counties_daily.csv'  # daily county counts (also used for allocating deaths when req.)
csv_ST_true = 'data\\us\\covid\\nyt_us_states.csv'          # this is cumulative ONLY; no _daily version exists
csv_CT_cumul_true = 'data\\us\\covid\\nyt_us_counties.csv'  # county cumulative counts
# reference file for clustering df
    # This assignment as done below assumes that the right file just has _clusters.csv appended.
    # You can enter the actual path manually if you'd like
cluster_ref_fln=os.path.splitext(mat_model)[0] + '_clusters.csv'



#-- Read and format true data to have correct columns
# Read correct file for requested setup
if isStateWide:
    # Plotting state-wide so use nyt state file (implicitly cumulative)
    true_df = pd.read_csv(csv_ST_true)
else:
    if isCumul:
        # plotting cumulative county-wide so pull this file
        true_df = pd.read_csv(csv_CT_cumul_true)
    else:
        # plotting daily county-wide so pull this file
        true_df = pd.read_csv(csv_true)
# The nyt_us_counties.csv file is SUPER FLAWED so we need to fix this:
    # - has some empty values in the fips column cousing prob. with .astype(int) 
    # - Straight up doesn't have fips entry for NYC so need to hardcode its fips
if (not isStateWide) and isCumul:
    # Reading in problematic file. 
    # Replace empty value on NYC with 36061
    true_df.loc[true_df.county=='New York City', 'fips'] = 36061
    # Remove rows with nans from the df (these are the counties we don't care about)
    true_df = true_df[true_df['fips'].notna()]
# Reformat some columns
true_df['fips'] = true_df['fips'].astype(int)
true_df['id'] = true_df['date'] + '-' + true_df['fips'].astype(str)


#-- Read and format model data to county-based
# read raw cube from epid. code
model_cube = cf.read_cube(mat_model)
# format to county-based in same way as format_sub
if isComputeDaily:
    model_cube = cf.calc_daily(model_cube)
if isAllocCounties:
    model_cube = cf.alloc_fromCluster(model_cube, cluster_ref_fln, alloc_day=alloc_day)


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
peak_inds = peak_inds[shift:plotN+shift]
# Extract the resulting counties
    # results will be implicitly sorted due to use of argsort
model_quants = model_quants[:,:,peak_inds]      # Get quantiles
model_fips   = model_cube[0,0,peak_inds]        # Get fips ID's


#-- Extract the same counties from the true data and add column with datetime date
# Pull desired counties from true_df
true_df = true_df[true_df.fips.isin(model_fips)]
# Add column of dates in datetime format
true_df['dateDT'] = pd.to_datetime(true_df['date'].values)

if isAllocCounties:
    #-- Read in cluster-to-fips translation (used for showing which counties were clustered)
    # Load cluster data
    fips_to_clst = pd.read_csv(cluster_ref_fln)
    # Extract useful columns
    fips_to_clst = fips_to_clst[['fips', 'cluster']]
    # Cast fips and cluster values to int
    fips_to_clst['fips'] = fips_to_clst['fips'].astype('int')
    fips_to_clst['cluster'] = fips_to_clst['cluster'].astype('int')
    # Cast to pandas series
    fips_to_clst = pd.Series(fips_to_clst.set_index('fips')['cluster'])
else:
    # Define empty list so that "in" check later doesn't cause errors
    fips_to_clst = []

#-- Create directory for output .svg files if necessary
if is_saveSVG:
    # Append sample filename just to get proper path
    tmp_flm = '%sstate_county_fips.svg'%svg_flm
    # Create directory if necessary
    Path(tmp_flm).parent.mkdir(parents=True, exist_ok=True)

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

    # Format title for state vs. county plots
    if isStateWide:
        # Don't add county item since it's not pertinent
        ptit = 'SEIIRD+Q Model: %s (%d)'%(cnty_true_df['state'].iloc[0], cnty)
    else:
        # Include county in title
        ptit = 'SEIIRD+Q Model: %s, %s (%d)'%(cnty_true_df['county'].iloc[0],cnty_true_df['state'].iloc[0], cnty)

    if cnty in fips_to_clst:
        # Add cluster ID when the county was clustered
        ptit += ' [Cluster %d]'%fips_to_clst[cnty]

    # Format y-axis label for cumulative vs. daily plots
    if isCumul or isStateWide:
        # NOTE: statewide is implicitly cumulative
        # Set y-axis label to show cumulative counts
        ylab = '# deaths total'
    else:
        # Set y-axis label to show deaths/day
        ylab = '# deaths/day'

    # Create figure for the plot
    p = bkp.figure( plot_width=600,
                    plot_height=400,
                    title = ptit,
                    x_axis_label = 't (days since %s)'%global_dayzero.date(),
                    y_axis_label = ylab)

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
    # fn = "Alex/conv/" + ptit.replace('SEIIRD+Q Model:','')
    # export_png(p,filename=fn)


    # Save output figures if desired
    if is_saveSVG:
        p.output_backend = "svg"
        # Format filename for state vs. county plots
        if isStateWide:
            suffix = ('%s_%d.svg'%(cnty_true_df['state'].iloc[0],cnty)).replace(' ','')
        else:
            suffix = ('%s_%s_%d.svg'%(cnty_true_df['state'].iloc[0],cnty_true_df['county'].iloc[0],cnty)).replace(' ','')
        bokeh.io.export_svgs(p, filename= svg_flm + suffix)
