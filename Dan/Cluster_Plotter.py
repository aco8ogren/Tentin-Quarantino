# %%
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
# Top N clusters to plot with the most deaths
    # Set to -1 to plot all
plotN = 20
# Cluster fips to plot
    # If isShowAllocations=True, all counties from the following cluster will be plotted
clst2Show = 10              # "FIPS" of cluster to show
# Data Manipulation flags (should match those used in creating submission file)
isComputeDaily = False           # Flag to translate cummulative data to daily counts
#- Plot-type control flags
isCumul     = True          # Flag to denote that the plot should be cumulative, not daily deaths
# NOTE: the following two flags are independent of each other (ie. you can run either, or, or both)
isShowClusters = True       # Flag to denote that each cluster should be plotted on its own
isShowAllocations = True    # Flag to denote that the counties within clst2Show should be shown
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
mat_model   = 'clustering.mat'#'Dan\\train_til_today.csv'
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
if isCumul:
    # plotting cumulative county-wide so pull this file
    true_df = pd.read_csv(csv_CT_cumul_true)
    # The nyt_us_counties.csv file is SUPER FLAWED so we need to fix this:
        # - has some empty values in the fips column cousing prob. with .astype(int) 
        # - Straight up doesn't have fips entry for NYC so need to hardcode its fips
    # Replace empty value on NYC with 36061
    true_df.loc[true_df.county=='New York City', 'fips'] = 36061
    # Remove rows with nans from the df (these are the counties we don't care about)
    true_df = true_df[true_df['fips'].notna()]
else:
    # plotting daily county-wide so pull this file
    true_df = pd.read_csv(csv_true)
# Reformat some columns
true_df['fips'] = true_df['fips'].astype(int)
true_df['id'] = true_df['date'] + '-' + true_df['fips'].astype(str)
# Add column of dates in datetime format
true_df['dateDT'] = pd.to_datetime(true_df['date'].values)


#-- Read model data and compute daily if requested
# read raw cube from epid. code
full_cube = cf.read_cube(mat_model)
# compute daily values
if isComputeDaily:
    full_cube = cf.calc_daily(full_cube)


#-- Remove panes that are not related to clusters (this code only deals with clusters)
    # Assumes cluster fips are <1000 and counties are >1000
clst_cube = full_cube[:,:,full_cube[0,0,:] < 1000]



###################################
#### Plot Clusters ################
###################################

#-- Calculate quantiles
# Quantiles to consider
perc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# Calculate along each column ignoring the first row of beef
clst_cube_quants = np.percentile(clst_cube[1:,:,:],perc_list,0)
    # cubes now have 9 rows, one for each of the quantiles requested
    #   The cols and panes are the same format as the input cube


#-- Order results by peak deaths/day predicted AND extract clusters for plotting from the cube
# Get maximum deaths/day ever hit by each cluster
    # Use 4th row to use the 50th percentile (ie. the central prediction)
peak_daily_deaths = np.max(clst_cube_quants[4,:,:],0)
# Get indices of sorted (descending) vector 
    # NOTE: argsort only works in ascdending order so use [::-1] to reverse
peak_inds = np.argsort(peak_daily_deaths)[::-1]
# Take the largest plotN counties (since these are the only ones requested by the user)
peak_inds = peak_inds[:plotN]
# Extract the resulting counties
    # results will be implicitly sorted due to use of argsort
clst_cube_quants = clst_cube_quants[:,:,peak_inds]      # Get quantiles
clst_cube_fips   = clst_cube[0,0,peak_inds]        # Get fips ID's


#-- Read in cluster-to-fips translation
# Load cluster data
clst_to_fips = pd.read_csv(cluster_ref_fln)
# Extract useful columns
clst_to_fips = clst_to_fips[['fips', 'cluster']]
# Cast fips and cluster values to int
clst_to_fips['fips'] = clst_to_fips['fips'].astype('int')
clst_to_fips['cluster'] = clst_to_fips['cluster'].astype('int')


#-- PLOT only if user has requested plots on cluster-by-cluster basis
if isShowClusters:
    #-- Create directory for output .svg files if necessary
    if is_saveSVG:
        # Append sample filename just to get proper path
        tmp_flm = '%sstate_county_fips.svg'%svg_flm
        # Create directory if necessary
        Path(tmp_flm).parent.mkdir(parents=True, exist_ok=True)

    #-- Iterate cluster-by-cluster plotting
    for ind, cnty in enumerate(clst_cube_fips):
        #-- Extract and format pertinent data
        # Get fips of counties in this cluster
        fips_in_clst = clst_to_fips[clst_to_fips['cluster'] == cnty].fips.values
        # Pull just the relevant counties from the reference file
        cnty_true_df = true_df[true_df.fips.isin(fips_in_clst)]
        # Pull just relevant counties from input cube
        cnty_model   = clst_cube_quants[:,:,ind]
        # Calculate total deaths in the cluster on a given day
        clst_deaths = cnty_true_df.groupby('dateDT')['deaths'].sum()
        # Ensure clst_deaths is chronolically sorted
        clst_deaths.sort_index(inplace=True)

        #-- Create x-axes
        # Create time vector referenced to global_dayzero (to have same reference point for both datasets)
        t_true = (clst_deaths.index - global_dayzero).days.values
        # Create time vector for input cube
        t_model = np.arange(cnty_model.shape[1])

        #-- Format plot
        # Include county in title
        ptit = 'SEIIRD+Q Model: Cluster %d'%cnty

        # Format y-axis label for cumulative vs. daily plots
        if isCumul:
            # Set y-axis label to show cumulative counts
            ylab = '# deaths total'
        else:
            # Set y-axis label to show deaths/day
            ylab = '# deaths/day'

# %%

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
        p.circle(t_true, clst_deaths.values, color ='black')

        # Apply training boundary if desired
        if boundary is not None:
            bd_day = (pd.to_datetime(boundary)-global_dayzero)/np.timedelta64(1, 'D')
            vline = Span(location=bd_day, dimension='height', line_color='black', line_width=2)
            p.renderers.extend([vline])

        # Show plot
        bokeh.io.show(p)

        # Save output figures if desired
        if is_saveSVG:
            p.output_backend = "svg"
            # Format filename
            suffix = ('%s_%s_%d.svg'%(cnty_true_df['state'].iloc[0],cnty_true_df['county'].iloc[0],cnty)).replace(' ','')
            bokeh.io.export_svgs(p, filename= svg_flm + suffix)



###################################
#### Plot Allocations #############
###################################

#-- Only do allocation stuff if requested
if isShowAllocations:

    #-- format to county-based in same way as format_sub
    # cnty_cube is now the cluster-data re-allocated to counties
    cnty_cube = cf.alloc_fromCluster(clst_cube, cluster_ref_fln, alloc_day=alloc_day)


    #-- Drop counties that are not from the desired cluster
    # Get all fips in the original cube
    fips_in_cube = cnty_cube[0,0,:]
    # Get fips of counties in the cluster
    fips_in_clst = clst_to_fips[clst_to_fips['cluster'] == clst2Show].fips.values
    # Do Josh conversion
    fips_in_clst = fips_in_clst[np.newaxis].T
    # Get boolean of which panes are from the cluster
    fips_in_clst=(fips_in_clst==fips_in_cube).any(0)
    # Extract those panes
    cnty_cube = cnty_cube[:,:,fips_in_clst]


    #-- Calculate quantiles
    # Calculate along each column ignoring the first row of beef
    cnty_cube_quants = np.percentile(cnty_cube[1:,:,:],perc_list,0)
        # cubes now have 9 rows, one for each of the quantiles requested
        #   The cols and panes are the same format as the input cube


    #-- Extract fips in final cube 
        # (yes, I know this is redundant... too lazy to change)
    cnty_cube_fips   = cnty_cube[0,0,:]        # Get fips ID's


    #-- Create directory for output .svg files if necessary
    if is_saveSVG:
        # Append sample filename just to get proper path
        tmp_flm = '%sstate_county_fips.svg'%svg_flm
        # Create directory if necessary
        Path(tmp_flm).parent.mkdir(parents=True, exist_ok=True)

    #-- Iterate county-by-county plotting
    for ind, cnty in enumerate(cnty_cube_fips):
        # Pull just the relevant county
        cnty_true_df = true_df[true_df['fips'] == cnty]
        cnty_model   = cnty_cube_quants[:,:,ind]
        # Ensure true_df is chronolically sorted
        cnty_true_df.sort_values(by=['dateDT'],inplace=True)
            
        # Create column with days since global_dayzero (to have same reference point for both datasets)
        cnty_true_df['rel_date'] = (cnty_true_df['dateDT'] - global_dayzero)/np.timedelta64(1,'D')

        # Create time axes
        t_true = cnty_true_df['rel_date'].values
        t_model = np.arange(cnty_model.shape[1])

        # Include county in title
        ptit = 'SEIIRD+Q Model: %s, %s (%d)'%(cnty_true_df['county'].iloc[0],cnty_true_df['state'].iloc[0], cnty)

        # Format y-axis label for cumulative vs. daily plots
        if isCumul:
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

        # Save output figures if desired
        if is_saveSVG:
            p.output_backend = "svg"
            # Format filename
            suffix = ('%s_%s_%d.svg'%(cnty_true_df['state'].iloc[0],cnty_true_df['county'].iloc[0],cnty)).replace(' ','')
            bokeh.io.export_svgs(p, filename= svg_flm + suffix)
