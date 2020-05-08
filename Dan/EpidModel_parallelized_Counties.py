# %%
from IPython import get_ipython
import pandas as pd
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

import bokeh.io
import bokeh.application
import bokeh.application.handlers
import bokeh.models
import bokeh.plotting as bkp

from scipy.optimize import least_squares
from bokeh.models import Span

import itertools

import holoviews as hv

import time

from scipy.io import loadmat,savemat

import multiprocessing
import sys

from scipy.linalg import svd

from sklearn.metrics import mean_squared_error

import os

# %%
tic0 = time.time()

# %%
#  Great! Let's also make a helper function to select data from a fips, starting when the pandemic hit to be able to fit models. #  

# return data ever since first min_deaths death
def select_region(df, region, min_deaths=50,mobility = False):
    # Get subdf for relevant fips
    d = df.loc[df['fips'] == region]
    if not mobility:
        # Extract entries for days beyond the threshold deaths
        d = d[d['deaths'] > min_deaths]
    return d


# %%
#  Define a funciton for calculating the "errors" from the least_squares model.
#  
#  The "errors" here actually seem to be the std of the parameters from our model. 
#    This std is calculated from the jacobian returned by the least_squares function.
#    It is currently unclear how the TA's obtained this conversion (res.jac --> std(params))

# return params, 1 standard deviation errors
def get_errors(res, p0):
    p0 = np.array(p0)
    ysize = len(res.fun)
    cost = 2 * res.cost  # res.cost is half sum of squares!
    popt = res.x
    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)

    warn_cov = False
    absolute_sigma = False
    if pcov is None:
        # indeterminate covariance
        pcov = zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(inf)
        warn_cov = True
    elif not absolute_sigma:
        if ysize > p0.size:
            s_sq = cost / (ysize - p0.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(inf)
            warn_cov = True

    if warn_cov:
        print('cannot estimate variance')
        return None
    
    perr = np.sqrt(np.diag(pcov))
    return perr


# %% [markdown]
#  ## MODEL 3 of 3: $\mathbf{SEI_A I_S R}$ with empirical quarantine
# %% [markdown]
#  The motivation for this model is to add two crucial ingredients: quarantine data and asymptomatic cases. For quarantine analysis, we find an effective population size based on what fraction of the population is moving according to https://citymapper.com/cmi/milan. (Since Milan is the capital of Lombardy, we perform the analysis for that region.) To make the quarantine more realistic, we model a "leaky" quarantine, where the susceptible population is given by the mobility from above plus some offset. To treat asymptomatic cases, we introduce states $I_A$ (asymptomatic) and $I_S$ (symptomatic) according to the following sketch and differential equations:
# %% [markdown]
#  ![SEIIR + quarantine](images/overview.png)
# %% [markdown]
#  Since this is prototyping the model, we manually enter the chart above (raw data is at the link above) and implement it. We also have a testing function $T(t)$ (called `tau(t)` in the code) that allows us to try out different testing strategies for asymptomatic populations. Sorry about the confusing variable names below: parameters are renamed as $\sigma\to$ `alpha`, $s\to$ `sigma`, and $d\to$ `delta`. Not shown in the equations above but included in the diagram is a fixed offset (`offset`) for the leaky quarantine model.

# %%
# TODO fix data imputation
def q(t, N, shift,mobility_data,offset):
    #moving = np.array([57, 54, 52, 51, 49, 47, 46, 45, 44, 43, 39, 37, 34, 23, 19, 13, 10, 7, 6, 5, 5, 7, 6, 5, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 2])/100
    if not mobility_data.empty:
        #column_list = [col for col in list(mobility_data.columns) if is_number(col)]
        moving_list = [mobility_data[col].values for col in list(mobility_data.columns) if is_number(col) and col>=offset]
        moving = np.squeeze(np.array(moving_list))/100 
        #moving = np.squeeze(np.array(moving_list))*shift/100 # I CHANGED THE MEANING OF SHIFT BY DOING THIS!
        if len(moving_list)==0:
            moving = (100 - 5*(t-offset))/100
    else:
        moving = (100 - 5*(t-offset))/100
        #moving = (100 - 5*t)*shift/100 # wow. this is terrible and not data-driven at all.
        
    Q = N*(1-moving-shift)
    #Q = N*(1-moving)
    try:
        if np.round(t) >= len(Q):
            if len(Q>0):
                return Q[-1]
    except TypeError:
        # print('ERROR in q(): Q was a scalar you poop')
        return Q

    return Q[int(np.round(t))]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def tau(t):
    return 0

def seiirq(dat, t, params, N, max_t, offset,mobility_data):
    if t >= max_t:
        return [0]*8
    beta = params[0]
    alpha = params[1] # rate from e to ia
    sigma = params[2] # rate of asymptomatic people becoming symptotic
    ra = params[3] # rate of asymptomatic recovery
    rs = params[4] # rate of symptomatic recovery
    delta = params[5] # death rate
    shift = params[6] # shift quarantine rate vertically from CityMapper data 
    # I CHANGED THE MEANING OF SHIFT! See the q() function
    
    s = dat[0]
    e = dat[1]
    i_a = dat[2]
    i_s = dat[3]

    # TODO: What's the point of tau if it just returns 0? Is this something we should update?
    Qind = (q(t + offset, N, shift,mobility_data,offset) - tau(t + offset)*i_a)/(s + e + i_a - tau(t + offset)*i_a)
    Qia = Qind + (1-Qind)*tau(t + offset)
    
    dsdt = - beta * s * i_a * (1 - Qind) * (1 - Qia) / N
    dedt = beta * s * i_a* (1 - Qind) * (1 - Qia) / N  - alpha * e
    diadt = alpha * e - (sigma + ra) * i_a
    disdt = sigma * i_a - (delta + rs) * i_s
    dddt = delta * i_s
    drdt = ra * i_a + rs * i_s
    
    
    # susceptible, exposed, infected, quarantined, recovered, died, (unsusceptible?)
    out = [dsdt, dedt, diadt, disdt, drdt, dddt]
    return out


# %%
def mse(A, B):
    Ap = np.nan_to_num(A)
    Bp = np.nan_to_num(B)
    Ap[A == -np.inf] = 0
    Bp[B == -np.inf] = 0
    Ap[A == np.inf] = 0
    Bp[B == np.inf] = 0
    return mean_squared_error(Ap, Bp)

def model_z(params, data,mobility_data, tmax=-1):
    # initial conditions
    N = data['Population'].values[0] # total population
    initial_conditions = N * np.array(params[-4:]) # the parameters are a fraction of the population so multiply by the population
    
    e0 = initial_conditions[0]
    ia0 = initial_conditions[1]
    is0 = initial_conditions[2]
    r0 = initial_conditions[3]
    
    d0 = data['deaths'].values[0]
    s0 = N - np.sum(initial_conditions) - d0

    offset = data['date_processed'].min()
    yz_0 = np.array([s0, e0, ia0, is0, r0, d0])
    
    n = len(data)
    if tmax > 0:
        n = int(np.round(tmax))
    
    # Package parameters into a tuple
    args = (params, N, n, offset,mobility_data)
    
    # Integrate ODEs
    try:
        s = scipy.integrate.odeint(seiirq, yz_0, np.arange(0, n), args=args)
    except RuntimeError:
#         print('RuntimeError', params)
        return np.zeros((n, len(yz_0)))

    return s

def fit_leastsq_z(params, data, mobility_data):
    Ddata = (data['deaths'].values)
    Idata = (data['cases'].values)
    s = model_z(params, data, mobility_data)

    # S = s[:,0]
    # E = s[:,1]
    # I_A = s[:,2]
    I_S = s[:,3]
    # R = s[:,4]
    D = s[:,5]
    
    death_weight = 10
    weight_errors = np.linspace(0,1,len(Ddata))
    weight_errors = weight_errors + 1
    weight_errors = weight_errors/np.linalg.norm(weight_errors)
    death_errors = death_weight*np.multiply(weight_errors,(D-Ddata))
    # symptomatic_infected_errors = np.multiply(weight_errors,I_S - Idata)
    symptomatic_infected_errors = np.multiply(weight_errors,-LeakyReLU(Idata,I_S,alpha=.2))

    error = np.concatenate((death_errors, symptomatic_infected_errors))
    return error

def LeakyReLU(pred,true,alpha=0):
    result = []
    for i in range(len(pred)):
        if pred[i]>true[i]:
            result.append(pred[i]-true[i])
        else:
            result.append(alpha*(pred[i]-true[i]))
    return np.array(result)

# %% [markdown]
#  Again, we find some good initial parameters.
# 
#  *** We need to find some good initial parameters. ):


# %%
def plot_with_errors_sample_z(res, df, mobility_df, region, d_thres, const, extrapolate=1, boundary=None, plot_asymptomatic_infectious=False,plot_symptomatic_infectious=True,):
    data = select_region(df, region, min_deaths=d_thres)
    mobility_data = select_region(mobility_df,region, min_deaths=d_thres, mobility = True)
    errors = res.x*.05 # ALEX --> the parameter error is simply 5% of the parameter. i.e. we know the parameter to within +or- 5%.
    
    all_s = []
    samples = 100
    for i in range(samples):
        sample = np.random.normal(loc=res.x, scale=errors)
        s = model_z(sample, data, mobility_data, len(data)*extrapolate)
        all_s.append(s)
        
    all_s = np.array(all_s)
    
    s = model_z(res.x, data,mobility_data, len(data)*extrapolate)
    S = s[:,0]
    E = s[:,1]
    I_A = s[:,2]
    I_S = s[:,3]
    R = s[:,4]
    D = s[:,5]

    #-- Perform bokeh stuff when not multiprocessing (so that plots are actually shown)
    if (not const['isMultiProc']) and const['isPlotBokeh']:
        t = np.arange(0, len(data))
        tp = np.arange(0, int(np.round(len(data)*extrapolate)))

        ptit = '%s, %s - (%d) - SEIIRD+Q Model'%(const['fips_to_county'][region], const['fips_to_state'][region], region)
        p = bkp.figure(plot_width=600,
                                plot_height=400,
                                title = ptit,
                                x_axis_label = 't (days)',
                                y_axis_label = '# people')

    quantiles = [10, 20, 30, 40]
    for quantile in quantiles:
        s1 = np.percentile(all_s, quantile, axis=0)
        s2 = np.percentile(all_s, 100-quantile, axis=0)
        if (not const['isMultiProc']) and const['isPlotBokeh']:
            if plot_asymptomatic_infectious:
                p.varea(x=tp, y1=s1[:, 2], y2=s2[:, 2], color='red', fill_alpha=quantile/100) # Asymptomatic infected
            if plot_symptomatic_infectious:
                p.varea(x=tp, y1=s1[:, 3], y2=s2[:, 3], color='purple', fill_alpha=quantile/100) # Symptomatic infected
            p.varea(x=tp, y1=s1[:, 5], y2=s2[:, 5], color='black', fill_alpha=quantile/100) # deaths
    
    if (not const['isMultiProc']) and const['isPlotBokeh']:
        if plot_asymptomatic_infectious:
            p.line(tp, I_A, color = 'red', line_width = 1, legend_label = 'Asymptomatic infected')
        if plot_symptomatic_infectious:
            p.line(tp, I_S , color = 'purple', line_width = 1, legend_label = 'Symptomatic infected')
        p.line(tp, D, color = 'black', line_width = 1, legend_label = 'Deceased')
    
    if (not const['isMultiProc']) and const['isPlotBokeh']:
        # death
        p.circle(t, data['deaths'], color ='black')
        # quarantined
        if plot_symptomatic_infectious:
            p.circle(t, data['cases'], color ='purple')
        if boundary is not None:
            vline = Span(location=boundary, dimension='height', line_color='black', line_width=3)
            p.renderers.extend([vline])
        p.legend.location = 'top_left'
        bokeh.io.show(p)
    D_quantiles = np.array([s1[:,5],s2[:,5]])
    D

    return all_s,D_quantiles,D,errors


# %%
def determine_fips_start_day(fips,global_dayzero,fips_to_dayzero):
    fips_dayzero = fips_to_dayzero[fips]
    fips_start_day = int((fips_dayzero - global_dayzero)/np.timedelta64(1,'D'))
    return fips_start_day

def determine_extrapolate(daysofdata,fips_start_day,num_days):
    extrapolate = (num_days - fips_start_day)/daysofdata
    return extrapolate

# %%
#-- Create function to be parallelized (to be performed on each core)
def par_fun(fips_in_core, main_df, mobility_df, coreInd, const, ErrFlag):
    cube = np.zeros((100+1,const['num_days'],len(fips_in_core)))
    for ind, fips in enumerate(fips_in_core):
        try:
            # Check if ErrFlag is valuable (ie if multiprocessing) THEN check if it's set
                # "and" is short circuited so ErrFlag.is_set() only gets called when needed
            if (ErrFlag is not None) and ErrFlag.is_set():
                print('Detected ErrFlag; exiting loop')
                break
            
            data = select_region(main_df, fips, min_deaths=const['train_Dfrom'])#[:boundary]
            # if const['train_til'] is not None:
            #     # crop data 
            #     data = data[data['date_processed'] <= const['train_til']]
            boundary = len(data)

            if boundary <= const['min_train_days']:
                # Skip the fips when train_til causes it to not have enough data
                    # This occurs because the fips had more than D_THRES deaths in the most recent dataset 
                    # BUT train_til occurs before the train_Dfrom threshold is met
                print('    ----       v v v v       ----\n' + \
                      '    core index: %d \n'%(coreInd) + \
                      '       Skipping: (%d) - %s - %s\n'%(fips, const['fips_to_county'][fips],const['fips_to_state'][fips]) + \
                      '       train_til leaves too few trainable days for this county\n' + \
                      '    ----       ^ ^ ^ ^       ----\n')
                continue

            daysofdata = const['fips_to_daysofdata'][fips]
            fips_start_day = determine_fips_start_day(fips,const['global_day0'], const['fips_to_day0'])
            extrap = determine_extrapolate(daysofdata,fips_start_day,const['num_days'])
            
            mobility_data = select_region(mobility_df, fips, min_deaths=const['train_Dfrom'], mobility = True)
            tic = time.time()
            res = least_squares(fit_leastsq_z, const['guesses'], args=(data,mobility_data), bounds=np.transpose(np.array(const['ranges'])),jac = '2-point')
            #plot_with_errors_sample_z(res, const['params'], const['initial_conditions'], main_df, mobility_df, state, extrapolate=extrap, boundary=boundary, plot_asymptomatic_infectious=False,plot_symptomatic_infectious=True);
            all_s, _, _, _ = plot_with_errors_sample_z(res, main_df, mobility_df, fips, const['train_Dfrom'], const, extrapolate=extrap, boundary=boundary, plot_asymptomatic_infectious=False,plot_symptomatic_infectious=False)
            toc = time.time()
            cube[0,:,ind] = fips
            # CONSIDER changing this to the first day when train_Dfrom was crossed
                # would need to change the d_thres provided to plot_with_errors_sample_z above
            cube[1:,fips_start_day:,ind] = all_s[:,:,5]
            print('____%d (county %d of %d)____\n'%(fips, ind+1, len(fips_in_core)) + \
                  '    core: %d\n'%coreInd + \
                  '    time: %f \n'%(toc-tic))
            sys.stdout.flush()
        # except TypeError as TE:
        #     print('############################')
        #     print('Handled Exception Occurred:')
        #     print('core index: ', coreInd)
        #     print('   issue on: (%d) - %s - %s'%(fips, const['fips_to_county'][fips],const['fips_to_state'][fips]))
        #     print(TE)
        #     print('############################')
        except:
            print('############################\n' + \
                  'UNHANDLED Exception Occurred:\n' + \
                  'core index: %d \n'%(coreInd) + \
                  '   issue on: (%d) - %s - %s\n'%(fips, const['fips_to_county'][fips],const['fips_to_state'][fips]) + \
                  'Setting ErrFlag for other workers\n' + \
                  '############################\n')
            if ErrFlag is not None:
                # Only set ErrFlag when there are other workers listening
                ErrFlag.set()
            raise
    # do non-risky stuff outside of try context
    print('############################\n' + \
          '### Core %2d has finished ###\n'%coreInd + \
          '############################\n')
    return cube

def apply_by_mp(func, workers, args):
    # pool = multiprocessing.Pool(processes=workers)
    # result = pool.map(func, args)
    # pool.close()
    with multiprocessing.Pool(processes=workers) as pool:
        result = pool.starmap(func, args)
    # Recombine result matrices
    result = np.dstack(result)
    return result
    
    
# %%
if __name__ == '__main__':
    #-- Define control parameters
    # Flag to choose whether to save the results or not
    isSaveRes = False
    # Filename for saved .npy and .mat files (can include path)
        # Make sure the directory structure is present before calling
    sv_flnm_np  = 'Dan\\PracticeOutputs\\NYC_Jeff_Only.npy'
    sv_flnm_mat = 'Dan\\PracticeOutputs\\NYC_Jeff_Only.mat'


    # Flag to choose whether multiprocessing should be used
    isMultiProc = False
    # Number of cores to use (logical cores, not physical cores)
    workers = 20
    # Threshold of deaths at and below which a COUNTY will not be trained on
        # Filters which COUNTIES are looped over in optimization/minimization loop
    D_THRES = 50
    # Last day used for training (good for testing)
        # must be a valid pandas.to_datetime() string
        # OR: leave as None to train until the latest data for which there is data
    train_til = '2020 04 24'
    # Minimum deaths considered in training
        # Sets the first DAY which will be calculated as part of the optimization
        # by only including days with more than this many deaths. THIS IS DIFFERENT than 
        # D_THRES. D_THRES selects which counties are trained on and train_Dfrom selects 
        # which DAYS are used for the optimization
    
    # NOTE: Since we are training by county, some counties only just passed D_THRES
        # As such, we almost certainly want to set train_Dfrom < DTHRES so that we have enough days of data for which to train
    train_Dfrom = 7
    # Minimum number of days required for a county to be trained on
        # After filtering using train_Dfrom and D_THRES, this makes sure that there
        # are at least min_train_days worth of days to train the model on (for fit_leastsqz)
    min_train_days = 5

    #- Sub-select counties to train on
    # Flag to choose whether to sub-select
    isSubSelect = True
    # List of counties which should be considered
        # NOTE: This just removes ALL other counties from the df as soon as it can
    just_train_these_fips = [36061, 1073] #


    #-- When not multiprocessing, enable bokeh plotting (since won't cause issue)
    # Flag to stating whether to plot. This only matters when not multiprocessing (isMultiProc=False)
        # When isMultiProc=True, bokeh will cause errors so we ignore this flag
    isPlotBokeh     = True      
    # Turn on plotting if available
    if (not isMultiProc) and isPlotBokeh:
        bokeh.io.output_notebook()
        hv.extension('bokeh')


    # %%
    #  Let's load the data from the relevant folder. If this data doesn't exist for you, you'll need to run the `processing/raw_data_processing/daily_refresh.sh` script (which may require `pip install us`).
    # import git

    # repo = git.Repo("./", search_parent_directories=True)
    # homedir = repo.working_dir
    # # homedir = "C:/Users/alex/OneDrive - California Institute of Technology/Documents/GitHub/Tentin-Quarantino"
    # datadir = f"{homedir}/data/us/"

    # os.chdir(homedir)

    HomeDIR='Tentin-Quarantino'
    wd=os.getcwd()
    DIR=wd[:wd.find(HomeDIR)+len(HomeDIR)]
    os.chdir(DIR)

    homedir = DIR
    datadir = f"{homedir}/data/us/"

    # %%
    #  Load the US data by county 
    df = pd.read_csv(datadir + 'covid/nyt_us_counties.csv')

    # %%
    #-- Do some reformatting of existing entries
    # # (OPTIONAL) show the NaN entries
    # df[~df['fips'].notna()]

    # Apply 36061 fips to all "New York City" entries
    df.loc[df['county']=='New York City','fips'] = 36061
    # Remove remaining NaN elements in fips column (This is mostly state entries as well as territories with no FIPS I guess)
    df = df[df['fips'].notna()]

    # Cast fips to integer
    df['fips'] = df['fips'].astype(int)


    #-- When desired, extract only the counties which the user has provided for training
    if isSubSelect:
        df = df[df.fips.isin(just_train_these_fips)]

    # %%
    #-- Format date stuff
    # We want to count days as integers from some starting point.
    df['date_processed'] = pd.to_datetime(df['date'].values)
    df['date_timestamp'] = pd.to_datetime(df['date'].values)

    #NOTE/TODO: I'M PRETTY SURE WE NEED TO SET THIS TO GLOBAL_DAY_ZERO, NOT JUST THE MIN OF COUNTY VALUES
        # This has happened to work before because we usually have the one county in washington
        # that has data on 1/21 but once we remove counties, it causes issue
    day_zero = df['date_processed'].min()
    print('---- Day zero is ',day_zero)
    df['date_processed'] = (df['date_processed'] - day_zero) / np.timedelta64(1, 'D')

    # %%
    #-- Define key dates
    global_dayzero = pd.to_datetime('2020 Jan 21')
    global_end_day = pd.to_datetime('2020 June 30')
    num_days = int((global_end_day-global_dayzero)/np.timedelta64(1, 'D'))


    #-- Set day until which to train
    if train_til is not None:
        # User provided a boundary date for training; translate to absolute time w.r.t global_dayzero
        train_til = pd.to_datetime(train_til)
        print('---- Only training until: ', train_til)
        train_til = int((train_til-global_dayzero)/np.timedelta64(1,'D'))
    # NOTE: commented out else to see if working with None directly in par_func works
    # else:
    #     # User did not provide a boundary; set boundary as last day in dataset
    #     train_til = df.date_processed.max()

    #-- Remove days beyond our training limit day
    df = df[df['date_processed'] < train_til]


    # %%
    # Load county-based population data
    populations = pd.read_csv(datadir + 'demographics/county_populations.csv')

    # change some column names:
        # 1) FIPS to lowercase to match df (simplifies the merge done in a bit)
        # 2) total_pop to population (for clarity)
    populations.rename(columns={'FIPS':'fips', 'total_pop':'Population'}, inplace=True)

    # Merge to insert population data into main df
    df = df.merge(populations[['fips','Population']],how='left',on='fips')

    # %%
    #  Just checking to make sure the data is here...
    # print(df)
    # print(' ')

    # %% 
    # Our main control list (equivalent to list_of_states) is based off the fips codes
    #   since many counties have the same name in multiple states meanwhile FIPS is a unique identifier
    list_of_fips = list(np.unique(df['fips']))

    # %%
    # load mobility data
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
    mobility_df = mobility_df[mobility_df.fips.isin(list_of_fips)]

    # %%
    # Convert mobility data column headers to date_processed format
    date_dict = dict()
    for col in list(mobility_df.columns):
        col_dt = pd.to_datetime(col,errors = 'coerce')
        if not (isinstance(col_dt,pd._libs.tslibs.nattype.NaTType)): # check if col_dt could be converted. NaT means "Not a Timestamp"    
            date_dict[col] = (col_dt - day_zero) / np.timedelta64(1, 'D')

    temp_dict = dict()
    temp_dict['admin1'] = 'state'

    mobility_df = mobility_df.rename(columns = temp_dict)
    mobility_df = mobility_df.rename(columns = date_dict)

    # %%
    # beta, alpha, sigma, ra, rs, delta, shift
    # param_ranges = [(1.0, 2.0), (0.1, 0.5), (0.1, 0.5), (0.05, 0.5), (0.32, 0.36), (0.005, 0.05), (0.1, .6)]
    # susceptible, exposed, infected_asymptomatic, infected_symptomatic
    # OR? conditions: E, IA, IS, R
    # initial_ranges = [(1.0e-7, 0.001), (1.0e-7, 0.001), (1.0e-7, 0.001), (1.0e-7, 0.001)]

    # %%
    # Get maximum death count in each county 
        # (as a pandas series)
    fips_to_maxdeaths = df.groupby('fips')['deaths'].max()


    # %%
    # Find counties with enough deaths to be trained on ( > D_THRES)
    list_of_fips_to_train = list(fips_to_maxdeaths[fips_to_maxdeaths > D_THRES].index)

    # Find counties with too few deaths to be trained on ( <= D_THRES)
    list_of_low_death_fips = list(fips_to_maxdeaths[fips_to_maxdeaths <= D_THRES].index)


    #-- Print name of counties to be trained on
    # Get county names (as list in same order as list_of_fips_to_train)
        # 1) fips.isin to get just row entries of the relevant fips
        # 2) .drop_duplicates('fips') to get the unique row elements (since nyt data has multiple rows for each county)
        # 3) .sort_values(by='fips') to get in same order as list_of_fips_to_train
        # 4) ['county'].values to extract county name
    list_of_county_names = list(df[df.fips.isin(list_of_fips_to_train)].drop_duplicates('fips').sort_values(by='fips')['county'].values)

    # Get series with fips and county associated to each other
        # (as a pandas series)
    fips_to_county = pd.Series(df[df.fips.isin(list_of_fips_to_train)].drop_duplicates('fips')[['fips','county']].set_index('fips').to_dict()['county'])
    
    # Get series with fips and state associated to each other
        # (as a pandas series)
    fips_to_state = pd.Series(df[df.fips.isin(list_of_fips_to_train)].drop_duplicates('fips')[['fips','state']].set_index('fips').to_dict()['state'])

    # Sort the fips to train by number of deaths
    print_order = fips_to_maxdeaths[list_of_fips_to_train].sort_values(ascending=False)

    # Print
    print('deaths |  fips   |        county        |     state     ')
    print('========================================================')
    for fips in print_order.index:
        print('%5d  |  %5d  |  %18s  |  %s'%(fips_to_maxdeaths[fips], fips, fips_to_county[fips], fips_to_state[fips]))
    print('========================================================')
    print('deaths |  fips   |        county        |     state     ')
    print('-- Total Counties: %d --'%len(print_order))
    

    # %% 
    # Get first day that each county passed D_THRES
        # (as a pandas series)
    fips_to_dayzero = df[df['deaths'] > train_Dfrom].groupby('fips')['date_timestamp'].min()


    # %%
    # Get number of days with > D_THRES deaths 
        # (as pandas series)
    fips_to_daysofdata = df[df['deaths'] > train_Dfrom].groupby('fips').size()


    # %% 
    # Prepare for parallelization
    
    

    if isMultiProc:
        # Check that user hasn't requested more cores than are available
        if workers > multiprocessing.cpu_count():
            raise ValueError('More workers requested than cores: workers=%d, cores=%d'%(workers, multiprocessing.cpu_count()))
    else:
        # Set number of workers to 1 since not multiprocessing
        workers = 1

    print('Using %d cores'%workers)

    #-- Split into parallelizable chunks
        # Here I am splitting arbitrarily. Could consider ordering by number of days of data and then splitting s.t. we evenly 
        # distribute the work among the cores. This'll reduce net runtime since no core will be running longer than the others
    fips_for_cores = np.array_split(list_of_fips_to_train,workers)
    # Split the dataframes by core allocations
    main_dfs = [df[df.fips.isin(fips_in_core)] for fips_in_core in fips_for_cores]
    mobility_dfs = [mobility_df[mobility_df.fips.isin(fips_in_core)] for fips_in_core in fips_for_cores]

    #-- Simplify arguments for function by placing them in a dict
    param_ranges = [(1.0, 3.0), (0.1, 0.5), (0.01, .9), (0.0001, 0.9), (0.0001, 0.9), (0.001, 0.1), (0.05, .7)] # shift can go 0 to 100 for alex method
    initial_ranges = [(1.0e-7, 0.05), (1.0e-7, 0.05), (1.0e-7, 0.05), (1.0e-7, 0.01)]
    # beta, alpha, sigma, ra, rs, delta, shift
    params = [1.8, 0.35, 0.1, 0.15, 0.34, 0.015, .5]
    # OR? conditions: E, IA, IS, R
    initial_conditions = [4e-6, 0.005, 0.005, 0.005]

    guesses = params + initial_conditions
    ranges = param_ranges + initial_ranges

    const = dict()
    const['fips_to_daysofdata'] = fips_to_daysofdata
    const['fips_to_county'] = fips_to_county
    const['fips_to_state'] = fips_to_state
    const['global_day0'] = global_dayzero
    const['fips_to_day0'] = fips_to_dayzero
    const['num_days'] = num_days
    const['guesses'] = guesses
    const['ranges'] = ranges
    const['params'] = params
    const['initial_conditions'] = initial_conditions
    const['D_THRES'] = D_THRES
    const['train_til'] = train_til
    const['train_Dfrom'] = train_Dfrom
    const['min_train_days'] = min_train_days
    const['isMultiProc'] = isMultiProc
    const['isPlotBokeh'] = isPlotBokeh


    # %% 
    #-- Call to run in parallel
    if isMultiProc:
        # Flag to manage exceptions from other cores
        ErrFlag = multiprocessing.Manager().Event()
    else:
        # Set ErrFlag to None since don't want any multiprocessing stuff
        ErrFlag = None
    # Format arguments
    args = [(fips_for_cores[i], main_dfs[i], mobility_dfs[i], i, const, ErrFlag) for i in range(workers)]
    if isMultiProc:
        # Call parallelizer function
        res = apply_by_mp(par_fun,workers,args)
    else:
        # Call parfun directly
        res = par_fun(*(args[0]))
    
    print('--------Total Time: %f----------'%(time.time()-tic0))
    print(res.shape)
    if isSaveRes:
        # Save results only when requested to do so
        np.save(sv_flnm_np, res)
        savedict = {'cube': res}
        savemat(sv_flnm_mat, savedict)
    print(res[:2,100,:10])


# %%
