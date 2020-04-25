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

# %%
# Helper functions to get populations and state identifiers
def get_population(region):
    return populations[populations['state'] == region]['population'].values[0]

def get_ticker(state,ticker_to_state):
    # state_to_ticker = dict([('CA','California'),('TX','Texas'),('FL','Florida'),('NY','New York'),('PA','Pennsylvania'),('IL','Illinois'),('OH','Ohio'),('GA','Georgia'),('NC','North Carolina'),
    #                        ('MI','Michigan'),('NJ','New Jersey'),('VA','Virginia'),('WA','Washington'),('AZ','Arizona'),('MA','Massachusetts'),('TN','Tennessee'),('IN','Indiana'),('MO','Missouri'),
    #                        ('MD','Maryland'),('WI','Wisconsin'),('CO','Colorado'),('MN','Minnesota'),('SC','South Carolina'),('AL','Alabama'),('LA','Louisiana'),('KY','Kentucky'),('OR','Oregon'),
    #                        ('OK','Oklahoma'),('CT','Connecticut'),('UT','Utah'),('IA','Iowa'),('NV','Nevada'),('AR','Arkansas'),('MS','Mississippi'),('KS','Kansas'),('NM','New Mexico'),('NE','Nebraska'),
    #                        ('WV','West Virginia'),('ID','Idaho'),('HI','Hawaii'),('NH','New Hampshire'),('ME','Maine'),('MT','Montana'),('RI','Rhode Island'),('DE','Delaware'),('SD','South Dakota'),
    #                        ('ND','North Dakota'),('AK','Alaska'),('DC','District of Columbia'),('VT','Vermont'),('WY','Wyoming'),('PR','Puerto Rico'),('VI','Virgin Islands'),('GU','Guam'),
    #                        ('NMI','Northern Mariana Islands')])
    state_to_ticker = {v: k for k, v in ticker_to_state.items()} # I accidentally typed the dictionary backwards (:
    return state_to_ticker[state]


# %%
#  Great! Let's also make a helper function to select data from a state, starting when the pandemic hit to be able to fit models. 
#  
#  We could change the parameter "min_deaths" here if we want... it seems like an arbitrary choice.

# return data ever since first min_cases cases
def select_region(df, region, min_deaths=50,mobility = False):
    d = df.loc[df['state'] == region]
    if not mobility:
        start = np.where(d['deaths'].values > min_deaths)[0][0]
        d = d[start:]
    return d

def select_region2(df, region, min_deaths=50):
    d = df.loc[df['state'] == region]
    start = np.where(d['deaths'].values > min_deaths)[0][0]
    d = d[start:]
    return d

def select_min_deaths(df,min_deaths=50):
    d = df.loc[df['deaths'].values > min_deaths]
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
# 
#  ***Regarding "TODO fix data imputation", we need to somehow get this data based on US states. Currently, the mobility data (and therefore quarantine data) is hard coded into the vector 'moving'. I'm not sure what shift is. Some arbitrary parameter that probably also needs to be fit...


# %%
# mobility_data = select_region(mobility_df,'New York',mobility = True)
# t = 5
# N = 100
# shift = .1
# offset = 50
# q(t,N,shift,mobility_data,offset)
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
    if np.round(t) >= len(Q):
        if len(Q>0):
            return Q[-1]

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

    S = s[:,0]
    E = s[:,1]
    I_A = s[:,2]
    I_S = s[:,3]
    R = s[:,4]
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
def plot_with_errors_sample_z(res, p0_params, p0_initial_conditions, df, mobility_df, region, extrapolate=1, boundary=None, plot_asymptomatic_infectious=False,plot_symptomatic_infectious=True):
    data = select_region(df, region)
    mobility_data = select_region(mobility_df,region,mobility = True)
    # errors = get_errors(res, list(p0_params) + list(p0_initial_conditions))
    errors = res.x*.05 # ALEX --> the parameter error is simply 5% of the parameter. i.e. we know the parameter to within +or- 5%.
    #errors[len(p0_params):] = 0
    
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

    t = np.arange(0, len(data))
    tp = np.arange(0, int(np.round(len(data)*extrapolate)))

    p = bkp.figure(plot_width=600,
                              plot_height=400,
                             title = region + ' SEIIRD+Q Model',
                             x_axis_label = 't (days)',
                             y_axis_label = '# people')

    quantiles = [10, 20, 30, 40]
    for quantile in quantiles:
        s1 = np.percentile(all_s, quantile, axis=0)
        s2 = np.percentile(all_s, 100-quantile, axis=0)
        if plot_asymptomatic_infectious:
            p.varea(x=tp, y1=s1[:, 2], y2=s2[:, 2], color='red', fill_alpha=quantile/100) # Asymptomatic infected
        if plot_symptomatic_infectious:
            p.varea(x=tp, y1=s1[:, 3], y2=s2[:, 3], color='purple', fill_alpha=quantile/100) # Symptomatic infected
        p.varea(x=tp, y1=s1[:, 5], y2=s2[:, 5], color='black', fill_alpha=quantile/100) # deaths
    
    if plot_asymptomatic_infectious:
        p.line(tp, I_A, color = 'red', line_width = 1, legend_label = 'Asymptomatic infected')
    if plot_symptomatic_infectious:
        p.line(tp, I_S , color = 'purple', line_width = 1, legend_label = 'Symptomatic infected')
    p.line(tp, D, color = 'black', line_width = 1, legend_label = 'Deceased')
    
    # death
    p.circle(t, data['deaths'], color ='black')
    # quarantined
    if plot_symptomatic_infectious:
        p.circle(t, data['cases'], color ='purple')
    if boundary is not None:
        vline = Span(location=boundary, dimension='height', line_color='black', line_width=3)
        p.renderers.extend([vline])
    p.legend.location = 'top_left'
    #bokeh.io.show(p)
    D_quantiles = np.array([s1[:,5],s2[:,5]])
    D

    return all_s,D_quantiles,D,errors


# %%
def determine_state_start_day(state,global_dayzero,state_to_dayzero):
    state_dayzero = state_to_dayzero[state]
    state_start_day = int((state_dayzero - global_dayzero)/np.timedelta64(1,'D'))
    return state_start_day

def determine_extrapolate(daysofdata,state_start_day,num_days):
    extrapolate = (num_days - state_start_day)/daysofdata
    return extrapolate

# %%
#-- Create function to be parallelized (to be performed on each core)
def par_fun(args):
    states_in_core, main_df, mobility_df, coreInd, const = args
    cube = np.zeros((100+1,const['num_days'],len(states_in_core)))
    for ind, state in enumerate(states_in_core):
        #try:
        daysofdata = const['state_to_daysofdata'][state]
        state_start_day = determine_state_start_day(state,const['global_day0'], const['state_to_day0'])
        extrap = determine_extrapolate(daysofdata,state_start_day,const['num_days'])
        
        # if counter < 3: # NEED TO REMOVE THIS EVENTUALLY
        data = select_region(main_df, state)#[:boundary]
        boundary = len(data)
        mobility_data = select_region(mobility_df, state,mobility = True)
        tic = time.time()
        res = least_squares(fit_leastsq_z, const['guesses'], args=(data,mobility_data), bounds=np.transpose(np.array(const['ranges'])),jac = '2-point')
        #plot_with_errors_sample_z(res, const['params'], const['initial_conditions'], main_df, mobility_df, state, extrapolate=extrap, boundary=boundary, plot_asymptomatic_infectious=False,plot_symptomatic_infectious=True);
        all_s, _, _, _ = plot_with_errors_sample_z(res, const['params'], const['initial_conditions'], main_df, mobility_df, state, extrapolate=extrap, boundary=boundary, plot_asymptomatic_infectious=False,plot_symptomatic_infectious=False)
        toc = time.time()
        #except:
        #    print('core index: ', coreInd)
        #    print(state)
        #    print('Exception went off')
        
        cube[0,:,ind] = const['state_to_fips'][state]
        cube[1:,state_start_day:,ind] = all_s[:,:,5]
        print('____%s (%d of %d)____'%(state, ind, len(states_in_core)))
        print('    core index: %d'%coreInd)
        print('    sec elapsed: '%(toc-tic))
        #sys.stdout.flush()

    return cube

def apply_by_mp(func, workers, args):
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(func, args)
    pool.close()
    return result
    
    
# %%
if __name__ == '__main__':
    # pool = multiprocessing.Pool(processes=workers)
    # result = pool.map(funtest, [(states_for_cores[i], main_dfs[i], mobility_dfs[i], i, const) for i in range(workers)])
    # pool.close()
    # bokeh.io.output_notebook()
    hv.extension('bokeh')


    # %%
    #  Let's load the data from the relevant folder. If this data doesn't exist for you, you'll need to run the `processing/raw_data_processing/daily_refresh.sh` script (which may require `pip install us`).
    import git

    repo = git.Repo("./", search_parent_directories=True)
    homedir = repo.working_dir
    # homedir = "C:/Users/alex/OneDrive - California Institute of Technology/Documents/GitHub/Tentin-Quarantino"
    datadir = f"{homedir}/data/us/"

    # %% [markdown]
    #  Load the US data by state (50 states, but we also make a bunch of other countries pay taxes without getting the right to vote (Puerto Rico, Guam, etc.), so we include their data too).
    #  This results in more than 50 "regions".

    # %%
    df = pd.read_csv(datadir + 'covid/nyt_us_states.csv')

    # %% [markdown]
    # We want to count days as integers from some starting point.

    # %%
    df['date_processed'] = pd.to_datetime(df['date'].values)
    df['date_timestamp'] = pd.to_datetime(df['date'].values)
    # %%
    day_zero = df['date_processed'].min()
    print('Day zero is ',day_zero)
    df['date_processed'] = (df['date_processed'] - day_zero) / np.timedelta64(1, 'D')

    # %% [markdown]
    # Now we import the population for each state, and add that information correctly to the Pandas "dataframe".

    # %%
    populations = pd.read_csv(datadir + 'demographics/state_populations_augmented.csv')

    ticker_to_state = dict([('CA','California'),('TX','Texas'),('FL','Florida'),('NY','New York'),('PA','Pennsylvania'),('IL','Illinois'),('OH','Ohio'),('GA','Georgia'),('NC','North Carolina'),
                            ('MI','Michigan'),('NJ','New Jersey'),('VA','Virginia'),('WA','Washington'),('AZ','Arizona'),('MA','Massachusetts'),('TN','Tennessee'),('IN','Indiana'),('MO','Missouri'),
                            ('MD','Maryland'),('WI','Wisconsin'),('CO','Colorado'),('MN','Minnesota'),('SC','South Carolina'),('AL','Alabama'),('LA','Louisiana'),('KY','Kentucky'),('OR','Oregon'),
                            ('OK','Oklahoma'),('CT','Connecticut'),('UT','Utah'),('IA','Iowa'),('NV','Nevada'),('AR','Arkansas'),('MS','Mississippi'),('KS','Kansas'),('NM','New Mexico'),('NE','Nebraska'),
                            ('WV','West Virginia'),('ID','Idaho'),('HI','Hawaii'),('NH','New Hampshire'),('ME','Maine'),('MT','Montana'),('RI','Rhode Island'),('DE','Delaware'),('SD','South Dakota'),
                            ('ND','North Dakota'),('AK','Alaska'),('DC','District of Columbia'),('VT','Vermont'),('WY','Wyoming'),('PR','Puerto Rico'),('VI','Virgin Islands'),('GU','Guam'),
                            ('NMI','Northern Mariana Islands'),('AMSA','American Samoa')])

    state_to_ticker = {v: k for k, v in ticker_to_state.items()} # I accidentally typed the dictionary backwards (:

    list_of_states = list(state_to_ticker.keys())
    # %% [markdown]
    # Let's also include the state ticker (i.e. Minnesota --> MN) as a column in the Pandas dataframe. (Because the US population info is given by state tickers).

    # %%
    df['state_ticker'] = df.apply(lambda row: get_ticker(row.state,ticker_to_state),axis = 1)
    df['Population'] = df.apply(lambda row: get_population(row.state_ticker), axis=1)

    # %% [markdown]
    #  Just checking to make sure the data is here...

    # %%
    df.head()

    # %%
    mobility_df = pd.read_csv(datadir + 'mobility/DL-us-m50_index.csv')
    # %%
    # Only use statewise mobility data
    mobility_df = mobility_df.loc[mobility_df['admin_level'] == 1]
    # %%
    # Convert mobility data column headers to date_processed format
    date_dict = dict()
    for col in list(mobility_df.columns):
        col_dt = pd.to_datetime(col,errors = 'coerce')
        if not (isinstance(col_dt,pd._libs.tslibs.nattype.NaTType)): # check if col_dt could be converted. NaT means "Not a Timestamp"    
            days_since_day_zero = (col_dt - day_zero) / np.timedelta64(1, 'D')
            date_dict[col] = days_since_day_zero

    temp_dict = dict()
    temp_dict['admin1'] = 'state'

    mobility_df = mobility_df.rename(columns = temp_dict)
    mobility_df = mobility_df.rename(columns = date_dict)

    # %%
    if False:
        get_ipython().run_line_magic('matplotlib', 'notebook')
        get_ipython().run_line_magic('matplotlib', 'inline')

        plt.figure()
        data = select_region(df, 'Washington')
        mobility_data = select_region(mobility_df,'Washington',mobility=True)
        # parameters: beta, alpha, sigma, ra, rs, delta, shift
        params = [1.8, 0.35, 0.1, 0.15, 0.34, 0.015, 0.5]
        # conditions: E, IA, IS, R
        initial_conditions = [4e-6, 0.0009, 0.0005, 0.0002]
        s = model_z(params + initial_conditions, data, mobility_data)
        plt.scatter(data['date_processed'], data['deaths'])
        plt.plot(data['date_processed'], s[:, 5])
        plt.show()

    # %%
    # beta, alpha, sigma, ra, rs, delta, shift
    # param_ranges = [(1.0, 2.0), (0.1, 0.5), (0.1, 0.5), (0.05, 0.5), (0.32, 0.36), (0.005, 0.05), (0.1, .6)]
    # susceptible, exposed, infected_asymptomatic, infected_symptomatic
    # OR? conditions: E, IA, IS, R
    # initial_ranges = [(1.0e-7, 0.001), (1.0e-7, 0.001), (1.0e-7, 0.001), (1.0e-7, 0.001)]
    if False:
        param_ranges = [(1.0, 3.0), (0.1, 0.5), (0.01, .9), (0.0001, 0.9), (0.0001, 0.9), (0.001, 0.1), (0.05, .7)] # shift can go 0 to 100 for alex method
        initial_ranges = [(1.0e-7, 0.05), (1.0e-7, 0.05), (1.0e-7, 0.05), (1.0e-7, 0.01)]

        # parameters: beta, alpha, sigma, ra, rs, delta, shift
        # params = [1.8, 0.35, 0.1, 0.15, 0.34, 0.015, 0.5]
        # conditions: E, IA, IS, R
        # initial_conditions = [4e-6, 0.0009, 0.0005, 0.0002]

        params = [1.8, 0.35, 0.1, 0.15, 0.34, 0.015, .5]
        initial_conditions = [4e-6, 0.005, 0.005, 0.005]

        guesses = params + initial_conditions
        ranges = param_ranges + initial_ranges

        boundary = 15
        state = 'New York'
        data = select_region(df, state)[:boundary]
        mobility_data = select_region(mobility_df, state,mobility = True)
        res = least_squares(fit_leastsq_z, guesses, args=(data,mobility_data), bounds=np.transpose(np.array(ranges)),jac = '3-point')
        all_s,D_quantiles,D,errors = plot_with_errors_sample_z(res, params, initial_conditions, df, mobility_df, state, extrapolate=1, boundary=boundary, plot_asymptomatic_infectious=False,plot_symptomatic_infectious=True)
        all_s,D_quantiles,D,errors = plot_with_errors_sample_z(res, params, initial_conditions, df, mobility_df, state, extrapolate=1, boundary=boundary, plot_asymptomatic_infectious=False,plot_symptomatic_infectious=False)
        #perr = get_errors(res,np.zeros((11,1)))
        #print('parameter standard deviations = ',perr)

    # %%
    state_to_maxdeaths = dict()
    for state in list_of_states:
        d = select_region(df,state,min_deaths = -1)
        state_to_maxdeaths[state] = d['deaths'].max()

    # maxdeaths_to_state = {v: k for k, v in state_to_maxdeaths.items()} # NOT INVERTIBLE!

    # %%
    list_of_low_death_states = []
    for state in list_of_states:
        if state_to_maxdeaths[state]<=50:
            list_of_low_death_states.append(state)

    # %% 
    state_to_dayzero = dict()
    for state in list_of_states:
        if list_of_low_death_states.count(state)==0:
            print(state)
            d = select_region(df,state,min_deaths = 50)
        # else:
        #     d = select_region(df,state,min_deaths = 1)
        dayzero = d['date_timestamp'].min()
        state_to_dayzero[state] = dayzero

    # %%
    state_to_daysofdata = dict()
    for state in list_of_states:
        if list_of_low_death_states.count(state)==0:
            d = select_region(df,state,min_deaths = 50)
        # else:
        #     d = select_region(df,state,min_deaths = 1)
        # daysofdata = (d['date_timestamp'].max() - d['date_timestamp'].min())/np.timedelta64(1, 'D')
        daysofdata = len(d)
        state_to_daysofdata[state] = daysofdata

    # %%
    state_to_fips = dict()
    for state in list_of_states:
        temp = select_region(df,state,min_deaths = -1)
        state_to_fips[state] = temp['fips'].iloc[0]

    # %% 
    # Prepare for parallelization
    # Number of cores to use
    workers = 8
    if workers > multiprocessing.cpu_count():
        raise('More workers requested than cores: workers=%d, cores=%d'%(workers, multiprocessing.cpu_count()))

    #-- Split into parallelizable chunks
    states_to_consider = [list_of_low_death_states.count(i) for i in list_of_states]
    states_to_consider = np.array(list_of_states)[np.array(states_to_consider)==0]
        # Here I am splitting arbitrarily. Could consider ordering and then splitting s.t. we evenly distribute the work
        # among the cores. This'll reduce net runtime since no core will be running longer than the others
    states_for_cores = np.array_split(states_to_consider,workers)
    # Split the dataframes by core allocations
    main_dfs = [df[df.state.isin(states_in_core)] for states_in_core in states_for_cores]
    mobility_dfs = [mobility_df[mobility_df.state.isin(states_in_core)] for states_in_core in states_for_cores]


    #-- Simplify arguments for function by placing them in a dict
    global_dayzero = pd.to_datetime('2020 Jan 21')
    global_end_day = pd.to_datetime('2020 June 30')
    num_days = int((global_end_day-global_dayzero)/np.timedelta64(1, 'D'))

    param_ranges = [(1.0, 3.0), (0.1, 0.5), (0.01, .9), (0.0001, 0.9), (0.0001, 0.9), (0.001, 0.1), (0.05, .7)] # shift can go 0 to 100 for alex method
    initial_ranges = [(1.0e-7, 0.05), (1.0e-7, 0.05), (1.0e-7, 0.05), (1.0e-7, 0.01)]
    params = [1.8, 0.35, 0.1, 0.15, 0.34, 0.015, .5]
    initial_conditions = [4e-6, 0.005, 0.005, 0.005]

    guesses = params + initial_conditions
    ranges = param_ranges + initial_ranges

    const = dict()
    const['state_to_daysofdata'] = state_to_daysofdata
    const['global_day0'] = global_dayzero
    const['state_to_day0'] = state_to_dayzero
    const['num_days'] = num_days
    const['guesses'] = guesses
    const['ranges'] = ranges
    const['params'] = params
    const['initial_conditions'] = initial_conditions
    const['state_to_fips'] = state_to_fips


    # %% 
    # Call to run in parallel
    args = [(states_for_cores[i], main_dfs[i], mobility_dfs[i], i, const) for i in range(workers)]
    res = apply_by_mp(par_fun,workers,args)


# savedict = {'cube':cube}
# savemat('wkspc.mat',savedict)
