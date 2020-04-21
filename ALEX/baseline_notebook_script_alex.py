# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
from IPython import get_ipython

# %% [markdown]
#  # COVID-19 Epidemiology Models
# %% [markdown]
#  First, some preliminary imports. You may need to pip install `holoviews` and `GitPython`. For `holoviews`, some extras might be needed (see https://holoviews.org/install.html).

# %%
import pandas as pd
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

import bokeh.io
import bokeh.application
import bokeh.application.handlers
import bokeh.models

from scipy.optimize import least_squares
from bokeh.models import Span

import itertools

import holoviews as hv

bokeh.io.output_notebook()
hv.extension('bokeh')

# %% [markdown]
#  Let's load the data from the relevant folder. If this data doesn't exist for you, you'll need to run the `processing/raw_data_processing/daily_refresh.sh` script (which may require `pip install us`).

# %%
import git

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
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
df['date_processed'] = (df['date_processed'] - df['date_processed'].min()) / np.timedelta64(1, 'D')

# %% [markdown]
# Now we import the population for each state, and add that information correctly to the Pandas "dataframe".

# %%
populations = pd.read_csv(datadir + 'demographics/state_populations_augmented.csv')
def get_population(region):
    return populations[populations['state'] == region]['population'].values[0]

def get_ticker(state):
    state_to_ticker = dict([('CA','California'),('TX','Texas'),('FL','Florida'),('NY','New York'),('PA','Pennsylvania'),('IL','Illinois'),('OH','Ohio'),('GA','Georgia'),('NC','North Carolina'),
                           ('MI','Michigan'),('NJ','New Jersey'),('VA','Virginia'),('WA','Washington'),('AZ','Arizona'),('MA','Massachusetts'),('TN','Tennessee'),('IN','Indiana'),('MO','Missouri'),
                           ('MD','Maryland'),('WI','Wisconsin'),('CO','Colorado'),('MN','Minnesota'),('SC','South Carolina'),('AL','Alabama'),('LA','Louisiana'),('KY','Kentucky'),('OR','Oregon'),
                           ('OK','Oklahoma'),('CT','Connecticut'),('UT','Utah'),('IA','Iowa'),('NV','Nevada'),('AR','Arkansas'),('MS','Mississippi'),('KS','Kansas'),('NM','New Mexico'),('NE','Nebraska'),
                           ('WV','West Virginia'),('ID','Idaho'),('HI','Hawaii'),('NH','New Hampshire'),('ME','Maine'),('MT','Montana'),('RI','Rhode Island'),('DE','Delaware'),('SD','South Dakota'),
                           ('ND','North Dakota'),('AK','Alaska'),('DC','District of Columbia'),('VT','Vermont'),('WY','Wyoming'),('PR','Puerto Rico'),('VI','Virgin Islands'),('GU','Guam'),
                           ('NMI','Northern Mariana Islands')])
    state_to_ticker = {v: k for k, v in state_to_ticker.items()} # I accidentally typed the dictionary backwards (:
    return state_to_ticker[state]

# %% [markdown]
# Let's also include the state ticker (i.e. Minnesota --> MN) as a column in the Pandas dataframe. (Because the US population info is given by state tickers).

# %%
df['state_ticker'] = df.apply(lambda row: get_ticker(row.state),axis = 1)
df['Population'] = df.apply(lambda row: get_population(row.state_ticker), axis=1)

# %% [markdown]
#  Just checking to make sure the data is here...

# %%
df.head()

# %% [markdown]
#  Great! Let's also make a helper function to select data from a state, starting when the pandemic hit to be able to fit models. 
#  
#  We could change the parameter "min_deaths" here if we want... it seems like an arbitrary choice.

# %%
# return data ever since first min_cases cases
def select_region(df, region, min_deaths=50):
    d = df.loc[df['state'] == region]
    start = np.where(d['deaths'].values > min_deaths)[0][0]
    d = d[start:]
    return d

# %% [markdown]
#  Define a funciton for calculating the "errors" from the least_squares model.
#  
#  The "errors" here actually seem to be the std of the parameters from our model. 
#    This std is calculated from the jacobian returned by the least_squares function.
#    It is currently unclear how the TA's obtained this conversion (res.jac --> std(params))

# %%
from scipy.linalg import svd
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
# TODO fix data imputation
def q(t, N, shift):
    moving = np.array([57, 54, 52, 51, 49, 47, 46, 45, 44, 43, 39, 37, 34, 23, 19, 13, 10, 7, 6, 5, 5, 7, 6, 5, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 2])/100
    q = N*(1-moving) - shift*N
    if np.round(t) >= len(q):
        return q[-1]
    return q[int(np.round(t))]
    
def tau(t):
    return 0

def seiirq(dat, t, params, N, max_t, offset):
    if t >= max_t:
        return [0]*8
    beta = params[0]
    alpha = params[1] # rate from e to ia
    sigma = params[2] # rate of asymptomatic people becoming symptotic
    ra = params[3] # rate of asymptomatic recovery
    rs = params[4] # rate of symptomatic recovery
    delta = params[5] # death rate
    shift = params[6] # shift quarantine rate vertically from CityMapper data
    
    s = dat[0]
    e = dat[1]
    i_a = dat[2]
    i_s = dat[3]

    Qind = (q(t + offset, N, shift) - tau(t + offset)*i_a)/(s + e + i_a - tau(t + offset)*i_a)
    Qia = Qind + (1-Qind)*tau(t + offset)
    
    dsdt = - beta * s * i_a * (1 - Qind) * (1 - Qia) / N
    dedt = beta * s * i_a* (1 - Qind) * (1 - Qia) / N  - alpha * e
    diadt = alpha * e - (sigma + ra) * i_a
    disdt = sigma * i_a - (delta + rs) * i_s
    dddt = delta * i_s
    drdt = ra * i_a + rs * i_s
    
    
    # susceptible, exposed, infected, quarantined, recovered, died, unsusceptible
    out = [dsdt, dedt, diadt, disdt, drdt, dddt]
    return out


# %%
from sklearn.metrics import mean_squared_error

def mse(A, B):
    Ap = np.nan_to_num(A)
    Bp = np.nan_to_num(B)
    Ap[A == -np.inf] = 0
    Bp[B == -np.inf] = 0
    Ap[A == np.inf] = 0
    Bp[B == np.inf] = 0
    return mean_squared_error(Ap, Bp)

def model_z(params, data, tmax=-1):
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
        n = tmax
    
    # Package parameters into a tuple
    args = (params, N, n, offset)
    
    # Integrate ODEs
    try:
        s = scipy.integrate.odeint(seiirq, yz_0, np.arange(0, n), args=args)
    except RuntimeError:
#         print('RuntimeError', params)
        return np.zeros((n, len(yz_0)))

    return s

def fit_leastsq_z(params, data):
    Ddata = (data['deaths'].values)
    Idata = (data['cases'].values)
    s = model_z(params, data)

    S = s[:,0]
    E = s[:,1]
    I_A = s[:,2]
    I_S = s[:,3]
    R = s[:,4]
    D = s[:,5]
    
    error = np.concatenate((D-Ddata, I_S - Idata))
    return error

# %% [markdown]
#  Again, we find some good initial parameters.
# 
#  *** We need to find some good initial parameters. ):

# %%
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()
d = select_region(df, 'New York')
# parameters: beta, alpha, sigma, ra, rs, delta, shift
params = [1.8, 0.35, 0.1, 0.15, 0.34, 0.015, 0.5]
# conditions: E, IA, IS, R
initial_conditions = [4e-6, 0.0009, 0.0005, 0.0002]
s = model_z(params + initial_conditions, d)
plt.scatter(d['date_processed'], d['deaths'])
plt.plot(d['date_processed'], s[:, 5])
plt.show()


# %%
def plot_with_errors_sample_z(res, p0_params, p0_initial_conditions, df, region, extrapolate=1, boundary=None, plot_infectious=False):
    data = select_region(df, region)
    # errors = get_errors(res, list(p0_params) + list(p0_initial_conditions))
    errors = res.x*.05 # ALEX --> the parameter error is simply 5% of the parameter. i.e. we know the parameter to within +or- 5%.
    #errors[len(p0_params):] = 0
    
    all_s = []
    samples = 100
    for i in range(samples):
        sample = np.random.normal(loc=res.x, scale=errors)
        s = model_z(sample, data, len(data)*extrapolate)
        all_s.append(s)
        
    all_s = np.array(all_s)
    
    s = model_z(res.x, data, len(data)*extrapolate)
    S = s[:,0]
    E = s[:,1]
    I_A = s[:,2]
    I_S = s[:,3]
    R = s[:,4]
    D = s[:,5]

    t = np.arange(0, len(data))
    tp = np.arange(0, len(data)*extrapolate)

    p = bokeh.plotting.figure(plot_width=600,
                              plot_height=400,
                             title = region + ' SEIIRD+Q Model',
                             x_axis_label = 't (days)',
                             y_axis_label = '# people')


    quantiles = [10, 20, 30, 40]
    for quantile in quantiles:
        s1 = np.percentile(all_s, quantile, axis=0)
        s2 = np.percentile(all_s, 100-quantile, axis=0)
        if plot_infectious:
            p.varea(x=tp, y1=s1[:, 2], y2=s2[:, 2], color='red', fill_alpha=quantile/100) # Asymptomatic infected
        p.varea(x=tp, y1=s1[:, 3], y2=s2[:, 3], color='purple', fill_alpha=quantile/100) # Symptomatic infected
        p.varea(x=tp, y1=s1[:, 5], y2=s2[:, 5], color='black', fill_alpha=quantile/100) # deaths
    
    if plot_infectious:
        p.line(tp, I_A, color = 'red', line_width = 1, legend_label = 'Asymptomatic infected')
    p.line(tp, D, color = 'black', line_width = 1, legend_label = 'Deceased')
    p.line(tp, I_S , color = 'purple', line_width = 1, legend_label = 'Symptomatic infected')

    # death
    p.circle(t, data['deaths'], color ='black')

    # quarantined
    p.circle(t, data['cases'], color ='purple')
    
    if boundary is not None:
        vline = Span(location=boundary, dimension='height', line_color='black', line_width=3)
        p.renderers.extend([vline])

    p.legend.location = 'top_left'
    bokeh.io.show(p)

    D_quantiles = np.array([s1[:,5],s2[:,5]])
    D

    return all_s,D_quantiles,D,errors

# %% [markdown]
#  Let's define the initial ranges of the constants for the ODE.

# %%
# beta, alpha, sigma, ra, rs, delta, shift
param_ranges = [(1.0, 2.0), (0.1, 0.5), (0.1, 0.5), (0.05, 0.5), (0.32, 0.36), (0.005, 0.05), (0.1, 0.6)]
initial_ranges = [(1.0e-7, 0.001), (1.0e-7, 0.001), (1.0e-7, 0.001), (1.0e-7, 0.001)]

guesses = params + initial_conditions
ranges = param_ranges + initial_ranges

boundary = 16
res = least_squares(fit_leastsq_z, guesses, args=(select_region(df, 'Washington')[:boundary],), bounds=np.transpose(np.array(ranges)),jac = '3-point')
all_s,D_quantiles,D,errors = plot_with_errors_sample_z(res, params, initial_conditions, df, 'Washington', extrapolate=1, boundary=boundary, plot_infectious=True)
perr = get_errors(res,np.zeros((11,1)))
print('parameter standard deviations = ',perr)


# %%

# start = 8
# step = 4
# ind = 0
# results = []
# one_more = False
# while start + ind*step <= 18:
#     boundary = start + ind*step
#     res = least_squares(fit_leastsq_z, guesses, args=(select_region(df, 'New York')[:boundary],), bounds=np.transpose(np.array(ranges)))
#     plot_with_errors_sample_z(res, params, initial_conditions, df, 'New York', extrapolate=2, boundary=boundary, plot_infectious=True)
#     ind += 1

# %% [markdown]
# ***HUUUUGE uncertainty bars. Need to figure out why. Maybe because we don't have good fit parameters, so the covariance is huge.
# 
# 
#  Just like SEIR-QD, the predictions of cases aren't so great, although the predictions of fatalaties (which matters more and has less data bias) is reasonably accurate. The model also produces prediction that around 2/3 of the cases are asymptomatic (or at least not tested). This corresponds roughly to some recent studies, such as the 50-75% number reported after testing an entire town of 3,300 in Italy (https://www.repubblica.it/salute/medicina-e-ricerca/2020/03/16/news/coronavirus_studio_il_50-75_dei_casi_a_vo_sono_asintomatici_e_molto_contagiosi-251474302/?ref=RHPPTP-BH-I251454518-C12-P3-S2.4-T1) and similar results in Iceland (https://www.government.is/news/article/2020/03/15/Large-scale-testing-of-general-population-in-Iceland-underway/).

# %%


