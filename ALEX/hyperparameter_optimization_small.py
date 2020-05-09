
#hyperparameter_optimization
# %%
import os
import sys
HomeDIR='Tentin-Quarantino'
wd=os.path.dirname(os.path.realpath(__file__))

DIR=wd[:wd.find(HomeDIR)+len(HomeDIR)]
os.chdir(DIR)
sys.path.append(os.getcwd())

homedir = DIR
datadir = f"{homedir}"
# %%

from skopt import gp_minimize
from skopt.plots import plot_convergence
import numpy as np
from Dan.format_sub import format_file_for_evaluation
from Dan.EpidModel_parallelized_Counties import SEIIRQD_model
from Alex.copy_of_evaluator import evaluate_predictions

# %%
def test_error(HYPERPARAMS,train_til,test_from): 
    # Given a set of hyperparameters and days to train from and until,
    # this function trains a model, then evaluates and returns pinball loss
    SEIIRQD_model(HYPERPARAMS = HYPERPARAMS,isSaveRes = True,sv_flnm_np='temp_raw.npy',
                    sv_flnm_mat = 'temp_raw.mat',isMultiProc = True,workers = 6,train_til = train_til,
                    train_Dfrom = 7,min_train_days = 5,isSubSelect = False, # CHANGE isSubSelect TO FALSE WHEN DONE DEBUGGING! New York is 36061
                    just_train_these_fips = [],isPlotBokeh = False)
    format_file_for_evaluation('temp_raw.mat','temp_processed.csv',isAllocCounties = True,isComputeDaily = True)
    
    # score = score_all_predictions('temp_processed.csv', date, model_date, mse=False, key='cases', bin_cutoffs=[20, 1000])
    score = evaluate_predictions('temp_processed.csv',test_from)
    return score

def f(HYPERPARAMS):
    train_til = '2020 04 24'
    test_from = '2020-04-24'
    return test_error(HYPERPARAMS,train_til,test_from)


# %%
res = gp_minimize(f,                  # the function to minimize
                                      # the bounds on each dimension of x
                  [
                        (0,.1),       # p_err_frac: Parameter error estimate fraction (i.e. .05 --> 5% error)
                        (50,60),             # D_THRES: If a state does not have more than this number of deaths by train_til, we do not make predictions (or, we make cluster predictions)
                        (5,15),             # death_weight: factor by which to weigh error for death data more than symptomatic infected data during SEIIRQD optimization
                        (0,.5)            # alpha: the alpha from LeakyReLU determines how much to penalize the SEIIRQD objective function for over predicting the symptomatic infected
                  ],      
                  acq_func="EI",      # the acquisition function
                  n_calls=5,         # the number of evaluations of f
                  n_random_starts=1,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)   # the random seed

# %%
plot_convergence(res)
#%%


