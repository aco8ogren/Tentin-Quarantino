
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

from skopt import gp_minimize
from skopt.plots import plot_convergence
import numpy as np
from Dan.format_sub import format_file_for_evaluation
from Dan.EpidModel_parallelized_Counties import SEIIRQD_model
from Alex.copy_of_evaluator import evaluate_predictions

# %%
def test_error(HYPERPARAMS,train_til,test_from,test_til): 
    # Given a set of hyperparameters and days to train from and until,
    # this function trains a model, then evaluates and returns pinball loss
    temp_raw_npy = 'temp_raw_snowflake.npy'
    temp_raw_mat = 'temp_raw_snowflake.mat'
    temp_processed_csv = 'temp_processed_snowflake.csv'
    SEIIRQD_model(HYPERPARAMS = HYPERPARAMS,isSaveRes = True,sv_flnm_np = temp_raw_npy,
                    sv_flnm_mat = temp_raw_mat,isMultiProc = True,workers = 20,train_til = train_til,
                    train_Dfrom = 7,min_train_days = 5,isSubSelect = False, # CHANGE isSubSelect TO FALSE WHEN DONE DEBUGGING! New York is 36061
                    just_train_these_fips = None,isPlotBokeh = False,
                    isConstInitCond=True,init_vec=(2, .85, 3))
    format_file_for_evaluation(temp_raw_mat,temp_processed_csv,isAllocCounties = False,isComputeDaily = True)
    
    # score = score_all_predictions('temp_processed.csv', date, model_date, mse=False, key='cases', bin_cutoffs=[20, 1000])
    score = evaluate_predictions(temp_processed_csv,test_from,end_date = test_til)
    return score

def f(HYPERPARAMS):
    train_til = '2020 04 24'
    test_from = '2020-04-24'
    test_til  = '2020-05-05'
    return test_error(HYPERPARAMS,train_til,test_from,test_til)


# %%
if __name__ == '__main__':
    res = gp_minimize(f,                  # the function to minimize
                                        # the bounds on each dimension of x
                    [
                            (0,.1),       # p_err_frac: Parameter error estimate fraction (i.e. .05 --> 5% error)
                            (30,100),             # D_THRES: If a state does not have more than this number of deaths by train_til, we do not make predictions (or, we make cluster predictions)
                            (5,15),             # death_weight: factor by which to weigh error for death data more than symptomatic infected data during SEIIRQD optimization
                            (0,.5)            # alpha: the alpha from LeakyReLU determines how much to penalize the SEIIRQD objective function for over predicting the symptomatic infected
                    ],      
                    acq_func="EI",      # the acquisition function
                    n_calls=30,          # the number of evaluations of f
                    n_random_starts=10,  # the number of random initialization points
                    noise=0.1**2,       # the noise level (optional)
                    random_state=1234,  # the random seed
                    verbose = True)   
# %%
    fig = plot_convergence(res)
    fig.figure.savefig("Alex/bayesian_optimization_plots/bayesian_optimization_convergence_snowflake.pdf")
    print(res)
#%%


