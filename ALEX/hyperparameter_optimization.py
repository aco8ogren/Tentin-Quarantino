
#hyperparameter_optimization
# %%
from skopt import gp_minimize
from skopt.plots import plot_convergence
import numpy as np
from model_scoring import score_all_predictions
from format_sub import format_file_for_evaluation

# %%
def test_error(HYPERPARAMS,train_til,test_from): 
    # Given a set of hyperparameters and days to train from and until,
    # this function trains a model, then evaluates and returns pinball loss
    SEIIRQD_model(HYPERPARAMS = (.05,50,10,.2),isSaveRes = True,sv_flnm_np='temp_raw.npy',
                    sv_flnm_mat = 'temp_raw.mat',isMultiProc = False,workers = 1,train_til = train_til,
                    train_Dfrom = 7,min_train_days = 5,isSubSelect = False,
                    just_train_these_fips = [],isPlotBokeh = False)
    format_file_for_evaluation('temp_raw.mat','temp_processed.csv',isAllocCounties = True,isComputeDaily = True)
    score = score_all_predictions('temp_processed.csv', date, model_date, mse=False, key='cases', bin_cutoffs=[20, 1000])
    return score[0]


'2020 04 24'

# %%
if 
res = gp_minimize(f,                  # the function to minimize
                                      # the bounds on each dimension of x
                  [
                        (-2.0, 2.0),       # p_err_frac: Parameter error estimate fraction (i.e. .05 --> 5% error)
                        (1,5),             # D_THRES: If a state does not have more than this number of deaths by train_til, we do not make predictions (or, we make cluster predictions)
                        (4,7),             # death_weight: factor by which to weigh error for death data more than symptomatic infected data during SEIIRQD optimization
                        (9, 10)            # alpha: the alpha from LeakyReLU determines how much to penalize the SEIIRQD objective function for over predicting the symptomatic infected
                  ],      
                  acq_func="EI",      # the acquisition function
                  n_calls=100,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)   # the random seed

# %%
plot_convergence(res)
#%%
    pred_file = os.path.join(os.getcwd(), 'erf_model_predictions_0413.csv')
    scores = score_all_predictions(pred_file, '2020-04-14', '2020-04-13', key='deaths')

