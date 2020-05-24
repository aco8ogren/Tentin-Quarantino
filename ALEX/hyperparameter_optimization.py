
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
import numpy as np
from Dan.format_sub import format_file_for_evaluation
from Dan.EpidModel_parallelized_Counties import SEIIRQD_model
from Alex.copy_of_evaluator import evaluate_predictions
import time
import pandas as pd

# %%
def test_error(HYPERPARAMS,train_til,test_from,test_til): 
    # Given a set of hyperparameters and days to train from and until,
    # this function trains a model, then evaluates and returns pinball loss

    temp_raw_npy = 'temp_raw_snowflake.npy'
    temp_raw_mat = 'temp_raw_snowflake.mat'
    temp_processed_csv = 'temp_processed_snowflake.csv'
    p_err_frac, D_THRES, death_weight, alpha, ERF_THRES = HYPERPARAMS[0:5]
    init_vec = (HYPERPARAMS[5],HYPERPARAMS[6],HYPERPARAMS[7])
    cluster_max_radius = HYPERPARAMS[8]
    train_Dfrom = HYPERPARAMS[9]
    min_train_days = HYPERPARAMS[10]
    isAllocNN = HYPERPARAMS[11]

    print('======== CURRENT PARAMETERS BEING EVALUATED ========')
    print('p_err_frac =',p_err_frac)
    print('D_THRES =',D_THRES)
    print('death_weight =',death_weight)
    print('alpha =',alpha)
    print('ERF_THRES =',ERF_THRES)
    print('init_vec[0] =',init_vec[0])
    print('init_vec[1] =',init_vec[1])
    print('init_vec[2] =',init_vec[2])
    print('cluster_max_radius =',cluster_max_radius)
    print('train_Dfrom =',train_Dfrom)
    print('min_train_days =',min_train_days)
    print('isAllocNN =',isAllocNN)
    print('====================================================')

    # if HYPERPARAMS[7] == 1:
    #     isCluster = True
    SEIIRQD_model(HYPERPARAMS = [p_err_frac,D_THRES,death_weight,alpha,ERF_THRES],
                    isSaveRes = True,sv_flnm_np=temp_raw_npy,sv_flnm_mat = temp_raw_mat,
                    isMultiProc = True,workers = 20,
                    train_til = train_til,train_Dfrom = train_Dfrom,min_train_days = min_train_days,
                    isSubSelect = False,just_train_these_fips = None,
                    isPlotBokeh = False, isSaveMatplot = False, save_time = None, 
                    isConstInitCond = False, init_vec=init_vec,
                    verbosity = 2, least_squares_verbosity = 0,
                    isCluster=True, cluster_max_radius = cluster_max_radius)
        
        # HYPERPARAMS = HYPERPARAMS[0:4],isSaveRes = True,sv_flnm_np = temp_raw_npy,
        #             sv_flnm_mat = temp_raw_mat,isMultiProc = True,workers = 20,train_til = train_til,
        #             train_Dfrom = 7,min_train_days = 5,isSubSelect = False, # CHANGE isSubSelect TO FALSE WHEN DONE DEBUGGING! New York is 36061
        #             just_train_these_fips = None,isPlotBokeh = False,
        #             isConstInitCond=False,init_vec=HYPERPARAMS[4:],
        #             verbosity = 2,
        #             isCluster = isCluster)
    
    if isAllocNN:
        isAllocCounties = False
    else:      
        isAllocCounties = True

    format_file_for_evaluation( temp_raw_mat,
                                temp_processed_csv,
                                isAllocCounties = isAllocCounties,
                                isComputeDaily = True,
                                alloc_day = train_til,
                                num_alloc_days = 5,
                                isAllocNN = isAllocNN,
                                retrain = True,
                                numDeaths=2,
                                numCases=2,
                                numMobility=2,
                                lenOutput=6,
                                remove_sparse=True,
                                Patience=4,
                                DropoutRate=.1,
                                modelDir='Alex\\temp')
    
    # score = score_all_predictions('temp_processed.csv', date, model_date, mse=False, key='cases', bin_cutoffs=[20, 1000])
    score = evaluate_predictions(temp_processed_csv,test_from,end_date = test_til,only_score_these_fips = None)
    return score

def f(HYPERPARAMS):
    train_til = '2020-05-08'
    test_from = '2020-05-09'
    test_til  = None #'2020-05-05'
    if HYPERPARAMS[4] <= HYPERPARAMS[1]:
        return test_error(HYPERPARAMS,train_til,test_from,test_til)
    else:
        return .21 # if HYPERPARAMS[4] <= HYPERPARAMS[1] does not hold, then return a reasonable yet high error. This means ERF_THRES > D_THRES.


# %%
if __name__ == '__main__':
    checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9) # keyword arguments will be passed to `skopt.dump`
    tic = time.time()
    n_calls = 100
    n_random_starts = 50
    res = gp_minimize(f,                  # the function to minimize
                                        # the bounds on each dimension of x
                    [
                            (0,.1),        # p_err_frac: Parameter error estimate fraction (i.e. .05 --> 5% error)
                            (30,1000),     # D_THRES: If a state does not have more than this number of deaths by train_til, we do not make predictions (or, we make cluster predictions)
                            (1,15),        # death_weight: factor by which to weigh error for death data more than symptomatic infected data during SEIIRQD optimization
                            (0,.5),        # alpha: the alpha from LeakyReLU determines how much to penalize the SEIIRQD objective function for over predicting the symptomatic infected
                            (0,1000),      # ERF_THRES
                            (0.1,5),       # init_vec #1
                            (.001,1),      # init_vec #2
                            (.0001,10),    # init_vec #3
                            (0.0,4.0),     # cluster_max_radius
                            (1,20),        # train_Dfrom
                            (5,15)         # min_train_days
                    ],   
                    # x0 = [.1,100,5,0,4.901,0.020,0.114],   
                    # y0 = [],
                    acq_func="EI",      # the acquisition function
                    n_calls=n_calls,          # the number of evaluations of f
                    n_random_starts=n_random_starts,  # the number of random initialization points
                    noise=0.1**2,       # the noise level (optional)
                    random_state=1234,  # the random seed
                    callback = [checkpoint_saver],
                    verbose = True)   
    toc = time.time()

# %%
    fig = plot_convergence(res)
    fig.figure.savefig("Alex/bayesian_optimization_plots/bayesian_optimization_convergence_snowflake.pdf")
    print(res)

    print('===== BAYESIAN OPTIMIZATION COMPLETE =====')
    print('Time elapsed (total):',round(toc-tic),'sec')
    print('Total number of iterations:',n_calls)
    print('Best score:',res.fun)

    HYPERPARAMS = res.x
    p_err_frac, D_THRES, death_weight, alpha, ERF_THRES = HYPERPARAMS[0:5]
    init_vec = (HYPERPARAMS[5],HYPERPARAMS[6],HYPERPARAMS[7])
    cluster_max_radius = HYPERPARAMS[8]
    train_Dfrom = HYPERPARAMS[9]
    min_train_days = HYPERPARAMS[10]

    print('~~~~~~~~~~~~~ OPTIMAL VALUES ~~~~~~~~~~~~~')
    print('p_err_frac =',p_err_frac)
    print('D_THRES =',D_THRES)
    print('death_weight =',death_weight)
    print('alpha =',alpha)
    print('ERF_THRES =',ERF_THRES)
    print('cluster_max_radius =',cluster_max_radius)
    print('train_Dfrom =',train_Dfrom)
    print('min_train_days =',min_train_days)
    print('init_vec[0] =',init_vec[0])
    print('init_vec[1] =',init_vec[1])
    print('init_vec[2] =',init_vec[2])
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    params = 'p_err_frac, D_THRES, death_weight, alpha, ERF_THRES, init_vec[0], init_vec[1], init_vec[2], cluster_max_radius, train_Dfrom, min_train_days'.split(', ')
    Results = pd.DataFrame(res.x_iters, columns = params)
    Results['Loss']=res.func_vals
    Results.to_csv('Alex\\bayesian_optimization_plots\\bayesian_optimization_record_snowflake.csv')
    print(Results)

#%%


