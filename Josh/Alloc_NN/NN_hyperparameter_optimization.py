
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
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
import numpy as np
from Dan.format_sub import format_file_for_evaluation
from Dan.EpidModel_parallelized_Counties import SEIIRQD_model
from Alex.copy_of_evaluator import evaluate_predictions

# %%
def test_error(HYPERPARAMS,train_til,test_from,test_til): 
    temp_processed_csv = r'Josh\NN_opt\temp_NN_opt.csv'
    numDeaths, numCases, numMobility, lenOutput, remove_sparse, Patience, DropoutRate = HYPERPARAMS

    format_file_for_evaluation( input_fln='clustering.mat',
                                output_fln=temp_processed_csv,
                                isAllocCounties = False,
                                isComputeDaily = True,
                                alloc_day=train_til,
                                num_alloc_days=5,
                                isAllocNN=True,
                                retrain=True,
                                numDeaths=numDeaths,
                                numCases=numCases,
                                numMobility=numMobility,
                                lenOutput=lenOutput,
                                remove_sparse=remove_sparse,
                                Patience=Patience,
                                DropoutRate=DropoutRate,
                                modelDir=r'Josh\Alloc_NN\NN_opt\Model')
    
    # score = score_all_predictions('temp_processed.csv', date, model_date, mse=False, key='cases', bin_cutoffs=[20, 1000])
    score = evaluate_predictions(temp_processed_csv,test_from,end_date = test_til)
    return score

def f(HYPERPARAMS):
    train_til = '2020 05 10'
    test_from = '2020 05 10'
    test_til  = None
    
    return test_error(HYPERPARAMS,train_til,test_from,test_til)


# %%
if __name__ == '__main__':
    checkpoint_saver = CheckpointSaver(r'Josh\Alloc_NN\NN_opt\checkpoints', compress=9) # keyword arguments will be passed to `skopt.dump`

    res = gp_minimize(f,                  # the function to minimize
                                        # the bounds on each dimension of x
                    [
                            (0,10),         # number of days of death data to use as input
                            (0,10),         # number of days of cases data to use as input
                            (0,10),         # number of days of mobility data to use as input
                            (1,10),         # number of days of death data to average over for target
                            [True,False],    # boolean to remove sparse data
                            (1,10),         # patience of early stopping criteria on validation loss
                            (.05,.9)       # dropout rate
                    ],   
                    # x0 = [.1,100,5,0,4.901,0.020,0.114],   
                    # y0 = [],
                    acq_func="EI",      # the acquisition function
                    n_calls=3,          # the number of evaluations of f
                    n_random_starts=2,  # the number of random initialization points
                    noise=0.1**2,       # the noise level (optional)
                    random_state=1234,  # the random seed
                    callback = [checkpoint_saver],
                    verbose = True)   
# %%
    fig = plot_convergence(res)
    fig.figure.savefig(r"Josh\Alloc_NN\NN_opt\NN_optimization_convergence.pdf")
    print(res)