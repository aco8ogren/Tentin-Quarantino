# %% Imports and path setup

import os
import sys

HomeDIR='Tentin-Quarantino'
wd=os.path.dirname(os.path.realpath(__file__))
DIR=wd[:wd.find(HomeDIR)+len(HomeDIR)]
os.chdir(DIR)

sys.path.append(os.getcwd())

from Dan.EpidModel_parallelized_Counties import SEIIRQD_model
from Alex.copy_of_evaluator import evaluate_predictions
from Dan.format_sub import format_file_for_evaluation

# NOTE: Apparently we still need to shield. Can't say I understand why...
if __name__ == '__main__':

# %% Setup model training run

    #-- Flag to choose whether to train the model
        # If this is true, the output file from this run will be used for
        # the remainder of the sections
    isTrainModel = True

    #-- Define control parameters
    # Flag to choose whether to save the results or not
    isSaveRes = True
    # Filename for saved .npy and .mat files (can include path)
        # Make sure the directory structure is present before calling
    sv_flnm_mat = 'Dan\\PracticeOutputs\\TestScript.mat'
    sv_flnm_np  = os.path.splitext(sv_flnm_mat)[0] + '.npy'


    #-- Multiprocessing settings
    # Flag to choose whether multiprocessing should be used
    isMultiProc = True
    # Number of cores to use (logical cores, not physical cores)
    workers = 8


    #-- Filtering parameters
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
    train_Dfrom = 7
    # Minimum number of days required for a county to be trained on
        # After filtering using train_Dfrom and D_THRES, this makes sure that there
        # are at least min_train_days worth of days to train the model on (for fit_leastsqz)
    min_train_days = 5


    #-- Sub-select counties to train on
    # Flag to choose whether to sub-select
    isSubSelect = True
    # List of counties which should be considered
        # NOTE: This just removes ALL other counties from the df as soon as it can
    just_train_these_fips = [36061, 1073, 56035, 6037] 


    #-- Method used for choosing initial conditions
        # True: Use the same vector (hardcoded) as the initial conditions for all counties
        # False: Calculate unique initial conditions for each county 
    isConstInitCond = True


    #-- When not multiprocessing, enable bokeh plotting (since won't cause issue)
    # Flag to stating whether to plot. This only matters when not multiprocessing (isMultiProc=False)
        # When isMultiProc=True, bokeh will cause errors so we ignore this flag
    isPlotBokeh     = False


    #-- Set hyperparameters
    # NOTE/TODO: Alex to explain what these are
    p_err_frac = 0.05   
    death_weight = 10
    alpha = 0.2


# %% Setup Formatter run

    #-- Flag to choose whether to format a model .mat
    isFormat = True

    #-- Define control parameters
    # Flag to distribue state deaths amongst counties
    isAllocCounties = False
    # Flag to translate cummulative data to daily counts
    isComputeDaily = True

    #-- When a model was not trained, provide filename to format
        # if a model was trained, that filename will automatically be used
    format_flnm_in = 'Dan/PracticeOutputs/Blah.mat'

    #-- Provide filename for output file (if isFormat=True, we'll always)
    format_flnm_out = os.path.splitext(format_flnm_in)[0] + '.csv'



# %% Setup evaluator run

    #-- Flag to choose whether to evaluate a .csv file
    isEval = True

    #-- When model was not formatted, provide a filename to evaluate
        # if a model was trained and formatted, that filename will automatically be used
    eval_flnm_in = 'Dan/PracticeOutputs/Foo.csv'

    #-- Day from which we should evaluate 
        # in format 'YYYY-MM-DD'
    eval_start_day = '2020-04-24'

    #-- Day until which we should evaluate
        # in format 'YYYY-MM-DD'
        # Set to None to evaluate until most recent day of data
    eval_end_day = '2020-05-05'


# %% Prepare for calls

    #-- If the user says to train a model but doesn't save the result, we can't
        # run the remaining sections since we won't have the results to format/eval
    if isTrainModel and (not isSaveRes):
        if isFormat:
            raise ValueError("isTrainModel=True but isSaveRes=false so we can't format the file.")
        if isEval:
            raise ValueError("isTrainModel=True but isSaveRes=false so we can't evaluate the file.")

    #-- If the user trains a model, use the output to format
    if isFormat and isTrainModel:
        format_flnm_in = sv_flnm_mat
        format_flnm_out = os.path.splitext(format_flnm_in)[0] + '.csv'

    # If the user formats a model, use the output to evaluate
    if isEval and isFormat:
        eval_flnm_in = format_flnm_out


# %% Run sections as needed

    if isTrainModel:
        print('\n\n------ Training Model ------\n')
        SEIIRQD_model(HYPERPARAMS = (p_err_frac,D_THRES,death_weight,alpha),
                        isSaveRes = isSaveRes,
                        sv_flnm_np=sv_flnm_np,
                        sv_flnm_mat = sv_flnm_mat,
                        isMultiProc = isMultiProc,
                        workers = workers,
                        train_til = train_til,
                        train_Dfrom = train_Dfrom,
                        min_train_days = min_train_days,
                        isSubSelect = isSubSelect,
                        just_train_these_fips = just_train_these_fips,
                        isPlotBokeh = isPlotBokeh, 
                        isConstInitCond = isConstInitCond)

        print('*** Model results saved to:\n    %s\n    %s'%(sv_flnm_mat, sv_flnm_np))


    if isFormat:
        print('\n\n------ Formatting File ------')

        print('*** Input filename:\n    %s'%format_flnm_in)

        format_file_for_evaluation(format_flnm_in,
                                    format_flnm_out,
                                    isAllocCounties = isAllocCounties,
                                    isComputeDaily = isComputeDaily)

        print('*** Formatted file:\n    %s'%format_flnm_out)
        

    if isEval:
        print('\n\n------ Evaluating File ------')

        print('*** Input filename:\n    %s'%eval_flnm_in)

        print('\n\n')
        score = evaluate_predictions(eval_flnm_in,
                                        eval_start_day,
                                        end_date = eval_end_day)

                