
if __name__ == '__main__':
    import os
    import sys

    HomeDIR='Tentin-Quarantino'
    wd=os.path.dirname(os.path.realpath(__file__))
    DIR=wd[:wd.find(HomeDIR)+len(HomeDIR)]
    os.chdir(DIR)

    # -%% Imports and path setup

# NOTE: Apparently we still need to shield. Can't say I understand why...

    sys.path.append(os.getcwd())

    from Dan.EpidModel_parallelized_Counties import SEIIRQD_model
    from Alex.copy_of_evaluator import evaluate_predictions
    from Dan.format_sub import format_file_for_evaluation


# -%% Setup model training run

    #-- Flag to choose whether to train the model
        # If this is true, the output file from this run will be used for
        # the remainder of the sections
    isTrainModel = False
    #-- Define control parameters
    # Flag to choose whether to save the results or not
    isSaveRes = False
    # Filename for saved .npy and .mat files (can include path)
        # Make sure the directory structure is present before calling
        # NOTE: when clustering, the .mat filename will be used for saving the cluster file
    sv_flnm_mat = 'Alex\\PracticeOutputs\\detective_work.mat'
    sv_flnm_np  = os.path.splitext(sv_flnm_mat)[0] + '.npy'


    #-- Multiprocessing settings
    # Flag to choose whether multiprocessing should be used
    isMultiProc = False
    # Number of cores to use (logical cores, not physical cores)
    workers = 20


    #-- Filtering parameters
    # Threshold of deaths at and below which a COUNTY will not be trained on
        # Filters which COUNTIES are looped over in optimization/minimization loop
    D_THRES = 97
    # Last day used for training (good for testing)
        # must be a valid pandas.to_datetime() string
        # OR: leave as None to train until the latest data for which there is data
    train_til = '2020 05 10'
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


    #-- Clustering settings
    # Enable clustering: combines the low-death counties into clusters for training
        # When False, the code will run as it used to
    isCluster = True

    cluster_max_radius = 2


    #-- Sub-select counties to train on
    # Flag to choose whether to sub-select
    isSubSelect = False
    # List of counties which should be considered
        # NOTE: This just removes ALL other counties from the df as soon as it can
    just_train_these_fips = [21131, 21051, 21193, 21119, 21109, 21189, 21025, 21071, 21115,
       21197, 21175, 21165, 21049, 21173, 21127, 21011, 21205, 21043,
       21181, 21069, 21019, 39087, 21089, 21135, 21161, 21023, 39145,
       39001, 39015, 39079, 39131, 39025, 39071, 39141, 39027, 39047,
       39129, 39057, 39113, 39045, 39097, 39023, 39109] # GOOD 6037,17031, TROUBLE 53061,36059,53033  NOT SURE 36087
    #[36061, 36059, 26163, 17031, 36103, 36119, 34013, 34003, 6037,  9001,  34017, 26125, 25017, 34039, 26099, 9003] 


    #-- Method used for choosing initial conditions
        # True: Use the same vector (hardcoded) as the initial conditions for all counties
        # False: Calculate unique initial conditions for each county 
    isConstInitCond = True
    # When calculating unique conditions for each county, define fudge factors:
    init_vec = (4.901,          # T : Is = T*cases      Old: 3.933
                0.020,          # R : Ia = R*Itot       Old: 0.862
                0.114)          # F : E  = F*Is         Old: 3.014


    #-- When not multiprocessing, enable bokeh plotting (since won't cause issue)
    # Flag to stating whether to plot. This only matters when not multiprocessing (isMultiProc=False)
        # When isMultiProc=True, bokeh will cause errors so we ignore this flag
    isPlotBokeh     = True


    #-- Set verbosity for printing
    #-- Verbosity explanation:
    # There are multiple levels of verbosity based on the provided integer
    #   0 :     No print statements are executed
    #   1 :     Only total time is printed
    #   2 :     Only prints in main function are shown (those in par_fun are suppressed)
    #   3 :     (DEFAULT) All print statements are executed
    # *** Error-related prints are always printed
    verbosity = 3

    #-- Set hyperparameters
    p_err_frac = 0.0995764604328379   # The size of the uncertainty that we have on our optimal SEIIRQD parameters. This affects the size of our quantile differences.
    death_weight = 5   # The weight with which we multiply the death error in SEIIRQD optimization. The death data is trusted death_weight times more than the symptomatic infected data.
    alpha = 0.00341564933361549         # alpha of the LeakyReLU for modifying the symptomatic infected error. i.e. if alpha = 0 ==> no penalty for overestimating Sympt Inf. alpha = 1 ==> as much penalty for overestimating as underestimating.

# -%% Setup Formatter run

    #-- Flag to choose whether to format a model's .mat output file
    isFormat = True

    #-- Define control parameters
    # Flag to distribue state deaths amongst counties
    isAllocCounties = False
    # Allocating using the mean number of num_alloc_days days BEFORE alloc_day
    num_alloc_days=5
    alloc_day=train_til
    # Flag to translate cummulative data to daily counts
    isComputeDaily = True

    # Flag to distribute deaths with neural net
    isAllocNN=True
    # Number of days of death inputs
    numDeaths=2
    # Number of days of cases inputs
    numCases=2
    # Number of days of mobility inputs
    numMobility=2
    # Number of days of deaths outputs to average over
    lenOutput=5
    # Flag to remove data points with zero deaths for inputs and output
    remove_sparse=True
    # Flag to retrain neural net or just look for model in directory
    retrain=True
    # Directory to save or load model from
    modelDir=r'Josh\Alloc_NN\ModelSaves\FirstNet'



    #-- When a model was not trained, provide filename to format
        # if a model was trained, that filename will automatically be used
    format_flnm_in = 'clusteringCopy.mat'

    #-- Provide filename for output file 
    format_flnm_out = os.path.splitext(format_flnm_in)[0] + '.csv'



# -%% Setup evaluator run

    #-- Flag to choose whether to evaluate a .csv file
    isEval = True
    #-- When model was not formatted, provide a filename to evaluate
        # if a model was formatted, that filename will automatically be used
    eval_flnm_in = 'clusteringCopy.csv'

    #-- Day from which we should evaluate 
        # in format 'YYYY-MM-DD'
    eval_start_day = '2020-05-10'

    #-- Day until which we should evaluate
        # in format 'YYYY-MM-DD'
        # Set to None to evaluate until most recent day of data
    # eval_end_day = '2020-05-05'
    eval_end_day = None


# -%% (OPTIONAL) Define parameters for init conditional optimization

    # @Alex, @Josh: YOU CAN PROBABLY IGNORE THIS SECTION
    # In general, IGNORE this unles you want to repeat the optimizations that Dan is doing

    # This section controls whether the initical conditions are optimized over
        # This is a hyper parameter optimization BUT different from the real one
        # that Alex is doing. 

    #-- Flag to do hyperparameter optimization over the init cond. fudge factors
    isRunInitConHyper = False

    #-- Bounds for the search
    init_bd = [(0.3,5), (0.01,1), (0.001,20)]

    #-- Other settings for hyperparam run
    acq_func = "EI"
    n_calls = 15 
    n_random_starts = 5 
    x0 = init_vec
    noise = 0.1**2
    random_state = 1234
    HypParamVerbose = True

# -%% Prepare for calls

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


# -%% Define function that actually runs the code
# Needed as a function for the optional hyperparameter optimization 

    def runFull(init_vec):
        if isTrainModel:
            print('\n\n------ Training Model ------')
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
                            isConstInitCond = isConstInitCond,
                            init_vec = init_vec,
                            verbosity = verbosity,
                            isCluster = isCluster, cluster_max_radius = cluster_max_radius)
            if isSaveRes and not isRunInitConHyper:
                print('*** Model results saved to:\n    %s\n    %s'%(sv_flnm_mat, sv_flnm_np))
                


        if isFormat:
            print('\n------ Formatting File ------')

            if not isRunInitConHyper:
                print('*** Input filename:\n    %s'%format_flnm_in)

            format_file_for_evaluation( format_flnm_in,
                                        format_flnm_out,
                                        isAllocCounties = isAllocCounties,
                                        isComputeDaily = isComputeDaily,
                                        alloc_day=alloc_day,
                                        num_alloc_days=num_alloc_days,
                                        isAllocNN=isAllocNN,
                                        retrain=retrain,
                                        numDeaths=numDeaths,
                                        numCases=numCases,
                                        numMobility=numMobility,
                                        lenOutput=lenOutput,
                                        remove_sparse=remove_sparse,
                                        modelDir=modelDir)


            if not isRunInitConHyper:
                print('*** Formatted file:\n    %s'%format_flnm_out)
            

        if isEval:
            print('\n------ Evaluating File ------')

            if not isRunInitConHyper:
                print('*** Input filename:\n    %s'%eval_flnm_in)
                print('\n\n')
            
            score = evaluate_predictions(eval_flnm_in,
                                            eval_start_day,
                                            end_date = eval_end_day)
            return score

        

# -%% Run 

    if isRunInitConHyper:
            
        from skopt import gp_minimize
        from skopt.plots import plot_convergence

        # Overwrite parameters necessary for optimization to run
        isConstInitCond = False
        isTrainModel = True
        isSaveRes = True
        isFormat = True
        isEval = True

        # Perform hyperparam optimization over initial conditions
        res = gp_minimize(runFull, init_bd, 
                          acq_func = acq_func,
                          n_calls = n_calls,
                          n_random_starts = n_random_starts, 
                          x0 = init_vec,
                          noise = noise, 
                          random_state = random_state, 
                          verbose = HypParamVerbose)
        print()
        print(res.x)
        print()
        print(res.fun)
        print()
        print(res)
        plot_convergence(res)
    else:
        # Perform a regular run of the code
        # runFull(init_vec)
        runFull(init_vec)


                