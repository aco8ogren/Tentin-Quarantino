import pandas as pd


import os
import sys
HomeDIR='Tentin-Quarantino'
wd=os.path.dirname(os.path.realpath(__file__))

DIR=wd[:wd.find(HomeDIR)+len(HomeDIR)]
os.chdir(DIR)
sys.path.append(os.getcwd())

homedir = DIR
datadir = f"{homedir}"

import numpy as np
from Dan.format_sub import format_file_for_evaluation
from Alex.copy_of_evaluator import evaluate_predictions



def test_error(params):
    train_til = '2020 05 10'
    test_from = '2020 05 11'
    test_til  = None
    temp_processed_csv = r'Josh\Alloc_NN\NN_opt\temp_NN_opt.csv'
    numDeaths, numCases, numMobility, lenOutput, remove_sparse, Patience, DropoutRate = params
    format_file_for_evaluation( input_fln='Josh/PracticeOutputs/Baseline_with_clusters.npy',
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
                                modelDir=r'Josh\Alloc_NN\NN_opt\RunModel')
    
    # score = score_all_predictions('temp_processed.csv', date, model_date, mse=False, key='cases', bin_cutoffs=[20, 1000])
    score = evaluate_predictions(temp_processed_csv,test_from,end_date = test_til)
    print('numDeaths={}, numCases={}, numMobility={}, lenOutput={}, remove_sparse={}, Patience={}, DropoutRate={}\nLoss: {}'.format(*params,score))
    return score
    
ParamsList=[[2,2,2,5,True,4,.1],
            [5,5,5,5,True,4,.1],
            [2,2,2,2,True,4,.1],
            [10,10,10,10,True,10,.1],
            [10,0,0,10,True,10,.1],
            [2,2,2,2,True,2,.3],
            [5,5,5,5,False,4,.1]]

results=[]
for i,params in enumerate(ParamsList):
    print('\nRun {}/{}'.format(i,len(ParamsList))
    results.append(test_error(params))

Results=pd.DataFrame(ParamsList,columns='numDeaths, numCases, numMobility, lenOutput, remove_sparse, Patience, DropoutRate'.split(', '))
Results['loss']=results
Results.to_csv(r'Josh\Alloc_NN\runs.csv')