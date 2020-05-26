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
    train_til = '2020-05-10'
    test_from = '2020-05-11'
    test_til  = None
    temp_processed_csv = r'Josh\Alloc_NN\NN_opt\temp_NN_opt.csv'
    numDeaths, numCases, numMobility, lenOutput, Patience, DropoutRate = params
    format_file_for_evaluation( input_fln= r'Josh\PracticeOutputs\NNAllocation\Checkoutfde400_TrainTil10_R0p2.mat',
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
                                remove_sparse=True,
                                Patience=Patience,
                                DropoutRate=DropoutRate,
                                modelDir=r'Josh\Alloc_NN\NN_opt\RunModel')
    
    # score = score_all_predictions('temp_processed.csv', date, model_date, mse=False, key='cases', bin_cutoffs=[20, 1000])
    score = evaluate_predictions(temp_processed_csv,test_from,end_date = test_til)
    print('numDeaths={}, numCases={}, numMobility={}, lenOutput={}, Patience={}, DropoutRate={}\nLoss: {}'.format(*params,score))
    return score

def Opti(fun, bounds, checkpointFln,Init='random', types=None,paramNames=None,floatRate=.05):
    if types == None:
        types=[type(b[0]) for b in bounds]
        if Init != 'random':
            types2 = [type(i) for i in Init]
            if types != types2:
                raise ValueError('Data types on bounds and Init do not match')
    if Init =='random':
        Init=[]
        for i in range(len(bounds)):
            T=types[i]
            if T is int:
                Init.append(np.random.randint(bounds[i][0],bounds[i][1]+1))
            elif T is float:
                Init.append(np.random.uniform(bounds[i][0],bounds[i][1]))
            # elif T is bool:
            #     Init.append(np.random.choice([True,False]))
            else:
                raise ValueError('Datatype %s not compatible'%T)
    if paramNames is None:
        paramNames=range(len(bounds))
    currParams=Init
    currLoss=fun(currParams)
    row=currParams+[currLoss]
    DF=pd.DataFrame([currParams],columns=paramNames)
    DF['loss']=currLoss
    DF.to_csv(checkpointFln)
    static=[]
    ongoing=list(range(len(bounds)))


    while len(ongoing)!=0:
        curr=np.random.choice(ongoing)
        if types[curr]==int:
            if currParams[curr]>bounds[curr][0]:
                lowParams=currParams.copy()
                lowParams[curr]-=1
                lowLoss=fun(lowParams)
            else:
                lowParams=currParams.copy()
                lowLoss=1e6
            if currParams[curr]<bounds[curr][1]:
                highParams=currParams.copy()
                highParams[curr]+=1
                highLoss=fun(highParams)
            else:
                highParams=currParams.copy()
                highLoss=1e6
        if types[curr]==float:
            if currParams[curr]-floatRate>=bounds[curr][0]:
                lowParams=currParams.copy()
                lowParams[curr]-=floatRate
                lowLoss=fun(lowParams)
            else:
                lowParams=currParams.copy()
                lowLoss=1e6
            if currParams[curr]+floatRate<=bounds[curr][1]:
                highParams=currParams.copy()
                highParams[curr]+=floatRate
                highLoss=fun(highParams)
            else:
                highParams=currParams.copy()
                highLoss=1e6
        losses=[lowLoss,currLoss,highLoss]
        params=[lowParams,currParams,highParams]
        best=np.argmin(losses)
        if min(losses)==currLoss:
            static.append(curr)
            ongoing.remove(curr)
        else:
            static=[]
            ongoing=list(range(len(bounds)))
            currParams=params[best]
            currLoss=losses[best]
            DF.loc[len(DF)]=currParams+[currLoss]
            print('\nLoss went down:')
            print(''.join(['{}: {}, '.format(p,v) for p,v in zip(paramNames,currParams)]))
            print('Loss: %f'%currLoss)
            DF.to_csv(checkpointFln)
    print('Optimization Complete: Results saved to %s'%checkpointFln)
    print(''.join(['{}: {}, '.format(p,v) for p,v in zip(paramNames,currParams)]))
    print('Loss: %f'%currLoss)
    return 

        
bounds=[[0,10],     # numDeaths
        [0,10],     # numCases
        [0,10],     # numMobility
        [1,10],     # lenOutput
        [1,20],     # patience
        [0.,.5]]    # dropout
Init=[2,2,2,5,4,.15]
paramNames='numDeaths numCases numMobility lenOutput patience dropout'.split(' ')


# Opti(test_error,bounds,r'Josh\Alloc_NN\opt1.csv',Init=Init,paramNames=paramNames)
# Opti(test_error,bounds,r'Josh\Alloc_NN\opt2.csv',paramNames=paramNames)
# Opti(test_error,bounds,r'Josh\Alloc_NN\opt3.csv',paramNames=paramNames)
Opti(test_error,bounds,r'Josh\Alloc_NN\opt4.csv',paramNames=paramNames)
Opti(test_error,bounds,r'Josh\Alloc_NN\opt5.csv',paramNames=paramNames)



# ParamsList=[[2,2,2,6,True,5,.1],
#             [2,1,2,5,True,4,.1],
#             [2,1,2,6,True,5,.1],
#             [2,2,2,5,True,6,.1],
#             [2,2,2,7,False,4,.1]]

# # ParamsList=[[2,2,2,5,True,4,.1],
# #             [5,5,5,5,True,4,.1],
# #             [2,2,2,2,True,4,.1],
# #             [10,10,10,10,True,10,.1],
# #             [10,0,0,10,True,10,.1],
# #             [2,2,2,2,True,2,.3],
# #             [5,5,5,5,False,4,.1]]




# results=[]
# for i,params in enumerate(ParamsList):
#     print('\nRun {}/{}:'.format(i+1,len(ParamsList)))
#     results.append(test_error(params))

# Results=pd.DataFrame(ParamsList,columns='numDeaths, numCases, numMobility, lenOutput, remove_sparse, Patience, DropoutRate'.split(', '))
# Results['loss']=results
# Results.to_csv(r'Josh\Alloc_NN\runs3.csv')