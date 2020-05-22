# %%
import os
import sys

HomeDIR='Tentin-Quarantino'
wd=os.path.dirname(os.path.realpath(__file__))
DIR=wd[:wd.find(HomeDIR)+len(HomeDIR)]
os.chdir(DIR)
sys.path.append(os.getcwd())

from Alex import copy_of_erf_model
from benchmark_models import utils
import time
import os
import sys
import numpy as np

list_of_fips_to_erf = [36061] #[8073,8053,8063,8101,8111,8117]

erf_df = utils.get_processed_df()
tic_erf = time.time()
cube = copy_of_erf_model.predict_counties(erf_df, list_of_fips_to_erf,
                                                            last_date_pred='2020-06-30', 
                                                            out_file='erf_model_predictions.csv', 
                                                            boundary_date=None,
                                                            key='deaths',
                                                            verbose = True)
toc_erf = time.time()
print('Fitting erf done. Time elapsed', np.round(toc_erf-tic_erf,2),'sec')