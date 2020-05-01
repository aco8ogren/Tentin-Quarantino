#%%
import pandas as pd
import os
import datetime
import json
import numpy as np
import git
import itertools
import tensorflow as tf
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)
import sys 
sys.path.append('Josh/Transformer/tft')
sys.path.append('Josh/Transformer/Training')
from libs.tft_model import TemporalFusionTransformer
from libs import utils
from DataFormatter import model_params, train, valid, model_folder, df, data_formatter, InputTypes

PredLength=2
#%%
Pred=df[df.fips==36061]
# Pred=df.copy()
Date=[Pred.date.max()+pd.DateOffset(days=i) for i in np.arange(PredLength)+1.]
# DateDF=pd.DataFrame({'date':[Pred.date.max()+pd.DateOffset(days=i) for i in np.arange(PredLength)+1.]})
# fips=df.fips.unique()[:2]
# fips=[36061,36059]
fips=[36061]
DateDF=pd.DataFrame(itertools.product(Date,fips),columns=['date','fips'])


# FipsDF=pd.DataFrame({'fips':fips})
maxDate=Pred.date.max()+pd.DateOffset(days=PredLength)


ColDefs=data_formatter.get_column_definition()
KnownInputs=[]
for col in ColDefs:
    if col[2] in [InputTypes.KNOWN_INPUT,InputTypes.STATIC_INPUT]:
        KnownInputs.append(col[0])
KnownIns_NoDay=KnownInputs.copy()
KnownIns_NoDay.remove('day_of_week')
KnownDF=df[['fips']+KnownIns_NoDay]
KnownDF=KnownDF.drop_duplicates()
KnownDF=pd.merge(DateDF,KnownDF,how='left',on='fips')
KnownDF['day_of_week']=KnownDF['date'].dt.dayofweek
KnownDF['id']=KnownDF['fips'].copy()
KnownDF['deaths']=[1000]*len(KnownDF)
# Pred=pd.concat([Pred,KnownDF]).fillna(1000)
Pred=pd.concat([Pred,KnownDF])
PredVals, temp=data_formatter.split_data(Pred)


#%%


tf.reset_default_graph()
with tf.Graph().as_default(), tf.Session() as sess:

    tf.keras.backend.set_session(sess)
    
    # Create a model with same parameters as we trained with & load weights
    model = TemporalFusionTransformer(model_params);
    model.load(model_folder);
    
    # Make forecasts
    # output_map = model.predict(Pred, return_targets=True)
    output_map = model.predict(PredVals, return_targets=True)

    targets = data_formatter.format_predictions(output_map["targets"])








    # # Format predictions
    # p50_forecast = data_formatter.format_predictions(output_map["p50"])
    # p90_forecast = data_formatter.format_predictions(output_map["p90"])

    # def extract_numerical_data(data):
    #     """Strips out forecast time and identifier columns."""
    #     return data[[
    #       col for col in data.columns
    #       if col not in {"forecast_time", "identifier"}
    #     ]]

    # # Compute quantile losses using their functionality, but could easily be changed to pinball
    # p5
    # 0_loss = utils.numpy_normalised_quantile_loss(
    #     extract_numerical_data(targets), extract_numerical_data(p50_forecast),
    #     0.5)
    # p90_loss = utils.numpy_normalised_quantile_loss(
    #     extract_numerical_data(targets), extract_numerical_data(p90_forecast),
    #     0.9)

# %%
