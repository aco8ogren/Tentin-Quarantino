#%%
import pandas as pd
import os
import datetime
import json
import numpy as np
import git
import tensorflow as tf
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)
import sys 
sys.path.append('Josh/Transformer/tft')
sys.path.append('Josh/Transformer/Training')
from libs.tft_model import TemporalFusionTransformer
from libs import utils

from DataFormatter import model_params, train, valid, model_folder, df, data_formatter

tf.reset_default_graph()
with tf.Graph().as_default(), tf.Session() as sess:

    tf.keras.backend.set_session(sess)
    
    # Create a TFT model with our parameters
    model = TemporalFusionTransformer(model_params)
                                    

    # We don't have much data so this caching functionality is really not necessary,
    # but why not. We could also just directly pass train and validation data to the model.fit() method
    if not model.training_data_cached():
        model.cache_batched_data(train, "train")
        model.cache_batched_data(valid, "valid")


    # Train and save model
    model.fit()
    model.save(model_folder)