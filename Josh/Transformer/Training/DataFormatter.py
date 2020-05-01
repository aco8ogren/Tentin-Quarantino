#%%
import pandas as pd
import os
import datetime
import json
import numpy as np
import git

# Set the current directory at wherever the repo is cloned
# os.chdir('/mnt/c/Users/lassm/NotDocuments/CS156b/Tentin-Quarantino')
repo=git.Repo('.', search_parent_directories=True)
cwd=repo.working_dir
os.chdir(cwd)

# Suppress warnings in cells to improve readability
import warnings  
warnings.filterwarnings('ignore') 


# # Data Processing

def process_covid_data(df):
    df["date"] = pd.to_datetime(df["date"])
    # Aggregate county-level deaths and cases to state level
    # df = df.drop(["county", "fips"], axis=1)
    # df = df.groupby(["date", "state"]).sum().reset_index()
    # For simplicity, instead of imputing missing mobility and deaths data, we just remove Guam
    df = df.loc[df['state'] != 'Guam']
    return df

def process_mobility_data(df):
    df["date"] = pd.to_datetime(mobility_df["date"])
    df = df[["admin1", "date",'fips', "m50", "m50_index"]]
    df = df.rename(columns={'admin1': 'state'})

    # We make this change to be consistent with other data to make merging cleanP
    df.loc[df['state'] == 'Washington, D.C.', "state"] = "District of Columbia"
    df=df.loc[~np.isnan(df['fips'])]

    lens=[]
    fips=df['fips'].unique()
    for fip in fips:
        lens.append(len(df.loc[df['fips']==fip]))
    lens=np.array(lens)
    length=lens.max()
    # fipInds=np.nonzero(lens!=length)
    RemoveFips=fips[lens!=length][np.newaxis].T
    df=df.loc[ ~((df['fips'].values==RemoveFips).any(0))]
    return df




    

def process_bed_data(df):
    # Since the hospital data uses PO codes, we merge this data with a po code/state map
        # to get state name instead
    po_state_map = pd.read_json("data/us/processing_data/po_code_state_map.json", orient='records')
    df = df.merge(po_state_map, how='inner', left_on="state", right_on="postalCode")
    df = df[["bedspermille", "state_y"]]
    df = df.rename(columns={"state_y": "state"})
    return df


# In[6]:


# Read in the data
# covid_df = pd.read_csv(f"data/us/covid/nyt_us_counties_daily.csv")
# bed_df = pd.read_csv(f"data/us/hospitals/bed_densities.csv")
# mobility_df = pd.read_csv(f"data/us/mobility/DL-us-mobility-daterow.csv")


# In[7]:


# Apply the above processing steps
covid_df = pd.read_csv(f"data/us/covid/nyt_us_counties_daily.csv")
bed_df = pd.read_csv(f"data/us/hospitals/bed_densities.csv")
mobility_df = pd.read_csv(f"data/us/mobility/DL-us-mobility-daterow.csv")
cluster_df=pd.read_csv(f"Josh/Clustering/FinalClusters.csv")
# %%
# Apply the above processing steps
covid_df = process_covid_data(covid_df)
bed_df = process_bed_data(bed_df)
mobility_df = process_mobility_data(mobility_df)


# %%
# Right join to restrict our data to dates in the mobility dataset. Another option would be
    # to use a larger date range, but that would require imputing mobility data (the repo doesn't handle nans)
    # and making a larger percentage of our death data just a sequence of 0's (for states with no death/cases)
    # data
df = covid_df.merge(mobility_df, how='right', on=['date','fips', 'state'])

# Add hospital data
df = df.merge(bed_df, how='left', on='state')

# The initial right join will add nans for states without case/death data in the date range
    # of the mobility dataset, so we replace with 0's
df = df.fillna(value=0)
df=pd.merge(df,cluster_df,how='left',on='fips')



MeanCols=['deaths','cases','m50','m50_index']
MeanColNames=['mean_'+col for col in MeanCols]
MeanDf=df[['date','cluster']+MeanCols].groupby(['date','cluster']).mean().reset_index().rename(columns=dict(zip(MeanCols,MeanColNames)))

TotCols=['deaths','cases','m50']
TotColNames=['total_'+col for col in TotCols]
TotDf=df[['date','cluster']+TotCols].groupby(['date','cluster']).sum().reset_index().rename(columns=dict(zip(TotCols,TotColNames)))

df=pd.merge(df,MeanDf,how='right',on=['date','cluster'])
df=pd.merge(df,TotDf,how='right',on=['date','cluster'])


# Add an id column so our formatter class (below) can include state as both an identifier and categorical data
df['id'] = df['fips']
df['fipsref'] = df['fips']
df['day_of_week'] = df['date'].dt.dayofweek
df = df.sort_values(by='date')


# # Code to tell this repo how to use this data
# 
# Examples can be found in the repo in the data_formatters directory

# In[7]:
import sys
sys.path.append('Josh/Transformer/tft')

from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
from data_formatters.traffic import TrafficFormatter

# View available inputs and data types.
print("Available data types:")
for option in DataTypes:
    print(option)

print()
print("Available input types:")
for option in InputTypes:
    print(option)


# In[8]:


from libs import utils
import sklearn.preprocessing


# This class must inherit from GenericDataFormatter and implement the methods given below
# or NotImplemented errors will be raised
class covidFormatter(GenericDataFormatter):
    """Defines and formats data for the covid dataset"""
    # _column_definition = [
    #     ('id', DataTypes.CATEGORICAL, InputTypes.ID),
    #     ('date', DataTypes.DATE, InputTypes.TIME),
    #     ('deaths', DataTypes.REAL_VALUED, InputTypes.TARGET),
    #     ('cases', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #     ('m50', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #     ('m50_index', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #     ('state', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    #     ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
    #     ('bedspermille', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
    # ]


    _column_definition = [
        # ('fips', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('date', DataTypes.DATE, InputTypes.TIME),
        ('cases', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('deaths', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('m50', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('m50_index', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('bedspermille', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('long', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('lat', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('mean_deaths', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('mean_cases', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('mean_m50', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('mean_m50_index', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('total_deaths', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('total_cases', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('total_m50', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
    ]



    
    def split_data(self, df):
        """Split data frame into training-validation data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_boundary: Starting date for validation data

        Returns:
          Tuple of transformed (train, valid) data.
        """
        print('Formatting train-valid splits.')
        
        # This function is meant to provide functionality for splitting
        # the data into train/valid/test along date boundaries. To keep consistent date ranges
        # for each identifier however, splitting the data would require
        # designating contiguous chunks of time as train/valid/test.
        # However, with such a small date range as is (not to mention
        # such a split would ensure that test/valid are not at all representative of
        # train since the date ranges would be different), splitting 
        # by date is just not feasible
        
        # Instead, we just do nothing and make no split. Since the model fit
        # function requires validation data, we just duplicate the train data.
        # This is clearly not optimal, and a clear way to improve on this simple example
        
        # The best way to split data would likely be along state levels,
        # but unfortunately this repo is very finicky with categorical data
        # and would not be happy with train and valid having different state
        # categories, so some workarounds would have to be made
        
        self.set_scalers(df)
        return (self.transform_inputs(data) for data in [df.copy(), df.copy()])


    

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
          df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')
        # Code from their examples
        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        # For real-valued inputs (including our target), 
        # fit a transformation to scale to unit variance and zero mean
        # This is just the fitting step, the actual transformation can
        # be (or not be) applied in the next function
        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  

        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        # Fit an encoder to one-hot encode categorical inputs
        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].astype(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
              srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes
        
    def transform_inputs(self, df):
        """Performs feature transformations.

        Args:
          df: Data frame to transform.

        Returns:
          Transformed data frame.

        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Use the previously fit StandardScaler() to transform the data if desired
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)
        output[real_inputs] = output[real_inputs]

        # Use the previously fit LabelEncoder()
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output
    

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
          predictions: Dataframe of model predictions.

        Returns:
          Data frame of unnormalised predictions.
        """
        output = predictions.copy()

        column_names = predictions.columns
        # Use the inverse transform of our scaler to get back original scale
        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output
    
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps':21,     # Total width of the Temporal Fusion Decoder
            'num_encoder_steps': 14,    # Length of LSTM decoder (ie. # historical inputs)
            'num_epochs': 1,            # Max number of epochs for training 
            'early_stopping_patience': 5, # Early stopping threshold for # iterations with no loss improvement
            'multiprocessing_workers': 5  # Number of multi-processing workers
        }

        return fixed_params
    


# In[9]:


# Instatiate our custom class and prepare training data
data_formatter = covidFormatter()
train, valid  = data_formatter.split_data(df)


# In[10]:


data_params = data_formatter.get_experiment_params()
# Model parameters for calibration

# Another parameter you could set here is "quantiles",
# right now it just predicts the default quantiles
model_params = {'dropout_rate': 0.1,      # Dropout discard rate
                'hidden_layer_size': 50, # Internal state size of TFT
                'learning_rate': 0.01,   # ADAM initial learning rate
                'minibatch_size': 64,    # Minibatch size for training
                'max_gradient_norm': 100.,# Max norm for gradient clipping
                'num_heads': 2,           # Number of heads for multi-head attention
                'stack_size': 1,           # Number of stacks (default 1 for interpretability)
                # 'quantiles': .05,           # Quantiles to predict
               }

# Folder to save network weights during training.
model_folder = os.path.join("Josh", 'Transformer', 'Checkpoints','County Checkpoints 2')
model_params['model_folder'] = model_folder

model_params.update(data_params)
