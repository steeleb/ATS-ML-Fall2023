# activate conda environment in terminal: 'conda activate ~/miniconda3/envs/env_ATSML/'

# import modules
import os
import sys
import pandas as pd
import numpy as np
from datetime import date
import pickle

file_path = os.path.expanduser("~/OneDrive - Colostate/NASA-Northern/data/NN_train_val_test/")

# import training data
fn = os.path.join(file_path, "training_set_up_to_2021_v2023-11-08.csv")
with open(fn) as f:
    df = pd.read_csv(f, sep=',')

print(df.columns)

# we need to preprocess these data according to their distribuions and ranges. 
### Precip data needs log transformation before standardization due to histo frequency.
### Wind data needs sqrt transormation.
precip_vars = df.filter(like='precip').columns
wind_vars = df.filter(like="wind").columns

# first, we'll trasform the precip and wind vars
precip = df.filter(like='precip')
### add 0.0001 to values in all columns of precip, since we need to log-transform the values and there are zeros
precip += 0.0001
precip = np.log(precip)
print(precip.describe())
### and sqrt transform the wind vars
wind = df.filter(like='wind')
wind = np.sqrt(wind)
print(wind.describe())

# remove the precip and wind vars from df
df = df.drop(columns=precip_vars)
df = df.drop(columns=wind_vars)
# add the transformed precip and wind vars to df
df = df.join(precip)
df = df.join(wind)
print(df.head)
print(df.describe())

# now, we'll standardize the data so that all the values are between -1 and 1.
def standardize_column(df, col_name):
    col = df[col_name]
    return (col - col.mean()) / col.std()

# apply standardize_column function to all columns of df
# but first, drop the feature column
df_short = df.copy()
df_short = df_short.drop(columns=['feature', 'date'])
# and then apply the standardization function
df_standardized = df_short.apply(lambda col: standardize_column(df_short, col.name))
print(df_standardized.head)
print(df_standardized.describe())

# because we want to be able to apply these same standardizations to the test data, we should grab the mean/std values for each column
df_mean_std = pd.DataFrame({'mean': df_short.mean(), 'std': df_short.std()})
print(df_mean_std)

# lets save this as a .csv file for later use
# save df_mean_std to csv
# first, create file name
file_name = "mean_std_training_v" + str(date.today()) + ".csv"
# join with file path
fp = os.path.join(file_path, file_name)
# and save
df_mean_std.to_csv(fp, index=True)

# our training data are now ready to be split into training and validation sets.
training = df_standardized.copy()

# but let's add lake and date back in.
training = training.join(df[['feature', 'date']])

# now, we'll split the data into training and validation sets.
### I'm going to do this two ways:
### 1) by timeseries
### 2) by LOO by lake

# 1) by timeseries

# filter training dataframe by date range
start_date_1 = '1984-01-01'
end_date_1 = '1995-01-01'
val1_ts = training.loc[training['date'].between(start_date_1, end_date_1)]
train1_ts = training.merge(val1_ts, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)

start_date_2 = end_date_1
end_date_2 = '2005-01-01'
val2_ts = training.loc[training['date'].between(start_date_2, end_date_2)]
train2_ts = training.merge(val2_ts, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)

start_date_3 = end_date_2
end_date_3 = '2015-01-01'
val3_ts = training.loc[training['date'].between(start_date_3, end_date_3)]
train3_ts = training.merge(val3_ts, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)

start_date_4 = end_date_3
end_date_4 = '2021-01-01'
val4_ts = training.loc[training['date'].between(start_date_4, end_date_4)]
train4_ts = training.merge(val4_ts, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)

# 2) by LOO by lake
# using a leave-one-out method for train/validate
train1 = training.loc[training['feature'] != 'Grand Lake']
val1 = training.loc[training['feature'] == 'Grand Lake']

train2 = training.loc[training['feature'] != 'Horsetooth Reservoir']
val2 = training.loc[training['feature'] == 'Horsetooth Reservoir']

train3 = training.loc[training['feature'] != 'Shadow Mountain Reservoir']
val3 = training.loc[training['feature'] == 'Shadow Mountain Reservoir']

train4 = training.loc[training['feature'] != 'Granby Reservoir']
val4 = training.loc[training['feature'] == 'Granby Reservoir']

train5 = training.loc[training['feature'] != 'Carter Lake']
val5 = training.loc[training['feature'] == 'Carter Lake']

train6 = training.loc[training['feature'] != 'Willow Creek Reservoir']
val6 = training.loc[training['feature'] == 'Willow Creek Reservoir']

# and now save these all as pickle files
def save_df_as_pickle(df, file_name, file_path=''):
    """
    Save a pandas dataframe as a pickle file.
    
    Parameters:
    df (pandas.DataFrame): The dataframe to be saved.
    file_name (str): The name of the file to be saved.
    file_path (str): The path to the folder where the file will be saved.
    
    Returns:
    None
    """
    with open(os.path.join(file_path, file_name), 'wb') as f:
        pickle.dump(df, f)

# list of dataframes to save as pickle files with file names
list_ts_dfs = [[train1_ts, "train1_ts"],
               [val1_ts, "val1_ts"],
               [train2_ts, "train2_ts"],
               [val2_ts, "val2_ts"],
               [train3_ts, "train3_ts"],
               [val3_ts, "val3_ts"],
               [train4_ts, "train4_ts"],
               [val4_ts, "val4_ts"]]

list_dfs = [[train1, "train1"],
            [val1, "val1"],
            [train2, "train2"],
            [val2, "val2"],
            [train3, "train3"],
            [val3, "val3"],
            [train4, "train4"],
            [val4, "val4"],
            [train5, "train5"],
            [val5, "val5"],
            [train6, "train6"],
            [val6, "val6"]]

# map the function save_df_as_pickle across the list of dataframes list_ts_dfs and list_dfs
for df, name in list_ts_dfs:
    file_name = f"{name}.pkl"
    save_df_as_pickle(df, file_name, "/Users/steeleb/OneDrive - Colostate/NASA-Northern/data/NN_train_val_test/pickles/")

for df, name in list_dfs:
    file_name = f"{name}.pkl"
    save_df_as_pickle(df, file_name, "/Users/steeleb/OneDrive - Colostate/NASA-Northern/data/NN_train_val_test/pickles/")
