# activate conda environment in terminal: 'conda activate ~/miniconda3/envs/env_ATSML/'

# import modules
import os
import pandas as pd
import numpy as np
from datetime import date

file_path = os.path.expanduser("~/OneDrive - Colostate/NASA-Northern/data/NN_train_val_test/regional_daily_temp/")

# import training data
fn = os.path.join(file_path, "application/application_dataset_2023.csv")
with open(fn) as f:
    df = pd.read_csv(f, sep=',')

fn = os.path.join(file_path, "mean_std_training_v2024-10-18.csv")
with open(fn) as f:
    mean_std = pd.read_csv(f, sep=',')

# set indes to first column
mean_std = mean_std.set_index('Unnamed: 0')

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

### and sqrt transform the wind vars
wind = df.filter(like='wind')
wind = np.sqrt(wind)

# remove the precip and wind vars from df
df = df.drop(columns=precip_vars)
df = df.drop(columns=wind_vars)
# add the transformed precip and wind vars to df
df = df.join(precip)
df = df.join(wind)

# now, we'll standardize the data so that all the values are between -1 and 1.
def standardize_column(df, col_name, mean, std):
    col = df[col_name]
    return (col - mean) / std

# drop lake column
df_short = df.copy()
df_short = df_short.drop(columns = ["feature", "date"])

# apply standardize_column function to all columns of df
df_standardized = df_short.apply(lambda col: standardize_column(df_short, col.name, mean_std.loc[col.name, 'mean'], mean_std.loc[col.name, 'std']))

# save the file
fn = os.path.join(file_path, "application/application_dataset_2023_standardized.csv")
df_standardized.to_csv(fn, index=False)
