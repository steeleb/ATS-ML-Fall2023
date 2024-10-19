#high level modules
import os
import sys
import imp
import numpy as np
import pandas as pd
import pickle

# ml/ai modules
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# import pydot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors

# custom modules
this_dir = "/Users/steeleb/Documents/GitHub/ATS-ML-Fall2023/"

imp.load_source("universals", os.path.join(this_dir, "universal_functions.py"))
from universals import load_pickle_file, get_features_labels_test, predict_values_test, print_error_metrics

# import test data
file_path = os.path.expanduser("~/OneDrive - Colostate/NASA-Northern/data/NN_train_val_test/regional_daily_temp/")

test_fn = os.path.join(file_path, "test_2022_v2024-10-18.csv")
with open(test_fn) as f:
    test = pd.read_csv(f, sep=',')

trans_fn = os.path.join(file_path, "mean_std_training_v2024-10-18.csv")
with open(trans_fn) as f:
    transform = pd.read_csv(f, sep=',')
# now create and index from the first column
transform = transform.set_index(transform.columns[0])

# we need to preprocess these data like we did with the training and validation sets.
# first, we'll trasform the precip and wind vars
precip_vars = test.filter(like='precip').columns
wind_vars = test.filter(like="wind").columns
# transform precip
precip = test.filter(like='precip')
### add 0.0001 to values in all columns of precip, since we need to log-transform the values and there are zeros
precip += 0.0001
precip = np.log(precip)
### and sqrt transform the wind vars
wind = test.filter(like='wind')
wind = np.sqrt(wind)

# and now we'll join those back to the test df
test = test.drop(columns=precip_vars)
test = test.drop(columns=wind_vars)
test = test.join(precip)
test = test.join(wind)

# now, we'll standardize the data so that all the values are between -1 and 1. We're going to do this with the mean and std values from the training set.
test_trans = test.copy()

# and then apply the standardization function
def apply_transform(df, col_name, transform):
    col = df[col_name]
    return (col - transform.loc[col_name]['mean']) / transform.loc[col_name]['std']

for col in test_trans.columns:
    if col != 'value' and col != 'feature' and col != 'date':
        test_trans[col] = apply_transform(test_trans, col, transform)

# now we need to apply the models to the test data
# first, format the data for models.
test_features, test_labels = get_features_labels_test(test_trans)

# load the models
model_dir = '/Users/steeleb/OneDrive - Colostate/NASA-Northern/data/NN_train_val_test/regional_daily_temp/models/basic_leaky_updated/'

models = [f for f in os.listdir(model_dir) if 'history' not in f]

ts_models = [f for f in models if 'ts' in f]
ts_models.sort()

models = [f for f in models if 'ts' not in f]
models.sort()

ts_model_1 = load_pickle_file(ts_models[0], model_dir)
ts_model_2 = load_pickle_file(ts_models[1], model_dir)
ts_model_3 = load_pickle_file(ts_models[2], model_dir)
ts_model_4 = load_pickle_file(ts_models[3], model_dir)

model_1 = load_pickle_file(models[0], model_dir)
model_2 = load_pickle_file(models[1], model_dir)
model_3 = load_pickle_file(models[2], model_dir)
model_4 = load_pickle_file(models[3], model_dir)
model_5 = load_pickle_file(models[4], model_dir)
model_6 = load_pickle_file(models[5], model_dir)
model_7 = load_pickle_file(models[6], model_dir)
model_8 = load_pickle_file(models[7], model_dir)
model_9 = load_pickle_file(models[8], model_dir)

# and now we can predict values and back transform them
t_mean = transform.loc['value']['mean']
t_std = transform.loc['value']['std']

test["p_act_1"] = predict_values_test(model_1, test_features, t_mean, t_std)
test["p_act_2"] = predict_values_test(model_2, test_features, t_mean, t_std)
test["p_act_3"] = predict_values_test(model_3, test_features, t_mean, t_std)
test["p_act_4"] = predict_values_test(model_4, test_features, t_mean, t_std)
test["p_act_5"] = predict_values_test(model_5, test_features, t_mean, t_std)
test["p_act_6"] = predict_values_test(model_6, test_features, t_mean, t_std)
test["p_act_7"] = predict_values_test(model_7, test_features, t_mean, t_std)
test["p_act_8"] = predict_values_test(model_8, test_features, t_mean, t_std)
test["p_act_9"] = predict_values_test(model_9, test_features, t_mean, t_std)

test["p_act_1_ts"] = predict_values_test(ts_model_1, test_features, t_mean, t_std)
test["p_act_2_ts"] = predict_values_test(ts_model_2, test_features, t_mean, t_std)
test["p_act_3_ts"] = predict_values_test(ts_model_3, test_features, t_mean, t_std)
test["p_act_4_ts"] = predict_values_test(ts_model_4, test_features, t_mean, t_std)

selected_cols = ['feature', 'date', 'value', 'p_act_1', 'p_act_2', 'p_act_3', 'p_act_4', 'p_act_5', 'p_act_6', 'p_act_7', 'p_act_8', 'p_act_9', 'p_act_1_ts', 'p_act_2_ts', 'p_act_3_ts', 'p_act_4_ts']
test_selected = test[selected_cols]

test_selected["LOO_ensemble_pred"] = np.mean(test_selected[['p_act_1', 'p_act_2', 'p_act_3', 'p_act_4', 'p_act_5', 'p_act_6', 'p_act_7', 'p_act_8', 'p_act_9']], axis=1)
test_selected["ts_ensemble_pred"] = np.mean(test_selected[['p_act_1_ts', 'p_act_2_ts', 'p_act_3_ts', 'p_act_4_ts']], axis=1)
test_selected["ensemble_grand_mean_pred"] = np.mean(test_selected[['p_act_1', 'p_act_2', 'p_act_3', 'p_act_4', 'p_act_5', 'p_act_6', 'p_act_7', 'p_act_8', 'p_act_9', 'p_act_1_ts', 'p_act_2_ts', 'p_act_3_ts', 'p_act_4_ts']], axis=1)
ensemble_cols = ['feature', 'date', 'value', 'LOO_ensemble_pred', 'ts_ensemble_pred', "ensemble_grand_mean_pred"]
test_ensemble = test_selected[ensemble_cols]

test_ensemble_cols = ['feature', 'date', 'value', 'p_act_1_ts', 'p_act_2_ts', 'p_act_3_ts', 'p_act_4_ts']
test_ensemble_ts = test_selected[test_ensemble_cols]
