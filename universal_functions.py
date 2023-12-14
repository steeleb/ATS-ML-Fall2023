import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def save_to_pickle(obj, filepath):
    """
    Save an object to a pickle file.
    
    Parameters:
    obj (object): The object to be saved.
    filepath (str): The filepath to save the object to.
    
    Returns:
    None
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle_file(file_name, file_path):
    """
    Load a pickle file from a given file path and file name.

    Args:
    file_path (str): The path to the directory containing the pickle file.
    file_name (str): The name of the pickle file.

    Returns:
    any: The object stored in the pickle file.
    """
    with open(file_path + '/' + file_name, 'rb') as f:
        return pickle.load(f)


def get_features_labels(train_dfs, val_dfs):
  # grab the values we want to predict
  labels = np.array(train_dfs['value'])
  val_labels = np.array(val_dfs['value'])
  
  # and remove the labels from the dataset containing the feature set
  features = train_dfs.drop(['value', 'feature', 'date'], axis=1)
  val_features = val_dfs.drop(['value', 'feature', 'date'], axis=1)
  
  return features, labels, val_features, val_labels


def get_features_labels_test(test_df):
  # grab the values we want to predict
  labels = np.array(test_df['value'])
  
  # and remove the labels from the dataset containing the feature set
  features = test_df.drop(['value', 'feature', 'date'], axis=1)
  
  return features, labels


def calculate_vals(transformed_val, mean, std):
  actual_val = (transformed_val * std) + mean
  return actual_val

def predict_values(model, features, val_features, labels, val_labels, t_mean, t_std):
    pred = model.predict(features)
    val = model.predict(val_features)
    p_act = calculate_vals(pred, t_mean, t_std)
    l_act = calculate_vals(labels, t_mean, t_std)
    p_v_act = calculate_vals(val, t_mean, t_std)
    l_v_act = calculate_vals(val_labels, t_mean, t_std)
    return p_act, l_act, p_v_act, l_v_act

def predict_values_test(model, features, t_mean, t_std):
    pred = model.predict(features)
    p_act = calculate_vals(pred, t_mean, t_std)
    return p_act

def print_error_metrics(dataset_num, l_act, p_act, l_v_act, p_v_act):
    t_mse = mean_squared_error(l_act, p_act)
    t_mae = mean_absolute_error(l_act, p_act)
    v_mse = mean_squared_error(l_v_act, p_v_act)
    v_mae = mean_absolute_error(l_v_act, p_v_act)
    print("DATASET", dataset_num)
    print("Mean Squared Error for Training Dataset", dataset_num, ":", t_mse)
    print("Mean Absolute Error for Training Dataset", dataset_num, ":", t_mae)
    print("Mean Squared Error for Validation Dataset", dataset_num, ":", v_mse)
    print("Mean Absolute Error for Validation Dataset", dataset_num, ":", v_mae)
    print(' ')

def return_test_error_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    return mse, mae, rmse, mape