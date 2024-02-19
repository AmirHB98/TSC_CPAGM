import os
from scipy.io import arff
import numpy as np

def summon_all_series():
    """Chinatown dataset is made of 363 series, 20 in TRAIN set and 343 in TEST set, this function combines all.

    Returns
    -------
    series_set: list
        List of numpy arrays whcich are a series each
    series_label : list
        List of integers indicating calss of the series (1-Weekend 2-Weekday)
    """

    path = os.getcwd() +'\CPAGM\data\Chinatown_TEST.arff'
    series_set, series_label = get_data(path) # retireve data from dataset
    path = os.getcwd() +'\CPAGM\data\Chinatown_TRAIN.arff'
    series_set_2, serises_label_2 = get_data(path)
    series_set.extend(series_set_2)
    series_label.extend(serises_label_2)

    return series_set, series_label

def get_data(path: str):
    """Given the path to .arff file, return time series data as np.float64 type and their relevant label as np.int332 type 

    Parameters
    ----------
    path : str
        Relative or strict path to .arff files from Chinatown dataset

    Returns
    -------
    series_set:list
        List that contains time series data per day
    
    series_label:list
        List that contains each days class: 1- Weekend 2- Weekday 
        
    """
    load_data = arff.loadarff(path)
    dataset = load_data[0]
    # print(f'type of train set: {type(train_set)}, length of train set: {train_set.shape}' )
 
    field_names = list(dataset[0].dtype.names)
    float_fields = field_names[:-1]

    series_set = []
    series_label =[]

    for item in dataset:
        x =[]
        for field in float_fields:
            x.append(item[field].astype(np.float64))
    
        series_set.append(np.array(x,dtype = np.float64))
        series_label.append(item[field_names[-1]].astype(np.int32))
    
    # print(f'type of one vector: {type(item)}, last element type: {type(item[-1])}, first element type:{type(item[0])}')
    return series_set, series_label

def Xy_Split(series_set: list, lags : int):
    """Given a series of type list, Creates suitable input and output arrays to use in regression models

    Parameters
    ----------
    series_set : list
        List of several time series
    lags : int
        Lag features

    Returns
    -------
    X: numpy array
        Input Matrix for global regression model
    y: numpy array
        Desired outputs from global regression model
    """
    X = []
    y =[]
    for series in series_set:
        for i in range(lags,series.size):
            X.append(series[i-lags:i])
            y.append(series[i])

    X = np.array(X)
    y = np.array(y)

    return X,y
 