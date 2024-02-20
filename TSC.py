from sklearn.metrics import adjusted_rand_score
from utils.process_data import summon_all_series
from models.global_model import global_model
from models.local_model import local_model
from models.model_1 import main_algorithm
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import random

SPLIT_POINT = 19
CONVERGE_LIMIT = 0.8
NUM_CLUSTERS = 2
NUM_PLOTS = 2

#TODO: yaml or json

#TODO: Algorithm2 
    
#TODO: README with some plots

if __name__ == '__main__':
    
    # Read train and test data from dataset 
    series_set, series_label = summon_all_series()

    # Algorithm 1 + other models

    model_keys = ['Local Model', 'Global Model', 'CPAGM']
    report_valid = {key: [] for key in model_keys}
    report_test = {key:[] for key in model_keys}

    lags = list(range(2,17,2))
    lags_str = tuple([f'{lag}-Lags' for lag in lags])

    plot_lags = [4,8,10,16]
    plot_samples = False

    for lag in lags:

        plot_samples = True if lag in plot_lags else False

        print (f'Lag Features: {lag}')
        valid_mae, test_mae = local_model(series_set,SPLIT_POINT,lag,sample_plot=plot_samples) #LM
        report_valid[model_keys[0]].append(valid_mae)
        report_test[model_keys[0]].append(test_mae)

        valid_mae,test_mae = global_model(series_set, lag, SPLIT_POINT,sample_plot= plot_samples) #GM
        report_valid[model_keys[1]].append(valid_mae)
        report_test[model_keys[1]].append(test_mae)

        valid_mae, test_mae = main_algorithm(series_set,lag,NUM_CLUSTERS,SPLIT_POINT,CONVERGE_LIMIT,sample_plot=plot_samples, train_plot=plot_samples) #CPAGM
        report_valid[model_keys[2]].append(valid_mae)
        report_test[model_keys[2]].append(test_mae)

        print('')
    
    # Bar plot for comparison of models
    
    x = np.arange(len(lags_str))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for key, mae in report_test.items():

        offset = width * multiplier
        rects = ax.bar(x + offset, mae, width, label=key)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('MAE Test Error')
    ax.set_title('Comparing MAE of Each Model With Same Lag Feature')
    ax.set_xticks(x + width, lags_str)
    ax.legend(loc='upper left', ncols=len(model_keys))
    ax.set_ylim(0, 1500)

    plt.show()


    # Algorithm 2 
    all_custers = list(range(3,6))
    cluster_keys = tuple([f'{i}_Clusters' for i in all_custers])
    report_valid = {key: [] for key in cluster_keys}
    report_test = {key:[] for key in cluster_keys}
    

    for lag,num_clusters in product(lags,all_custers):
        print(f'Lag Features: {lag}, Number of clusters: {num_clusters}')
        a = num_clusters-3
        valid_mae,test_mae = main_algorithm(series_set,lag,num_clusters,SPLIT_POINT,CONVERGE_LIMIT)
        report_valid[cluster_keys[a]].append(valid_mae)
        report_test[cluster_keys[a]].append(test_mae)
        print('')

    x = np.arange(len(lags_str))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for key, mae in report_test.items():

        offset = width * multiplier
        rects = ax.bar(x + offset, mae, width, label=key)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('MAE Test Error')
    ax.set_title('Comparing MAE of Each Model With Same Lag Feature')
    ax.set_xticks(x + width, lags_str)
    ax.legend(loc='upper left', ncols=len(model_keys))
    ax.set_ylim(0, 1500)

    plt.show()