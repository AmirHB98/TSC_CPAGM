from scipy.io import arff
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

SPLIT_POINT = 19
CONVERGE_LIMIT = 0.95
NUM_CLUSTERS = 2
NUM_LAG = 4
NUM_PLOTS = 2
MODE = 'train'

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

def distribute_randomly(series_set, num_clusters:int = NUM_CLUSTERS):
    """_summary_

    Parameters
    ----------
    series_set : list
        List which contains all numpy arrays
    num_clusters : int
        Number of clusters

    Returns
    -------
    clusters: dict
        Clusters with randomly assigned series in each
    """
    clusters = {f'cluster_{i+1}': [] for i in range(num_clusters)}
    random.shuffle(series_set)    # Shuffle the series for random selection

    for series in series_set:
        key = random.choice(list(clusters.keys()))
        clusters[key].append(series)
    
    return clusters

def create_sub_sets(clusters: dict, split_point : int = SPLIT_POINT):
    """ Creates a train sets for each cluster

    Parameters
    ----------
    clusters : dict
        Clusters that should be used to extract prototypes
    split_point : int, optional
        Number of bservations in each series required for training, by default 19

    Returns
    -------
    train_cluster: dict
        A dictionary similar to input with limited observations
    test_cluster: dict
        A dictionary similar to input but containing only h final observations
    """

    train_clusters = {}
    test_clusters = {}
    for key, cluster in clusters.items():
        train_clusters[key] = [series[:split_point] for series in cluster]
        test_clusters[key] = [series[split_point+1:] for series in cluster]
    
    return train_clusters, test_clusters

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
        
def create_prototypes(clusters: dict, lags: int) -> dict:
    """Creates a trained model for each cluster

    Parameters
    ----------
    clusters : dict
        variable that contains all clusters
    lags : int
        Lag features

    Returns
    -------
    Prtotypes: dict
        Trained model for each cluster according to its members
    """

    prototypes = {}
    for key,cluster in clusters.items():
        if cluster:
            X,y = Xy_Split(cluster,lags)

            model = LinearRegression()
            model.fit(X,y)

        prototypes[key] = model
    
    return prototypes


def reassign_clusters(prototypes : dict, series_set: list, lags: int, split_point: int = SPLIT_POINT) -> dict:

    clusters = {key : [] for key in list(prototypes.keys())}


    for series in series_set:

        train_series = series[:split_point]
        X,y = Xy_Split([train_series], lags)
        
        Errors = {}

        for key, model in prototypes.items():
            y_predict = model.predict(X)
            Errors[key] = mean_absolute_error(y, y_predict)
        
        cluster_key = min(Errors, key=Errors.get)

        clusters[cluster_key].append(series)
    
    return clusters

def converge_clusters(old_clusters: dict, new_clusters:dict, converge_limit: float = CONVERGE_LIMIT):
    converge_rate_dict = {key : 0.0 for key in list(old_clusters.keys())}
    for key in old_clusters.keys():
        if key in new_clusters: # keys should co-exist in input dictionaries, this is just in case ...
            # Now check wether each key has same members
            old_set = set(tuple(array) for array in old_clusters[key])
            new_set = set(tuple(array) for array in new_clusters[key])

            converge_rate = len(old_set.intersection(new_set))/len(new_set)
            print(f'{key} converge rate: {converge_rate:.3f}, len_old: {len(old_set)}, len_new: {len(new_set)}')
            
        else:
            return converge_rate_dict, False
        
        converge_rate_dict[key] = converge_rate
    print('\n')
    if min(converge_rate_dict.values()) > converge_limit:
        return converge_rate_dict,True
    else:
        return converge_rate_dict,False

def in_sample_MAE(clusters: dict,prototypes: dict, lags : int, mode : str = MODE, split_point : int = SPLIT_POINT):
    
    valid_error = {}
    
    for key ,cluster in clusters.items():
        if key in list(prototypes.keys()):
            X,y = Xy_Split(cluster,lags)
            y_predict = prototypes[key].predict(X)
        
        if mode == 'train':
            y_act = y[:split_point-lag]
            y_pred = y_predict[:split_point-lag]
        elif mode == 'test':
            y_act = y[split_point-lag:]
            y_pred = y_predict[split_point-lag:]
        else:
            raise "Not a proper mode"
        
        valid_error[key] = round(mean_absolute_error(y_act,y_pred),ndigits=3)

    return valid_error

def plot_cluster(clusters: dict,prototypes: dict, lags: int, num_plots: int = NUM_PLOTS, mode : str = MODE, split_point : int = SPLIT_POINT):

    #TODO: add column name

    keys = list(clusters.keys())

    fig , ax = plt.subplots(num_plots, len(keys))
    fig.suptitle(f'Random members of each cluster vs. {lags}-lagged prediction in {mode} mode')
    
    for i , key in enumerate(keys):
        for n in range(num_plots):
            series = random.choice(clusters[key])
            x_axis = list(range(len(series)))

            X,y = Xy_Split([series],lags)
            y_predict = prototypes[key].predict(X)

            y_train = y[:split_point-lag]
            y_test = y[split_point-lag:]

            y_predict_train = y_predict[:split_point-lag]
            y_predict_test = y_predict[split_point-lag:]


            if mode == 'train':
                x_axis = x_axis[lag:split_point]
                y_pred = y_predict_train
                y_act = y_train
            elif mode == 'test':
                x_axis = x_axis[split_point:]
                y_pred = y_predict_test
                y_act = y_test
            else:
                assert "Not a proper mode"
            ax[n,i].plot(x_axis,y_act, label='Real series', linestyle='-', marker='o')
            ax[n,i].plot(x_axis,y_pred, label='Prediction results', linestyle='-')
            ax[n,i].set_xlabel('Hour (h)')
            ax[n,i].set_ylabel('Number of Pedestrians')
            ax[n,i].legend()
    
    plt.show()

def loc_to_glob(local_dict : dict, global_dict: dict):
    for key in list(global_dict.keys()):
        global_dict[key].append(local_dict[key])



#TODO: yaml

#TODO: ARI plot

#TODO: Algorithm2 
    
#TODO: README with some plots
        
#TODO: GLobal Model and Local Model


if __name__== '__main__':
    path = os.getcwd() +'\CPAGM\Data\Chinatown_TEST.arff'
    series_set, series_label = get_data(path) # retireve data from dataset
    path = os.getcwd() +'\CPAGM\Data\Chinatown_TRAIN.arff'
    series_set2, serises_label2 = get_data(path)
    series_set.extend(series_set2)

    lag = 4
    clusters = distribute_randomly(series_set)
    Keys = list(clusters.keys())
    global_converge_rate = {key : [] for key in Keys}
    global_valid_MAE = {key: [] for key in Keys}
    global_test_MAE = {key: [] for key in Keys}

    steps = 0
    done = False

    while not done:

        train_clusters,test_clusters = create_sub_sets(clusters)
        prototypes = create_prototypes(train_clusters, lag)
        new_clusters = reassign_clusters(prototypes,series_set,lag)
        valid_MAE = in_sample_MAE(new_clusters,prototypes,lag)
        test_MAE = in_sample_MAE(new_clusters,prototypes, lag, mode = 'test')
        loc_to_glob(valid_MAE,global_valid_MAE)
        loc_to_glob(test_MAE, global_test_MAE)
        print(f'Training step {steps}: In-sample MAE {valid_MAE} , out-sample test MAE {test_MAE}')
        converage_rate_step,done = converge_clusters(clusters,new_clusters)
        loc_to_glob(converage_rate_step,global_converge_rate)
        clusters = new_clusters
        steps += 1
    

    fig , ax = plt.subplots(len(Keys), sharex= True)
    for i,key in enumerate(Keys):
        ax[i].plot(global_converge_rate[key], label='Convergance Rate', linestyle='-', marker='o')
        ax[i].set_ylabel('COnvergance Rate')

    plt.show()
    
    fig , ax = plt.subplots(len(Keys), sharex= True)
    for i,key in enumerate(Keys):
        ax[i].plot(global_test_MAE[key], label='Test MAE', linestyle='-', marker='o')
        ax[i].plot(global_valid_MAE[key], label= 'Validation MAE',linestyle='-', marker='o' )
        ax[i].set_ylabel('MAE')
        ax[i].legend()
        
    plt.show()

    plot_cluster(new_clusters,prototypes,lag)
    plot_cluster(new_clusters,prototypes,lag, mode= 'test')



# for l in range(2,17,2):

# test set : the last 5 observations of each dataset (h = 5)
# training period: the forst 19 observations of each series
# validation period: observation from l+1 to 19 