import random
from sklearn.linear_model import LinearRegression
from utils.process_data import Xy_Split
from sklearn.metrics import mean_absolute_error
from utils.evaluate import in_cluster_MAE, converge_clusters
from utils.process_data import create_sub_sets
import numpy as np

def distribute_randomly(series_set, num_clusters:int):
    """ Distributes available time series to each cluster for initializing the algorithm

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

def reassign_clusters(prototypes : dict, series_set: list, lags: int, split_point: int) -> dict:
    """ Reassign clusters accorfing to minimum MAE from prototype

    Parameters
    ----------
    prototypes : dict
        The trained model for each cluster
    series_set : list
        all series in the problem
    lags : int
        Lag features for training model and testing model
    split_point : int, optional
        Number of elements in training series, by default SPLIT_POINT

    Returns
    -------
    dict
        _description_
    """

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

def loc_to_glob(local_dict : dict, global_dict: dict):
    """Assigna values from local dictionary (inside a loop) to global dictionary

    Parameters
    ----------
    local_dict : dict
        Dictionary inside a loop
    global_dict : dict
        Global dictionary
    """
    for key in list(global_dict.keys()):
        global_dict[key].append(local_dict[key])

def main_algorithm(series_set : list, lag : int, num_clusters : int, split_point : int, ARI : bool = True ):
    


    clusters = distribute_randomly(series_set, num_clusters)
    Keys = list(clusters.keys())
    global_converge_rate = {key : [] for key in Keys}
    global_valid_MAE = {key: [] for key in Keys}
    global_test_MAE = {key: [] for key in Keys}

    steps = 0
    done = False

    while not done:

        train_clusters,test_clusters = create_sub_sets(clusters,split_point,lag)
        prototypes = create_prototypes(train_clusters, lag)
        valid_MAE = in_cluster_MAE(train_clusters,prototypes,lag)
        test_MAE = in_cluster_MAE(test_clusters,prototypes,lag)
        loc_to_glob(valid_MAE,global_valid_MAE)
        loc_to_glob(test_MAE, global_test_MAE)
        print(f'Training step {steps}: \nIn-sample MAE {valid_MAE} \nout-sample test MAE {test_MAE}')
        new_clusters = reassign_clusters(prototypes,series_set,lag)
        converage_rate_step,done = converge_clusters(clusters,new_clusters)
        loc_to_glob(converage_rate_step,global_converge_rate)
        clusters = new_clusters
        steps += 1
    
    if ARI:
        cluster_label = np.zeros(len(series_set))

        for i,series in enumerate(series_set):
            for key,c_series in clusters.items():
                if np.any([np.array_equal(x,series) for x in c_series]):
                    cluster_label[i] = int(key[-1])
                    
                
        cluster_label = list(cluster_label)
    
    # for key in Keys:
        


    for key in Keys:
       plt.plot(global_converge_rate[key], label=f'Convergance Rate in {key}', linestyle='-', marker='o')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Convergance Rate')
    plt.title(' Convergence of Each Cluster')
    plt.legend()

    plt.show()
    
    fig , ax = plt.subplots(len(Keys), sharex= True)
    for i,key in enumerate(Keys):
        ax[i].plot(global_test_MAE[key], label='Test MAE', linestyle='-', marker='o')
        ax[i].plot(global_valid_MAE[key], label= 'Validation MAE',linestyle='-', marker='o' )
        ax[i].set_ylabel('MAE')
        ax[i].legend()
        
    plt.show()

    plot_cluster(clusters,prototypes,lag)
    plot_cluster(clusters,prototypes,lag, mode= 'test')

    if ARI:
        return cluster_label
