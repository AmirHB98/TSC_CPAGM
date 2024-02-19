from utils.process_data import Xy_Split
from sklearn.metrics import mean_absolute_error

def converge_clusters(old_clusters: dict, new_clusters:dict, converge_limit: float):
    """Decides wether the current clustering is good enough

    Parameters
    ----------
    old_clusters : dict
        The cluster that prototype is trained with
    new_clusters : dict
        The cluster that is formed by assigning each series to the cluster with least MAE
    converge_limit : float, optional
        Algoorithm is stopped if this perecent of series are existing in new cluster from the old cluster, by default CONVERGE_LIMIT

    Returns
    -------
    converge_rate : dict
        A dictionary that assigns the convergence value to each cluster
    done : bool
        Wether the convergence has hit the thresholds
    """
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


def in_cluster_MAE(clusters: dict,prototypes: dict, lags : int):
    """Calculates MAE loss of the members of a cluster, calculates validation MAE if given train_set and calculates test MAE if given test_set

    Parameters
    ----------
    clusters : dict
        Dictionary that contains cluster names along its members
    prototypes : dict
        Dictionary taht contains cluster names and trained models
    lags : int
        Lag features

    Returns
    -------
    MAE_error: dict
        returns validation/test error for each cluster
    """
    valid_error = {}
    
    for key ,cluster in clusters.items():
        if key in list(prototypes.keys()):
            X,y_act = Xy_Split(cluster,lags)
            y_pred = prototypes[key].predict(X)
        
        valid_error[key] = round(mean_absolute_error(y_act,y_pred),ndigits=3)

    return valid_error
