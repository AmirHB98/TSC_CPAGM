import matplotlib.pyplot as plt
from utils.process_data import Xy_Split
import random


def plot_cluster(train_clusters: dict, test_clusters: dict, prototypes: dict, lag: int, split_point: int, num_plots: int =2 ):

    for key in list(train_clusters.keys()):

        fig , ax = plt.subplots(num_plots,2, sharey= 'row')
        fig.suptitle(f'Random Members of {key} Along Their {lag}-Lagged Prediction')
        ax[0,0].set_title('Validation Samples')
        ax[0,1].set_title('Test Samples')
        ax[num_plots-1,0].set_xlabel('Hour (h)')
        ax[num_plots-1,1].set_xlabel('Hour (h)')

        model = prototypes[key]
        train_list = train_clusters[key]
        test_list = test_clusters[key]

        plot_samples(ax, lag, split_point, train_list, test_list, model)


def plot_samples(ax, lag: int, split_point: int, train_list: list, test_list: list, model: object, num_plots: int =2):
    """_summary_

    Parameters
    ----------
    ax : numpy array
        _description_
    lag : int
        _description_
    split_point : int
        _description_
    train_list : list
        _description_
    test_list : list
        _description_
    model : object
        _description_
    num_plots : int
        _description_
    """
    for i in range(num_plots):

        j = random.randint(0,len(train_list)-1)

        series = train_list[j]

        X,y = Xy_Split([series],lag)
        y_pred = model.predict(X)
            
        x_axis = range(lag,split_point)
        ax[i,0].plot(x_axis,y, marker = 'o', label = 'Actual')
        ax[i,0].plot(x_axis,y_pred, marker = 'o', label = 'Preddiction', linestyle= '--')
        ax[i,0].set_ylabel('Number of Pedestrians')
        ax[i,0].legend()

        series = test_list[j]

        X,y = Xy_Split([series],lag)
        y_pred = model.predict(X)

        x_axis = range(split_point,24)
        ax[i,1].plot(x_axis,y, marker = 'o', label = 'Actual')
        ax[i,1].plot(x_axis,y_pred, marker = 'o', label = 'Preddiction', linestyle= '--')
        ax[i,1].legend()
    
    plt.show()