from utils.process_data import Xy_Split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import random

def local_model(series_set: list, split_point : int, lag: int,sample_plot : bool = False, num_plots : int =2):
    """For each series, fits a uniqe model according to the train_set. That is 363 models trained during this process

    Parameters
    ----------
    series_set : list
        List which contains all of the series in chinatown dataset
    split_point : int
        Where trains_set and test set are divided
    lag : int
        Lag features for training models
    sample_plot: bool
        If you want to plot some series along with their predictions randomly set it to True
    num_plots : int, optional
        Sample visualizations to be plotted while training the model, by default 2
    """

    valid_MAE =[]
    test_MAE = []
    to_plot = [random.randint(0,len(series_set)-1) for _ in range(num_plots)]
    if sample_plot:
        fig, ax = plt.subplots(num_plots,2, sharey= 'row')
        fig.suptitle(f'Sampled Prediction Using Local Model (LM) for {lag} lags')
        ax[0,0].set_title('Validation Samples')
        ax[0,1].set_title('Test Samples')
        a = 0

    for i,series in enumerate(series_set):

        train_series = series[:split_point] 
        test_series = series[split_point-lag:]

        X,y_train = Xy_Split([train_series],lag)

        model = LinearRegression()
        model.fit(X,y_train)
        y_pred_train = model.predict(X)

        valid_MAE.append(mean_absolute_error(y_train,y_pred_train))

        X,y_test = Xy_Split([test_series],lag)
        y_pred_test = model.predict(X)

        test_MAE.append(mean_absolute_error(y_test,y_pred_test))

        if sample_plot:
            if i in to_plot:

                x_axis = range(lag,split_point)
                ax[a,0].plot(x_axis,y_train, marker ='o', label = 'Actual series')
                ax[a,0].plot(x_axis,y_pred_train , marker = 'o', label = 'Predicted series', linestyle = '--')
                ax[a,0].set_ylabel('Number of Pedestrians')

                x_axis = range(split_point,len(series))
                ax[a,1].plot(x_axis,y_test, marker ='o', label = 'Actual series')
                ax[a,1].plot(x_axis,y_pred_test , marker = 'o', label = 'Predicted series', linestyle = '--')

                if a == num_plots-1:
                    ax[a,0].set_xlabel('Hour (h)')
                    ax[a,1].set_xlabel('Hour (h)')
                    plt.show()
                a += 1
            

    
    mean_valid_MAE = np.mean(valid_MAE)
    mean_test_MAE = np.mean(test_MAE)

    print(f'Local model -Validation MAE :{mean_valid_MAE: .2f}')
    print(f'Local model - Test MAE {mean_test_MAE: .2f}')