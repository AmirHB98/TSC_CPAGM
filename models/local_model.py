from utils.process_data import Xy_Split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import random

def local_model(series_set: list, split_point : int, lag: int, num_plots : int =2):
    """_summary_

    Parameters
    ----------
    series_set : list
        _description_
    split_point : int
        _description_
    lag : int
        _description_
    num_plots : int, optional
        Sample visualizations to be plotted while training the model, by default 2
    """

    valid_MAE =[]
    test_MAE = []
    to_plot = [random.randint(0,len(series_set)-1) for _ in range(num_plots)]
    fig, ax = plt.subplots(num_plots, sharex= True)
    fig.suptitle('Sample visualizations using Local Model (LM)')
    a = 0

    for i,series in enumerate(series_set):

        train_series = series[:split_point] 
        test_series = series[split_point-lag:]

        X,y = Xy_Split([train_series],lag)

        model = LinearRegression()
        model.fit(X,y)
        y_pred_train = model.predict(X)

        valid_MAE.append(mean_absolute_error(y,y_pred_train))

        X,y = Xy_Split([test_series],lag)
        y_pred_test = model.predict(X)

        test_MAE.append(mean_absolute_error(y,y_pred_test))

        if i in to_plot:
            y_pred = np.concatenate([y_pred_train,y_pred_test],axis=0)
            ax[a].plot(series, marker ='o', label = 'Actual series')
            ax[a].plot(range(lag,len(series)), y_pred, marker = 'o', label = 'Predicted series')
            ax[a].set_ylabel('Number of Pedestrians')
            if a == num_plots-1:
                ax[a].set_xlabel('Hour (h)')
            a += 1
            
    plt.show()

    
    mean_valid_MAE = np.mean(valid_MAE)
    mean_test_MAE = np.mean(test_MAE)

    print(f'Local model -Validation MAE :{mean_valid_MAE: .2f}')
    print(f'Local model - Test MAE {mean_test_MAE: .2f}')