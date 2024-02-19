
from sklearn.linear_model import LinearRegression
from utils.process_data import Xy_Split
from sklearn.metrics import mean_absolute_error
import random

def global_model(series_set:list, lag: int , split_point :int , num_plots : int = 2 ):
    """An autoregressive model to predict all fo the series

    Parameters
    ----------
    lag : int, optional
        Lag features for training the model, by default NUM_LAG
    split_point : int, optional
        Number of elements assigned to train and test for each series, by default SPLIT_POINT
    """

    train_set = [array[:split_point] for array in series_set]
    test_set = [array[split_point-lag:] for array in series_set]
    h = 24 - split_point

    model = LinearRegression()

    X,y = Xy_Split(train_set ,lag)
    model.fit(X,y)
    y_pred = model.predict(X)

    valid_MAE = mean_absolute_error(y,y_pred)
    print(f'Global model - Validation MAE: {valid_MAE :.2f}')

    to_plot = [random.randint(0,len(series_set)-1) for _ in range(num_plots)]
    
    fig , ax = plt.subplots(num_plots,sharex=True)
    fig.suptitle('In-sample prediction for validation set')
    for i in range(num_plots):
        ax[i].plot(y[(split_point-lag)*i:(split_point-lag)*(i+1)], marker = 'o', label = 'Actual')
        ax[i].plot(y_pred[(split_point-lag)*i:(split_point-lag)*(i+1)], marker = 'o', label = 'Preddiction', linestyle= '--')
        ax[i].legend()
    plt.show()
    

    X,y = Xy_Split(test_set,lag)
    y_pred = model.predict(X)
    test_MAE = mean_absolute_error(y,y_pred)
    print(f'Global model -Test MAE : {test_MAE}')

    fig , ax = plt.subplots(num_plots,sharex=True)
    fig.suptitle('Out-sample prediction for test set')
    for i in range(num_plots):
        ax[i].plot(y[h*i:h*(i+1)], marker = 'o', label = 'Actual')
        ax[i].plot(y_pred[h*i:h*(i+1)], marker = 'o', label = 'Preddiction', linestyle= '--')
        ax[i].legend()
    plt.show()