
from sklearn.linear_model import LinearRegression
from utils.process_data import Xy_Split
from sklearn.metrics import mean_absolute_error
from utils.plot import plot_samples
import random
import matplotlib.pyplot as plt

def global_model(series_set:list, lag: int , split_point :int , sample_plot : bool = False, num_plots : int = 2 ):
    """An autoregressive model to predict all fo the series

    Parameters
    ----------
    series_set: list
        List whcih contains all series in chinatown
    lag : int, optional
        Lag features for training the model, by default NUM_LAG
    split_point : int, optional
        Number of elements assigned to train and test for each series, by default SPLIT_POINT
    sample_plot: bool
        Plots random series along with their predcition using the trained global model
    num_plots: int
        number of sampled series for plotting sample plots
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

    X,y = Xy_Split(test_set,lag)
    y_pred = model.predict(X)

    test_MAE = mean_absolute_error(y,y_pred)
    print(f'Global model - Test MAE : {test_MAE}')

    if sample_plot:
        fig , ax = plt.subplots(num_plots,2,sharey='row')
        fig.suptitle(f'Sampled Prediction Using Global Model (GM) for {lag} lags')
        ax[0,0].set_title('Validation Samples')
        ax[0,1].set_title('Test Samples')
        ax[num_plots-1,0].set_xlabel('Hour (h)')
        ax[num_plots-1,1].set_xlabel('Hour (h)')

        plot_samples(ax,lag, split_point, train_set, test_set, model)
        