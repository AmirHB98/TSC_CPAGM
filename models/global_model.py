
from sklearn.linear_model import LinearRegression
from utils.process_data import Xy_Split
from sklearn.metrics import mean_absolute_error
import random
import matplotlib.pyplot as plt

def global_model(series_set:list, lag: int , split_point :int , num_plots : int = 2, sample_plot : bool = False ):
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
    print(f'Global model -Test MAE : {test_MAE}')

    if sample_plot:
        fig , ax = plt.subplots(num_plots,2,sharey='row')
        fig.suptitle('Sampled Prediction Using Global Model (GM)')
        ax[0,0].set_title('Validation Samples')
        ax[0,1].set_title('Test Samples')

        for i in range(num_plots):

            j = random.randint(0,len(series_set)-1)
            plot_train_set = train_set[j]
            plot_test_set = test_set[j]

            X,y = Xy_Split([plot_train_set],lag)
            y_pred = model.predict(X)
            
            x_axis = range(lag,split_point)
            ax[i,0].plot(x_axis,y, marker = 'o', label = 'Actual')
            ax[i,0].plot(x_axis,y_pred, marker = 'o', label = 'Preddiction', linestyle= '--')
            ax[i,0].set_ylabel('Number of Pedestrians')
            ax[i,0].legend()

            X,y = Xy_Split([plot_test_set],lag)
            y_pred = model.predict(X)

            x_axis = range(split_point,24)
            ax[i,1].plot(y, marker = 'o', label = 'Actual')
            ax[i,1].plot(y_pred, marker = 'o', label = 'Preddiction', linestyle= '--')
            ax[i,1].legend()

        plt.show()