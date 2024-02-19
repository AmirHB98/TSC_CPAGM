from sklearn.metrics import adjusted_rand_score
from utils.process_data import summon_all_series
from models.global_model import global_model
from models.local_model import local_model
from models.model_1 import main_algorithm

SPLIT_POINT = 19
CONVERGE_LIMIT = 0.99
NUM_CLUSTERS = 2
NUM_LAG = 4
NUM_PLOTS = 2

#TODO: yaml or json

#TODO: Algorithm2 
    
#TODO: README with some plots
        
#TODO: plots Fig2, Fig5, Fig6

#TODO: print errors for all algorithms together at the end + bar chart

#TODO: print convergence rate for test set.

if __name__ == '__main__':

    series_set, series_label = summon_all_series()
    local_model(series_set,SPLIT_POINT,NUM_LAG) #LM
    global_model(series_set, NUM_LAG, SPLIT_POINT)

    new_label = main_algorithm(series_set,lag=10)

    ari_4 = adjusted_rand_score(series_label,new_label)
    print(ari_4)

    # for l in range(2,17,2):
 