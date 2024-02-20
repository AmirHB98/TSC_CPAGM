from sklearn.metrics import adjusted_rand_score
from utils.process_data import summon_all_series
from models.global_model import global_model
from models.local_model import local_model
from models.model_1 import main_algorithm

SPLIT_POINT = 19
CONVERGE_LIMIT = 0.99
NUM_CLUSTERS = 2
NUM_PLOTS = 2

#TODO: yaml or json

#TODO: Algorithm2 
    
#TODO: README with some plots
        
#TODO: plots Fig2, Fig5, Fig6

#TODO:  errors for all algorithms bar chart

if __name__ == '__main__':
    
    series_set, series_label = summon_all_series()
    for lag in range(2,17,2):
        print (f'Lag Features: {lag}')
        local_model(series_set,SPLIT_POINT,lag) #LM
        global_model(series_set, lag, SPLIT_POINT) #GM
        main_algorithm(series_set,lag,NUM_CLUSTERS,SPLIT_POINT,CONVERGE_LIMIT) #CPAGM
        print('')

    # ari_4 = adjusted_rand_score(series_label,new_label)
    # print(ari_4)


 