import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class EarlyStopping():
    """
    Early stopping class to stop the training whenever
    patience runs out.

    Arguments
    ---------
    min_delta: float
        Minimum delta to consider the value passed as worse
        than the best value.
    patience: int
        Maximum number of worse values before stopping the
        training.
    """
    def __init__(self, min_delta: float = 0, patience: int = 0):
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -float("Inf")
        self.stop_training = False
    def on_epoch_end(self, epoch:int , current_value: float) -> bool:
        """
        Checks the value for the given epochs and decides whether to
        increase or reset the number of epochs worse than best. 
        If the number of worse epochs becomes equal to the patience,
        the training is stopped.
        
        Arguments
        ---------
        epoch: int
            The current epoch number.
        current_value:
            The current loss value.
            
        Returns
        -------
        stop_training: bool
            Whether or not the current training should be stopped.
        """
        if self.best < current_value:
            self.best = current_value
            self.wait = 0
        elif self.best <= current_value+self.min_delta:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
        # the remaining case is that self.best is in the following range, i.e.
        # (current_value - self.min_delta, current_value], then we do not reset the
        # patience nor change the best
        return self.stop_training

def node_position(n,m):
    # function used to define the position of the cycles in a dipolar graph, 
    # making teach cycle a regular polygon
    f1 = lambda x: [np.cos(x)-2,np.sin(x)]
    f2 = lambda x: [np.cos(x)+2,np.sin(x)]
    alfas = [2*k*np.pi/n + 2*np.pi/n for k in range(n)]
    betas = [2*k*np.pi/m + np.pi for k in range(m)]
    a = []
    for k in range(n):
        a.append(f1(alfas[k]))   
    b = []
    for k in range(m):
        b.append(f2(betas[k]))
    pos = np.array(a+b)

    return pos

    


def plot_saved_data(path, file_name = None, online = True, df = None, num_l = 12):
    if online:
        df = pd.read_csv(file_name)
    list_col = df.columns
    if online:
        list_col = list_col.drop(df.columns[0])
        df = df[list_col]
    with open('params_bipolar.json') as f:
        params = json.load(f)
    n_max = params['n_max']
    gap = params['gap']
    list_red = []
    for i in range(3,n_max+gap+1):
        for j in range(3,n_max+gap+1):
            df_red = df[df['1st_cycle']==i]
            df_red = df_red[df_red['2nd_cycle']==j]

            rating_std = df_red['rating'].std()
            rating_mean = df_red['rating'].mean()
            list_red.append({'1st_cycle':i, '2nd_cycle':j, 'rating':rating_mean, 'std': rating_std})

    red = pd.DataFrame.from_records(list_red)
    enc = 'dipolar'
    sns.relplot(data=red, x = '1st_cycle', y = '2nd_cycle', hue = 'rating', size = 'std')
    plt.savefig(path+'img_'+str(num_l)+'_layers_'+ enc +'.png')
    plt.close()

def make_path(dataset_type, gnn_type):
    now = lambda x: datetime.now().strftime("%"+x)
    date_list = [now('d'),now('m'), now('y')]
    date_str = "_".join(date_list)
    trial = 0
    path_f = lambda x: "/".join([dataset_type, date_str, gnn_type, 'trial_'+str(x)])+'/'
    while os.path.exists(path_f(trial)): 
        trial+=1
    path = path_f(trial)
    os.makedirs(path)
    os.makedirs(path+'hist/')
    os.makedirs(path+'test_loss/')
    return path


if __name__ == '__main__':
    plot_saved_data()