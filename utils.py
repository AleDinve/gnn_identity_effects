import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import torch


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
    
def cycle_position(n):
    f = lambda x: [np.cos(x),np.sin(x)]
    betas = [2*k*np.pi/n + np.pi for k in range(n)]
    b = []
    for k in range(n):
        b.append(f(betas[k]))
    pos = np.array(b)
    return pos
    

def node_position(n,m):
    # function used to define the position of the cycles in a dipolar graph, 
    # making each cycle a regular polygon
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
    with open('params_dicyclic.json') as f:
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

def tensorboard_writer(writer, train_loss, train_acc, test_loss, test_acc, epoch):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)

def histogram_dicyclic_saver(histogram_data, data_dict, n_max,  model, dataset, it, num_layers, device):
    test_list = ['3_3', str(n_max-2)+'_'+str(n_max-2), str(n_max-1)+'_'+str(n_max-1), str(n_max)+'_'+str(n_max),
                 str(n_max-1)+'_'+str(n_max),str(n_max)+'_'+str(n_max-4),
                 str(4)+'_'+str(n_max-1),str(6)+'_'+str(n_max)]
    for gr in test_list:
        data = next(iter(data_dict[gr])).to(device)
        histogram_data.append({'word': gr, 'rating': float(model(data.x, data.edge_index, data.batch)), 'it': it,
                                'num_layers': num_layers, 'dataset': dataset})
    return histogram_data

def histogram_data_saver(histogram_data, data_dict, model, dataset, it, num_layers, device):
    words_list = ['AA', 'xy', 'YY', 'ZZ','YZ','ZT','EY','SZ']
    for word in words_list:
        data = data_dict[word].to(device)
        out, _ = model(data.x, data.edge_index, data.batch)
        histogram_data.append({'word': word, 'rating': float(out), 'it': it,
                                'num_layers': num_layers, 'dataset': dataset})
    return histogram_data


def save_tensors(raws, hiddens, path, num_l, sep_vec = True):
    hidden_raw = raws[0]
    x1_raw = raws[1]
    x2_raw = raws[2]
    hidden_list = hiddens[0]
    hidden_list1 = hiddens[1]
    hidden_list2 = hiddens[2]

    if sep_vec:
        x1_pd = pd.DataFrame.from_records(x1_raw)
        x1_pd.to_csv(path+str(num_l)+'x1_features.csv')
        x2_pd = pd.DataFrame.from_records(x2_raw)
        x2_pd.to_csv(path+str(num_l)+'x2_features.csv')
        torch.save(hidden_list1, path+str(num_l)+'hidden_x1.pt')
        torch.save(hidden_list2, path+str(num_l)+'hidden_x2.pt')

    hidden_pd = pd.DataFrame.from_records(hidden_raw)
    hidden_pd.to_csv(path+str(num_l)+'hidden_features.csv')
    torch.save(hidden_list, path+str(num_l)+'hidden_features.pt') 

def prediction(model, hd, n_max, gap, gnn_type, iter_dl, sep_vec, df_data, device, path, num_l, it):
    n_graphs = n_max+gap-2
    hidden_list1 = torch.zeros((n_graphs,n_graphs,hd))
    hidden_list2 = torch.zeros((n_graphs,n_graphs,hd))

    if gnn_type =='gconv_stack':
        hd_stack = 2*hd
    else:
        hd_stack = hd
    hidden_list = torch.zeros((n_graphs,n_graphs,hd_stack))
        
    hidden_raw = []
    x1_raw = []
    x2_raw = []
    for i in range(3,n_max+gap+1):
        for j in range(3,n_max+gap+1):
            data = next(iter_dl).to(device)
            y, hidden = model(data.x, data.edge_index, data.batch)
            if gnn_type == 'gconv_glob':
                sep_vec = False
            else:                         
                x1 = hidden[1]
                x2 = hidden[2]
            hidden = hidden[0]
            hidden = hidden/torch.norm(hidden)
            hidden_list[i-3,j-3,:] = hidden
            hidden_raw.append({'1st_cycle':i, '2nd_cycle':j, 'hidd':hidden.cpu().detach().numpy() })
            if sep_vec:
                x1 = x1/torch.norm(x1)
                x2 = x2/torch.norm(x2)
                hidden_list1[i-3,j-3,:] = x1
                hidden_list2[i-3,j-3,:] = x2
                x1_raw.append({'1st_cycle':i, '2nd_cycle':j, 'hidd':x1.cpu().detach().numpy() })
                x2_raw.append({'1st_cycle':i, '2nd_cycle':j, 'hidd':x2.cpu().detach().numpy() })
            if j>=i:             
                df_data.append({'1st_cycle':i,'2nd_cycle':j,'rating': float(y), 'it': it+1, 'num_layers':num_l})
                if j!=i:
                    df_data.append({'1st_cycle':j,'2nd_cycle':i,'rating': float(y), 'it': it+1, 'num_layers':num_l})
                
    # hiddens = [hidden_list, hidden_list1, hidden_list2]
    # raws = [hidden_raw, x1_raw, x2_raw]
    # os.makedirs(path+str(hd)+'_hd/')
    # save_tensors(raws, hiddens, path+str(hd)+'_hd/', num_l, sep_vec = True)             
    
    
    dot_prod = torch.zeros((n_graphs**2,n_graphs**2))

    for i in range(0, n_graphs**2):
        for j in range(0, n_graphs**2):
            [i1, i2] = np.unravel_index(i, (n_graphs,n_graphs))
            [j1, j2] = np.unravel_index(j, (n_graphs,n_graphs))
            dot_prod[i,j] = torch.dot(hidden_list[i1,i2,:],hidden_list[j1,j2,:])
    np.savetxt(path+str(num_l)+'_scalar_products.txt',dot_prod.detach().numpy())

    return df_data



if __name__ == '__main__':
    plot_saved_data()