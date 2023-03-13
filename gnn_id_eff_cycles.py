
import seaborn as sns
import torch
import torch_geometric

import pandas as pd
from utils import plot_saved_data, make_path
import matplotlib.pyplot as plt
from dataset_gen import dataset_generator, all_bipolar, not_all_bipolar, get_complete_dataset
from train import training
import os
import json

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def histogram_bipolar_saver(histogram_data, data_dict, n_max,  model, dataset, it, num_layers):
    test_list = ['3_3', str(n_max-2)+'_'+str(n_max-2), str(n_max-1)+'_'+str(n_max-1), str(n_max)+'_'+str(n_max),
                 str(n_max-1)+'_'+str(n_max),str(n_max)+'_'+str(n_max-4),
                 str(4)+'_'+str(n_max-1),str(6)+'_'+str(n_max)]
    for gr in test_list:
        data = next(iter(data_dict[gr])).to(device)
        histogram_data.append({'word': gr, 'rating': float(model(data.x, data.edge_index, data.batch)), 'it': it,
                                'num_layers': num_layers, 'dataset': dataset})
    return histogram_data


def main_bipolar(gnn_type, num_layers, num_reps, epochs, hd, lr, n_max, take_random, path, early):
    raw_data = []
    histogram_data = []
    if gnn_type == 'gconv':
        for num_l in num_layers:
            print(f'Number of layers: {num_l}')
            for it in range(num_reps):          
                print(str(it+1)+'-th iteration')
                #seed manual setting
                torch_geometric.seed.seed_everything(it*100)
                data_dict = dataset_generator(dataset='bipolar', n_max = n_max, take_random=take_random)
                model, raw_data = training(gnn_type, data_dict, 'bipolar', hd, lr, num_l, epochs, it, raw_data, early = early)
                histogram_data = histogram_bipolar_saver(histogram_data, data_dict, n_max,  model, 'bipolar', it, num_l)
    
    data = pd.DataFrame.from_records(raw_data)
    data.to_csv(path+'bipolar_data')
    histogram_pd = pd.DataFrame.from_records(histogram_data)
    histogram_pd.to_csv(path+'bipolar_histogram')
    return data, histogram_pd

def dipolar_comparison(json_path):
    #loading parameters from json
    with open(json_path) as f:
        params = json.load(f)
    gnn_type = params['gnn_type']
    num_reps = params['num_reps']
    n_max = params['n_max']
    min_num_layers = params['min_num_layers']
    max_num_layers = params['max_num_layers']
    layers =[i for i in range(min_num_layers, max_num_layers+1)]
    epochs = params['epochs']
    hd = params['hd']
    lr = params['lr']
    take_random = params['take_rand']
    enc = 'bipolar'
    gap = params['gap']
    early = params['early']
    k = params['k']

    #path configuration
    path = make_path('cycles', gnn_type)
    #save config in txt
    config_json = pd.read_json(json_path, typ='series')
    config_json.to_csv(path+'config.txt')
    #path = path+'/no_gap/'
    if not os.path.exists(path):
        os.makedirs(path)
    data_dict = get_complete_dataset(n_max, k)
    for num_l in layers:
        df_data = []
        raw_data = []
        print(f'Number of layers: {num_l}')
        
        for it in range(num_reps):       
            print(str(it+1)+'-th iteration')     
            #seed manual setting
            torch_geometric.seed.seed_everything(it*100)
            #data_dict = dataset_generator(dataset='bipolar', n_max = n_max, take_random=take_random)
            
            model, raw_data = training(gnn_type, data_dict, 'bipolar', hd, lr, num_l, epochs, it, raw_data, device=device, early=early)
            data_loader = all_bipolar(n_max,gap)

            iter_dl = iter(data_loader)
            for i in range(3,n_max+gap+1):
                for j in range(i,n_max+gap+1):
                    data = next(iter_dl).to(device)
                    y = model(data.x, data.edge_index, data.batch)                
                    df_data.append({'1st_cycle':i,'2nd_cycle':j,'rating': float(y), 'it': it+1, 'num_layers':num_l})
                    if j!=i:
                        df_data.append({'1st_cycle':j,'2nd_cycle':i,'rating': float(y), 'it': it+1, 'num_layers':num_l})
        df = pd.DataFrame.from_records(df_data)
        df.to_csv(path+'trial_nmax_'+str(n_max-2)+'_gap_'+str(gap+2)+'_'+str(num_l)+'_layers.csv')
            # sns.relplot(data=df, x = '1st_cycle', y = '2nd_cycle', hue = 'rating')
            # plt.savefig('prova_comparison_'+str(num_l)+'_layers_'+ enc +'.png')
            # plt.close()
        plot_saved_data(path = path, file_name = None, online=False, df=df, num_l = num_l)


def main(json_path):
    with open(json_path) as f:
        params = json.load(f)
    gnn_type = params['gnn_type']
    num_reps = params['num_reps']
    n_max = params['n_max']
    min_num_layers = params['min_num_layers']
    max_num_layers = params['max_num_layers']
    layers =[i for i in range(min_num_layers, max_num_layers+1)]
    epochs = params['epochs']
    hd = params['hd']
    lr = params['lr']
    take_random = params['take_rand']
    enc = 'bipolar'
    early: bool = params['early']

    #path generator: determined on day of execution
    path = make_path('cycles', gnn_type)
    #save config in txt
    config_json = pd.read_json(json_path, typ='series')
    config_json.to_csv(path+'config.txt')

    data, histogram = main_bipolar(gnn_type, layers, num_reps, epochs, hd, lr, n_max, take_random, path, early)
        
    data_enc = data[data['dataset']==enc]
    
        

    sns.lineplot(data=data_enc, x='epoch', y='test_loss')
    plt.savefig(path+'test_loss/test_loss_bipolar.png')
    plt.close()
    sns.catplot(data=histogram, kind='bar', x='word', y='rating')
    plt.savefig(path+'hist/hist_'+ gnn_type +'.png')
    plt.close()

if __name__ == '__main__':
    dipolar_comparison('params_bipolar.json')



'''
Template for main 
def main(json_path):
    with open(json_path) as f:
        params = json.load(f)
    gnn_type = params['gnn_type']
    num_reps = params['num_reps']
    n_max = params['n_max']
    min_num_layers = params['min_num_layers']
    max_num_layers = params['max_num_layers']
    layers =[i for i in range(min_num_layers, max_num_layers+1)]
    epochs = params['epochs']
    hd = params['hd']
    lr = params['lr']
    take_random = params['take_rand']
    enc = 'bipolar'
    early: bool = params['early']

    #path generator: determined on day of execution
    path = make_path(enc)
    #save config in txt
    config_json = pd.read_json(json_path, typ='series')
    config_json.to_csv(path+'config.txt')

    data, histogram = main_bipolar(gnn_type, layers, num_reps, epochs, hd, lr, n_max, take_random, path)
        
'''