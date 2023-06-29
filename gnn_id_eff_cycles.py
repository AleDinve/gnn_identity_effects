
import seaborn as sns
import torch
import torch_geometric
import numpy as np
import pandas as pd
from utils import histogram_dicyclic_saver, plot_saved_data, make_path, save_tensors, prediction
import matplotlib.pyplot as plt
from dataset_gen import *
from train import training
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main_dicyclic(gnn_type, num_layers, num_reps, epochs, hd, lr, n_max, take_random, path, early):
    raw_data = []
    histogram_data = []
    if gnn_type == 'gconv':
        for num_l in num_layers:
            print(f'Number of layers: {num_l}')
            for it in range(num_reps):          
                print(str(it+1)+'-th iteration')
                #seed manual setting
                torch_geometric.seed.seed_everything(it*100)
                data_dict = dataset_generator(dataset='dicyclic', n_max = n_max, take_random=take_random)
                model, raw_data = training(gnn_type, data_dict, 'dicyclic', hd, lr, num_l, epochs, it, raw_data, early = early)
                histogram_data = histogram_dicyclic_saver(histogram_data, data_dict, n_max,  model, 'dicyclic', it, num_l, device)
    
    data = pd.DataFrame.from_records(raw_data)
    data.to_csv(path+'dicyclic_data')
    histogram_pd = pd.DataFrame.from_records(histogram_data)
    histogram_pd.to_csv(path+'dicyclic_histogram')
    return data, histogram_pd


def orthogonality(json_path):
    #loading parameters from json
    with open(json_path) as f:
        params = json.load(f)
    #gnn_type = params['gnn_type']
    num_reps = params['num_reps']
    n_max = params['n_max']
    min_num_layers = params['min_num_layers']
    max_num_layers = params['max_num_layers']
    layers =[i for i in range(min_num_layers, max_num_layers+1)]
    epochs = params['epochs']
    hd = params['hd']
    lr = params['lr']
    k = params['k']
    gap = params['gap']
    experiment_type = params['exp_type']
    early = params['early']
    gnn_types = ['gconv_diff', 'gconv_glob']
    for num_l in layers:
        df_data = []
        print(f'Number of layers: {num_l}')
        for gnn_type in gnn_types:
            if gap>0:
                path = make_path('cycles/'+experiment_type+'/gap', gnn_type)
            else:
                path = make_path('cycles/'+experiment_type+'/extraction', gnn_type)
            config_json = pd.read_json(json_path, typ='series')
            config_json.to_csv(path+'config.txt')
            raw_data = []           
            for it in range(num_reps):    
                sep_vec = True   
                print(str(it+1)+'-th iteration')  

                #seed manual setting
                torch_geometric.seed.seed_everything(it*100+10)
                if gap>0:
                    data_dict = get_complete_dataset_gap(n_max, gap)
                else:
                    data_dict = extraction_task_dataset(n_max, k)
                
                model, raw_data = training(gnn_type, data_dict, 'dicyclic', hd, lr, num_l, epochs, it, raw_data, device=device, path=path, early=early)
                
                data_loader = data_dict['test_loader_nobatch']
                iter_dl = iter(data_loader)

            df_data = prediction(model, hd, n_max, gap, gnn_type, iter_dl, sep_vec, df_data, device, path, num_l, it)
            df = pd.DataFrame.from_records(df_data)
            df.to_csv(path+'trial_nmax_'+str(n_max-2)+'_gap_'+str(gap+2)+'_'+str(num_l)+'_layers.csv')
            plot_saved_data(path = path, file_name = None, online=False, df=df, num_l = num_l)

        data = pd.DataFrame.from_records(raw_data)
        data.to_csv(path+'dicyclic_data')
        sns.lineplot(data=data, x='epoch', y='test_loss', hue='num_layers')
        plt.savefig(path+'test_loss/test_loss_dicyclic.png')
        plt.close()



if __name__ == '__main__':
    #dicyclic_comparison('params_dicyclic.json')
    orthogonality('params_dicyclic.json')










# def dicyclic_comparison(json_path):
#     #loading parameters from json
#     with open(json_path) as f:
#         params = json.load(f)
#     gnn_type = params['gnn_type']
#     num_reps = params['num_reps']
#     n_max = params['n_max']
#     min_num_layers = params['min_num_layers']
#     max_num_layers = params['max_num_layers']
#     layers =[i for i in range(min_num_layers, max_num_layers+1)]
#     epochs = params['epochs']
#     hd = params['hd']
#     lr = params['lr']
#     enc = 'dicyclic'
#     gap = params['gap']
#     early = params['early']
#     k = params['k']

#     #path configuration
#     path = make_path('cycles', gnn_type)
#     #save config in txt
#     config_json = pd.read_json(json_path, typ='series')
#     config_json.to_csv(path+'config.txt')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     if gap>0:
#         data_dict = get_complete_dataset_gap(n_max, gap)
#     else:
#         data_dict = extraction_task_dataset(n_max, k)
#     for num_l in layers:
#         df_data = []
#         raw_data = []
#         print(f'Number of layers: {num_l}')
        
#         for it in range(num_reps):       
#             print(str(it+1)+'-th iteration')     
#             #seed manual setting
#             torch_geometric.seed.seed_everything(it*100)
#             #data_dict = dataset_generator(dataset='dicyclic', n_max = n_max, take_random=take_random)
            
#             model, raw_data = training(gnn_type, data_dict, 'dicyclic', hd, lr, num_l, epochs, it, raw_data, device=device, early=early)
#             data_loader = data_dict['test_loader_nobatch']

#             iter_dl = iter(data_loader)
#             for i in range(3,n_max+gap+1):
#                 for j in range(i,n_max+gap+1):
#                     data = next(iter_dl).to(device)
#                     y,_ = model(data.x, data.edge_index, data.batch)              
#                     df_data.append({'1st_cycle':i,'2nd_cycle':j,'rating': float(y), 'it': it+1, 'num_layers':num_l})
#                     if j!=i:
#                         df_data.append({'1st_cycle':j,'2nd_cycle':i,'rating': float(y), 'it': it+1, 'num_layers':num_l})
#         df = pd.DataFrame.from_records(df_data)
#         df.to_csv(path+'trial_nmax_'+str(n_max-2)+'_gap_'+str(gap+2)+'_'+str(num_l)+'_layers.csv')
#             # sns.relplot(data=df, x = '1st_cycle', y = '2nd_cycle', hue = 'rating')
#             # plt.savefig('prova_comparison_'+str(num_l)+'_layers_'+ enc +'.png')
#             # plt.close()
#         plot_saved_data(path = path, file_name = None, online=False, df=df, num_l = num_l)


# def main(json_path):
#     with open(json_path) as f:
#         params = json.load(f)
#     gnn_type = params['gnn_type']
#     num_reps = params['num_reps']
#     n_max = params['n_max']
#     min_num_layers = params['min_num_layers']
#     max_num_layers = params['max_num_layers']
#     layers =[i for i in range(min_num_layers, max_num_layers+1)]
#     epochs = params['epochs']
#     hd = params['hd']
#     lr = params['lr']
#     take_random = params['take_rand']
#     enc = 'dicyclic'
#     early: bool = params['early']

#     #path generator: determined on day of execution
#     path = make_path('cycles', gnn_type)
#     #save config in txt
#     config_json = pd.read_json(json_path, typ='series')
#     config_json.to_csv(path+'config.txt')

#     data, histogram = main_dicyclic(gnn_type, layers, num_reps, epochs, hd, lr, n_max, take_random, path, early)
        
#     data_enc = data[data['dataset']==enc]
    
        

#     sns.lineplot(data=data_enc, x='epoch', y='test_loss')
#     plt.savefig(path+'test_loss/test_loss_dicyclic.png')
#     plt.close()
#     sns.catplot(data=histogram, kind='bar', x='word', y='rating')
#     plt.savefig(path+'hist/hist_'+ gnn_type +'.png')
#     plt.close()