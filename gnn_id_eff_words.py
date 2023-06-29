
import seaborn as sns
import torch
import torch_geometric
import pandas as pd
from utils import make_path, histogram_data_saver
import matplotlib.pyplot as plt
from dataset_gen import dataset_generator
from train import training

import json


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    


def main_letters(gnn_type, num_layers, num_reps, epochs, hd, lr, encoding_list, distributed_bits, dim_red, path, device, early):
    raw_data = []
    histogram_data = []
    for enc in encoding_list:
        print(enc)
        for num_l in num_layers:
            print(f'Number of layers: {num_l}')
            for it in range(num_reps):
                #seed manual setting
                torch_geometric.seed.seed_everything(it*100)
                data_dict = dataset_generator(dataset=enc, distributed_bits= distributed_bits, dim_red=dim_red)
                model, raw_data = training(gnn_type, data_dict, enc, hd, lr, num_l, epochs, it, raw_data, device, path=path, early=early)
                histogram_data = histogram_data_saver(histogram_data, data_dict, model, enc, it, num_l, device)

    data = pd.DataFrame.from_records(raw_data)
    data.to_csv(path+'data')
    histogram_pd = pd.DataFrame.from_records(histogram_data)
    histogram_pd.to_csv(path+'histogram')
    return data, histogram_pd



def main(json_path):
    # Call parameters from json file
    with open(json_path) as f:
        params = json.load(f)
    num_reps = params['num_reps']
    gnn_type = params['gnn_type']
    min_num_layers = params['min_num_layers']
    max_num_layers = params['max_num_layers']
    layers =[i for i in range(min_num_layers, max_num_layers+1)]
    epochs = params['epochs']
    hd = params['hd']
    lr = params['lr']
    experiment_type = params['exp_type']
    distributed_bits =params['distributed_bits']
    dim_red = params['dim_red']
    early: bool = params['early']

    #path generator: determined on day of execution
    path = make_path('words/'+experiment_type, gnn_type)
    #save config in txt
    config_json = pd.read_json(json_path, typ='series')
    config_json.to_csv(path+'config.txt')


    encoding_list = ['one-hot', 'haar', 'distributed', 'gaussian']


    data, histogram = main_letters(gnn_type, layers, num_reps, epochs, hd, lr, encoding_list, distributed_bits, dim_red, path, device, early)
    for l in layers:
        data_l = data[data['num_layers']==l]
        histogram_l = histogram[histogram['num_layers']==l]
        #sns.lineplot(data=data_l, x='epoch', y='test_loss')
        sns.lineplot(data=data_l, x='epoch', y='test_loss', hue='dataset')
        plt.savefig(path+'test_loss/test_loss_'+str(l)+'_layers.png')
        plt.close()

        #sns.catplot(data=histogram_l, kind='bar', x='word', y='rating')
        sns.catplot(data=histogram_l, kind='bar', x='word', y='rating', hue='dataset')
        plt.savefig(path+'hist/hist_'+str(l)+'_layers.png')
        plt.close()
            

if __name__=='__main__':

    main('params_letters.json')


    
    
## param config for experiments on words experiments:
# "min_num_layers": 1,
# "max_num_layers": 3,
# "epochs": 3000,
# "num_reps": 40,
# "lr":0.0025,
# "hd":32