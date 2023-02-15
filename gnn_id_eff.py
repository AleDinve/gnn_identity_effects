import numpy as np
import torch_geometric as tg
import seaborn as sns
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_add_pool, GraphConv
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from dataset_gen import dataset_generator


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nc):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(nc):
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr='add', bias=True))
            in_channels = hidden_channels

        # TODO: No dropout.
        self.mlp = MLP([hidden_channels, hidden_channels, out_channels])

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)

        return torch.sigmoid(self.mlp(x))




def training(data_dict, dataset, hd, lr, num_l, epochs, it, raw_data):
    

    
    train_loader = data_dict['train_loader']
    test_loader = data_dict['test_loader']
    xy = data_dict['xy']
        
        
        


    model = Net(26, hd, 1, num_l).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    @torch.enable_grad()
    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.binary_cross_entropy(out, data.y)
            
            #loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_correct = 0
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch) 
            
            loss = F.binary_cross_entropy(pred, data.y)

            pred = pred > 0.5
            total_correct += int((pred == data.y).sum())
        return total_correct * 100 / len(loader.dataset), loss.item()

    for epoch in range(1, epochs + 1):
        print('Epoch: '+str(epoch))
        loss = train()          
        train_acc, train_loss= test(train_loader)       
        test_acc, test_loss = test(test_loader) 
        raw_data.append({'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc,
                            'test_loss': test_loss,  'epoch': epoch, 'it': it,'num_layers':num_l, 'dataset':dataset})
    
    return model, raw_data
    
    

def main(num_layers, num_reps, epochs, hd, lr):
    colors = ["darkorange", "royalblue", "darkorchid"]
    raw_data = []
    histogram_data = []
    datasets = ['one-hot','haar']
    for dataset in datasets:
        for num_l in num_layers:
            print(f'Number of layers: {num_l}')
            for it in range(num_reps):
                #seed manual setting
                torch_geometric.seed.seed_everything(it*100)
                data_dict = dataset_generator(dataset=dataset)
                model, raw_data = training(data_dict, dataset, hd, lr, num_l, epochs, it, raw_data)
                histogram_data = histogram_data_saver(histogram_data, data_dict, model, dataset, it, num_l)

    data = pd.DataFrame.from_records(raw_data)
    data.to_csv('trial_14_feb')
    histogram_pd = pd.DataFrame.from_records(histogram_data)
    histogram_pd.to_csv('histogram_data_14_feb')
    return data, histogram_pd

def histogram_data_saver(histogram_data, data_dict, model, dataset, it, num_layers):
    words_list = ['AA', 'xy', 'YY', 'ZZ','YZ','ZT','EY','SZ']
    for word in words_list:
        data = data_dict[word].to(device)
        histogram_data.append({'word': word, 'pred': float(model(data.x, data.edge_index, data.batch)), 'it': it,
                                'num_layers': num_layers, 'dataset': dataset})
    return histogram_data

if __name__=='__main__':
    layers =[1,2,3]
    num_reps = 40
    epochs = 5000
    hd = 16
    lr = 0.0025
    data, histogram = main(layers, num_reps, epochs, hd, lr)
    for l in layers:
        data_l = data[data['num_layers']==l]
        histogram_l = histogram[histogram['num_layers']==l]
        

        sns.lineplot(data=data_l, x='epoch', y='test_loss', hue='dataset')
        plt.savefig('test_loss_'+str(l)+'_layers.png')
        plt.close()

        sns.catplot(data=histogram_l, kind='bar', x='word', y='pred', hue='dataset')
        plt.savefig('hist_'+str(l)+'_layers.png')
        plt.close()
    