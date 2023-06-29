'''@author: Giuseppe Alessio D'Inverno'''
'''@date: 02/03/2023'''

import torch
from torch_geometric.nn import WLConv
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import numpy as np
import networkx as nx




## WL class to find colors

def dataset_generator(card_set, num_nodes, batch_size=1):
    dataset = []
    for _ in range(card_set):
        A = np.random.randint(0,50,(num_nodes, num_nodes))
        A = (A > 20).astype(int)
        G = nx.from_numpy_matrix(A)
        g = from_networkx(G)
        g.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
        dataset.append(g)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)
    return data_loader


class WL(torch.nn.Module):
    def __init__(self,  num_it):
        super().__init__()
        self.num_it = num_it
        self.conv = WLConv()

    def forward(self, x, edge_index):
        for _ in range(self.num_it):
            x = self.conv(x, edge_index)
        
        return x
    
    def reset_parameters(self):
        return self.conv.reset_parameters()

def wl_colors(G,num_it):
    model = WL(num_it)
    model.eval()
    pred = model(G.x, G.edge_index)
    if torch.min(pred[torch.nonzero(pred)])!=1:
        
        pred = pred-torch.min(pred[torch.nonzero(pred)])+1
        pred[pred<0] = 0
    pred_original = pred
    pred, _, count = torch.unique(pred, return_inverse= True, return_counts = True)
    pred = torch.stack([pred,count])
    return pred, pred_original

def main():
    card_set = 50
    num_nodes = 10
    data_loader = dataset_generator(card_set, num_nodes, batch_size=1)
    batchdata_list = []
    for data in data_loader:
        batchdata_list.append(data)
    print(wl_colors(batchdata_list[0], 10))
    print(wl_colors(batchdata_list[1], 10))


if __name__ == '__main__':
    main()
