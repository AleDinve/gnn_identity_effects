import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import MLP, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn import GraphConv, GINConv, WLConv
import torch.nn.functional as F



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


class GCONV(torch.nn.Module):
    """GCONV"""
    def __init__(self, in_channels, hidden_channels, out_channels, nc, global_pool = False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.global_pool = global_pool
        self.convs = torch.nn.ModuleList()
        for _ in range(nc):
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr='add', bias=True))
            in_channels = hidden_channels
        mlp_input = hidden_channels if global_pool else 2*hidden_channels
        self.mlp = MLP([mlp_input, 4*hidden_channels, out_channels])

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))

        if self.global_pool:
            #x = torch.reshape(x, (-1, 2*self.hidden_channels))
            x = global_add_pool(x, batch)
        else:
            a,_,c = torch.unique(edge_index[0], return_inverse= True, return_counts = True)
            x = x[a[c>2]]
            batch = batch[a[c>2]]
            x = torch.reshape(x, (-1, 2*self.hidden_channels))
        #import pdb; pdb.set_trace()
        
        #
        #x = global_mean_pool(x, batch)
        #x = global_max_pool(x, batch)
        return torch.sigmoid(self.mlp(x))

class GIN_old(torch.nn.Module):
    """GIN(old version)"""
    def __init__(self, input_dim, dim_h, output_dim):
        super(GIN_old, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(input_dim, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, output_dim)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return F.sigmoid(h)

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, in_channels, hidden_channels, out_channels, nc):
        super(GIN, self).__init__()
        self.hidden_channels = hidden_channels
        self.nc = nc
        self.convs = torch.nn.ModuleList()
        for _ in range(self.nc):
            self.convs.append(GINConv(Sequential(Linear(in_channels, hidden_channels),
                       BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU())))
            in_channels = hidden_channels
        self.lin = Linear(nc*hidden_channels,out_channels)


    def forward(self, x, edge_index, batch):
        h = []
        for conv in self.convs:
            x = conv(x, edge_index)
            h.append(global_add_pool(x, batch))

        #x = torch.reshape(x, (-1, 2*self.hidden_channels))
        h_pool = h[0]
        for ind in range(1,self.nc):
            h_pool = torch.cat((h_pool,h[ind]),dim=1)
        return torch.sigmoid(self.lin(h_pool))
        

        
        # else:
        #     a,_,c = torch.unique(edge_index[0], return_inverse= True, return_counts = True)
        #     x = x[a[c>2]]
        #     batch = batch[a[c>2]]
        #     x = torch.reshape(x, (-1, 2*self.hidden_channels))
