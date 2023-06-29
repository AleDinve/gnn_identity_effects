
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from encodings_gen import get_encodings_list
from string import ascii_uppercase as letters
import networkx as nx
import random

def cycle_graph(n):
    '''
    Return the adjacency matrix of a cycle graph of length n
    '''
    A = np.zeros((n,n))
    A[0,-1]=1
    A[-1,0]=1
    for i in range(n-1):
        A[i,i+1]=1
        A[i+1,i]=1
    return A

def cycle_data(n, asymmetric = False):
    A = cycle_graph(n)
    s = np.shape(A)[0]
    G = nx.from_numpy_matrix(A)
    g = from_networkx(G)
    init_colors = [1 for i in range(s)]
    if asymmetric:
        init_colors[0] = 0
    g.x = torch.tensor(init_colors, dtype=torch.long)
    return g, init_colors


def dicyclic_graph(m,n):
    '''
    Return the adjacency matrix of a dyciclic graph composed by two cycles of length m,n
    '''
    A_1 = cycle_graph(m)
    A_2 = cycle_graph(n)
    A = np.zeros((m+n,m+n))
    A[:m,:m] = A_1
    A[m:,m:] = A_2
    A[m,m-1] = 1
    A[m-1,m] = 1
    return A

def dicyclic_data(m,n):
    A = dicyclic_graph(m,n)
    s = np.shape(A)[0]
    G = nx.from_numpy_matrix(A)
    g = from_networkx(G)
    g.x = torch.tensor([[1] for i in range(s)], dtype=torch.float)
    if m==n:
        g.y = torch.tensor([[1]],dtype=torch.float)
    else:
        g.y = torch.tensor([[0]],dtype=torch.float)
    return g


def get_dicyclic_dataset(n_max=12, W0_size=48, take_random = True):
    data_dict = {}
    W0_train = []
    W1_train = []
    
    ### Training set generation
    # Creating dataset of matched-dicyclic molecules
    for n in range(3,n_max-1):
        g = dicyclic_data(n,n)
        #balancing the dataset
        for _ in range(6):
            W1_train.append(g)
    # Creating dataset of mismatched-dicyclic molecules
    if take_random:
        count = 0
        while count < W0_size:
            train_indices_list = []
            train_indices = np.random.randint(3,n_max-1,size=2)
            if train_indices[0] != train_indices[1]:
                if train_indices_list.count(train_indices) == 0:
                    train_indices_list.append(train_indices)
                    count += 1
                    g = dicyclic_data(train_indices[0],train_indices[1])
                    W0_train.append(g)
    else:
        for ti in range(3,n_max-1):
            for tj in range(ti+1,n_max-1):
                g = dicyclic_data(ti,tj)
                W0_train.append(g)
                
    train_set = W0_train+W1_train
    
    ### Test set generation

    W0_test = []
    W1_test = []
    # Selecting good graphs for test set
    test_set_good_ind = [3, n_max-2, n_max-1, n_max]
    # Selecting bad graphs for test set
    test_set_bad_ind = [(n_max-1,n_max),(n_max,n_max-4),(4,n_max-1),(6,n_max)]
    # 3-cycle matching graph
    for ind in test_set_good_ind:
        g = dicyclic_data(ind,ind)

        W1_test.append(g)
        data_dict[str(ind)+'_'+str(ind)] = DataLoader([g])
    for ind in test_set_bad_ind:
        g = dicyclic_data(ind[0],ind[1])
        W0_test.append(g)
        data_dict[str(ind[0])+'_'+str(ind[1])] = DataLoader([g])
        

    test_set = W1_test + W0_test
    return data_dict, train_set, test_set


def all_dicyclic(n_max, gap, start = 3):
    dataset_list = []
    for i in range(start,n_max+gap+1):
        for j in range(start,n_max+gap+1):
            dataset_list.append(dicyclic_data(i,j))
    random.shuffle(dataset_list)
    dataloader = DataLoader(dataset_list, n_max-2, shuffle=False)
    return dataloader

def dicyclic_gap_val(n_max, gap, start = 3):
    dataset_list = []
    for i in range(start,n_max+gap+1):
        for j in range(gap):
            dataset_list.append(dicyclic_data(i,n_max+j+1))
            dataset_list.append(dicyclic_data(n_max+j+1,i))
    random.shuffle(dataset_list)
    dataloader = DataLoader(dataset_list, 2*n_max, shuffle=False)
    return dataloader

def all_dicyclic_test(n_max, gap, start=3):
    dataset_list = []
    for i in range(start, n_max+gap+1):
        for j in range(start, n_max+gap+1):
            dataset_list.append(dicyclic_data(i,j))
    dataloader = DataLoader(dataset_list, shuffle=False)
    return dataloader

def not_all_dicyclic(n_max, k):
    dataset_list = []
    batch_size = (n_max-3)**2
    for i in range(3,n_max+1):
        for j in range(3,n_max+1):
            if i!=k and j!=k:
                dataset_list.append(dicyclic_data(i,j))
    random.shuffle(dataset_list)
    dataloader = DataLoader(dataset_list, batch_size = batch_size, shuffle=False)
    return dataloader

def get_complete_dataset_gap(n_max, gap):
    data_dict = {}  
    data_dict['train_loader'] = all_dicyclic(n_max, 0)
    data_dict['test_loader'] = dicyclic_gap_val(n_max, gap)
    data_dict['test_loader_nobatch'] = all_dicyclic_test(n_max, gap)
    return data_dict

def extraction_task_dataset(n_max, k):
    data_dict = {}  
    data_dict['train_loader'] = not_all_dicyclic(n_max, k)
    data_dict['test_loader'] = all_dicyclic(n_max, 0)
    data_dict['test_loader_nobatch'] = all_dicyclic_test(n_max, 0)
    return data_dict

        




def dataset_generator(dataset, n_max = 12, take_random=True, distributed_bits=3, dim_red=16, train_batch_size = 72, test_batch_size = 8, train_size=48):
    # define universe of possible input values
    data_dict = {} 
    if dataset == 'dicyclic':
        W0_size = 48

        # n_max-1, n_max cycles are considered for test
        data_dict, train_set, test_set = get_dicyclic_dataset(n_max, W0_size, take_random)
        train_batch_size = len(train_set)
        random.shuffle(train_set)
        random.shuffle(test_set)
        train_loader = DataLoader(train_set, train_batch_size, shuffle=False)
        train_loader_nobatch = DataLoader(train_set, shuffle=False)
        test_loader = DataLoader(test_set, 1, shuffle=False)
        data_dict['train_loader_nobatch'] = train_loader_nobatch

    else:

        encodings = get_encodings_list(j=distributed_bits,dim = dim_red)[dataset]

        W0 = []
        W1 = []

        ## Train generator
        # Grammatically-correct words

        edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
        for k in range(24):
            x_correct = torch.tensor([encodings[k], encodings[k]], dtype=torch.float)
            y_correct = torch.tensor([[1]],dtype=torch.float)
            data_correct = Data(edge_index=edge_index, x=x_correct,  y=y_correct)
            W1.append(data_correct)

        # Grammatically-uncorrect words
        count = 0
        while count < train_size:
            train_indices = np.random.randint(0,24,size=2)
            if train_indices[0] != train_indices[1]:
                x_uncorrect = torch.tensor([encodings[train_indices[0]], encodings[train_indices[1]]], dtype=torch.float)
                y_uncorrect = torch.tensor([[0]],dtype=torch.float)
                data_uncorrect = Data(edge_index=edge_index, x=x_uncorrect,  y=y_uncorrect)
                count+=1       
                W0.append(data_uncorrect)

        ## Test generator
        # Draw xy from W0 and append it to test_set
        test_set = []
        xy_index = np.random.randint(0,train_size-1)
        xy = W0[xy_index]
        data_dict['xy'] = xy
        test_set.append(xy)
        # Append AA, YY, ZZ to test_set
        test_correct_index = [0, 24, 25]
        test_correct_words = ['AA', 'YY', 'ZZ']
        
        count = 0
        for i in test_correct_index:
            x_correct = torch.tensor([encodings[i], encodings[i]], dtype=torch.float)
            y_correct = torch.tensor([[1]],dtype=torch.float)
            data_correct = Data(edge_index=edge_index, x=x_correct,  y=y_correct)
            data_dict[test_correct_words[count]] = data_correct
            count +=1
            test_set.append(data_correct)
        # Append YZ, ZY, EY, SZ to test_set
        test_uncorrect_index = [[24,25], [25,24], [4,24], [18,25]]
        test_uncorrect_words = ['YZ','ZT','EY','SZ']

        count = 0
        for indices in test_uncorrect_index:
            x_uncorrect = torch.tensor([encodings[indices[0]], encodings[indices[1]]], dtype=torch.float)
            y_uncorrect=torch.tensor([[0]],dtype=torch.float)
            data_uncorrect = Data(edge_index=edge_index, x=x_uncorrect,  y=y_uncorrect)
            data_dict[test_uncorrect_words[count]] = data_uncorrect
            count +=1
            test_set.append(data_uncorrect)
    
        train_set = W0+W1
        random.shuffle(train_set)
        random.shuffle(test_set)
        train_loader = DataLoader(train_set, train_batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_set, test_batch_size, shuffle=False, drop_last=True)

    data_dict['train_loader'] = train_loader
    data_dict['test_loader'] = test_loader


    return data_dict
  

if __name__ == '__main__':
    data_dict = get_complete_dataset_gap(8,4)
    train_loader = data_dict['train_loader']
    for data in train_loader:
        print(data)



