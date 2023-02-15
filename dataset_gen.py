##Alphabet one-hot encoding
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from numpy import argmax
from scipy.stats import ortho_group


def dataset_generator(dataset, train_batch_size = 72, test_batch_size = 8, train_size=48):
    # define universe of possible input values
    data_dict = {}    
    encodings = []

    if dataset == 'one-hot':
        for value in range(26):
            letter = [0 for _ in range(26)]
            letter[value] = 1
            encodings.append(letter)
    elif dataset == 'haar':
        A = ortho_group.rvs(26)
        for i in range(26):
            encodings.append(A[i,:])

    W0 = []
    W1 = []

    ## Train generator
    # Grammatically-correct words

    edge_index = torch.tensor([[0],[1]], dtype=torch.long)
    for k in range(24):
        x_correct = torch.tensor([encodings[k], encodings[k]], dtype=torch.float)
        y_correct = torch.tensor([[1]],dtype=torch.float)
        data_correct = Data(edge_index=edge_index, x=x_correct,  y=y_correct)
        W1.append(data_correct)

    # Grammatically-uncorrect words
    count = 0
    while count < train_size:
        train_indices = np.random.randint(0,25,size=2)
        if train_indices[0] != train_indices[1]:
            x_uncorrect = torch.tensor([encodings[train_indices[0]], encodings[train_indices[1]]], dtype=torch.float)
            y_uncorrect = torch.tensor([[0]],dtype=torch.float)
            data_uncorrect = Data(edge_index=edge_index, x=x_uncorrect,  y=y_uncorrect)
            count+=1       
            W0.append(data_uncorrect)
 
    train_set = W0+W1
    train_loader = DataLoader(train_set, train_batch_size, shuffle=True)


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
    
    test_loader = DataLoader(test_set, test_batch_size, shuffle=True)

    data_dict['train_loader'] = train_loader
    data_dict['test_loader'] = test_loader


    return data_dict