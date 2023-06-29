import numpy as np
import torch
from torch.utils.data import DataLoader
from encodings_gen import get_encodings_list
import random



def create_vector_dataset(enc, distributed_bits=3, dim_red = 16, train_batch_size = 72, test_batch_size = 8, train_size=48):
    data_dict = {}
    W0 = []
    W1 = []
    encodings = get_encodings_list(distributed_bits, dim_red)[enc]
    ## Train generator
    # Grammatically-correct words

    for k in range(24):
        x_correct = torch.cat([torch.tensor(encodings[k], dtype=torch.float),
                               torch.tensor(encodings[k], dtype=torch.float)])
        y_correct = torch.tensor([1],dtype=torch.float)
        data_correct = [x_correct, y_correct]
        W1.append(data_correct)

    # Grammatically-uncorrect words
    count = 0
    while count < train_size:
        train_indices = np.random.randint(0,24,size=2)
        if train_indices[0] != train_indices[1]:
            x_uncorrect = torch.cat([torch.tensor(encodings[train_indices[0]], dtype=torch.float),
                                     torch.tensor(encodings[train_indices[1]], dtype=torch.float)])
            y_uncorrect = torch.tensor([0],dtype=torch.float)
            data_uncorrect = [x_uncorrect, y_uncorrect]
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
        x_correct = torch.cat([torch.tensor(encodings[i], dtype=torch.float), 
                               torch.tensor(encodings[i], dtype=torch.float)])
        y_correct = torch.tensor([1],dtype=torch.float)
        data_correct = [x_correct, y_correct]
        data_dict[test_correct_words[count]] = data_correct
        count +=1
        test_set.append(data_correct)

    # Append YZ, ZY, EY, SZ to test_set
    test_uncorrect_index = [[24,25], [25,24], [4,24], [18,25]]
    test_uncorrect_words = ['YZ','ZT','EY','SZ']

    count = 0
    for indices in test_uncorrect_index:
        x_uncorrect = torch.cat([torch.tensor(encodings[indices[0]], dtype=torch.float), 
                                 torch.tensor(encodings[indices[1]], dtype=torch.float)])
        y_uncorrect=torch.tensor([0],dtype=torch.float)
        data_uncorrect = [x_uncorrect, y_uncorrect]
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
    a = create_vector_dataset('one-hot')
    print(a['xy'])
