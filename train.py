
import torch

import torch.nn.functional as F
from utils import EarlyStopping, plot_saved_data, tensorboard_writer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from models import GCONV, GIN, GCONVDIFF, GCONVSTACK, GCONVSTACK_WORDS, GCONVDIFF_WORDS
import os

import json


def training(gnn_type, data_dict, dataset, hd, lr, num_l, epochs, it, 
             raw_data, device, path, tensorboard_session = False, early = False):
    
    #loading train and test loaders from data dictionary
    train_loader = data_dict['train_loader']
    test_loader = data_dict['test_loader']
        
    if dataset == 'dicyclic':
        input = 1
    else:
        input = (data_dict['xy'].x).size()[1]
    if gnn_type == 'gconv_glob':
        model = GCONV(input, hd, 1, num_l).to(device)
    elif gnn_type == 'gconv_diff':
        model = GCONVDIFF(input, hd, 1, num_l).to(device)
    elif gnn_type == 'gconv_stack':
        model = GCONVSTACK(input, hd, 1, num_l).to(device)
    elif gnn_type == 'gconv_stack_words':
        model = GCONVSTACK_WORDS(input, hd, 1, num_l).to(device)
    elif gnn_type == 'gconv_diff_words':
        model = GCONVDIFF_WORDS(input, hd, 1, num_l).to(device)
    elif gnn_type == 'gin':
        model = GIN(input, hd, 1, num_l).to(device)
        lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    @torch.enable_grad()
    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred, _ = model(data.x, data.edge_index, data.batch)
            loss = F.binary_cross_entropy(pred, data.y)
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
            pred, _ = model(data.x, data.edge_index, data.batch)            
            loss = F.binary_cross_entropy(pred, data.y)
            pred = pred > 0.5
            total_correct += int((pred == data.y).sum())
        return total_correct * 100 / len(loader.dataset), loss.item()


    if tensorboard_session:
        writer = SummaryWriter()
    if early:
        early_stopping = EarlyStopping(min_delta=3, patience = 120)
        print('Early Stopping activated')
    
    if not os.path.exists(path+'model_state/'):
        os.makedirs(path+'model_state/')
    #training starts here
    for epoch in range(1, epochs + 1):
        
        train()         
        train_acc, train_loss= test(train_loader)     
        test_acc, test_loss= test(test_loader) 

        raw_data.append({'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc,
                            'test_loss': test_loss,  'epoch': epoch, 'it': it,'num_layers':num_l, 'dataset':dataset})


        if train_loss == 0:
            print(f"Stopped on Epoch {epoch} with train accuracy {train_acc}% and test accuracy {test_acc}%")
            return model, raw_data

        
        if (epoch+1)%200==0:
            print(f'Epoch: {epoch+1}')
            print(f'Train accuracy: {train_acc}, train loss: {train_loss}')
            print(f'Test accuracy: {test_acc}, test loss: {test_loss}')
        
        
        if tensorboard_session:
            tensorboard_writer(writer, train_loss, train_acc, test_loss, test_acc, epoch)

        
        if early:
            stop_training: bool = early_stopping.on_epoch_end(epoch, test_acc)
            if stop_training:
                print(f"Stopped on Epoch {epoch} with train accuracy {train_acc}% and test accuracy {test_acc}%")
                return model, raw_data
        if  (epoch+1)%500==0: 
            torch.save(model.state_dict(), path+'model_state/'+'epoch'+str(epoch+1)+'_model_state_'+dataset+'.pt')
    
    print(f"Stopped on Epoch {epoch} with train accuracy {train_acc}% and test accuracy {test_acc}%")
    return model, raw_data