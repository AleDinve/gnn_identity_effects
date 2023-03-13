
import torch

import torch.nn.functional as F
from utils import EarlyStopping, plot_saved_data
import matplotlib.pyplot as plt
from dataset_gen import dataset_generator, all_bipolar, get_complete_dataset
from torch.utils.tensorboard import SummaryWriter
from models import GCONV, GIN

import json


def training(gnn_type, data_dict, dataset, hd, lr, num_l, epochs, it, raw_data, device, tensorboard_session = False, early = True):
    


    train_loader = data_dict['train_loader']
    test_loader = data_dict['test_loader']
    
        
        
    if dataset == 'bipolar':
        input = 1
        global_pool = False
    else:
        input = 26
        global_pool = True
    if gnn_type == 'gconv':
        model = GCONV(input, hd, 1, num_l, global_pool).to(device)
    elif gnn_type == 'gin':
        model = GIN(input, hd, 1, num_l).to(device)
        lr = 0.01
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
    if tensorboard_session:
        writer = SummaryWriter()
    if early:
        early_stopping = EarlyStopping(min_delta=3, patience = 120)
        print('Early Stopping activated')
    for epoch in range(1, epochs + 1):
        
        loss = train()         
        train_acc, train_loss= test(train_loader)       
        test_acc, test_loss = test(test_loader) 
        if (epoch+1)%100==0:
            print(f'Epoch: {epoch}')
            print(f'Train accuracy: {train_acc}, train loss: {train_loss}')
            print(f'Test accuracy: {test_acc}, test loss: {test_loss}')
        if early:
            stop_training: bool = early_stopping.on_epoch_end(epoch, test_acc)
            if stop_training:
                print(f"Stopped on Epoch {epoch} with train accuracy {train_acc}% and test accuracy {test_acc}%")
                return model, raw_data
        if tensorboard_session:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
        raw_data.append({'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc,
                            'test_loss': test_loss,  'epoch': epoch, 'it': it,'num_layers':num_l, 'dataset':dataset})
    print(f"Stopped on Epoch {epoch} with train accuracy {train_acc}% and test accuracy {test_acc}%")
    return model, raw_data