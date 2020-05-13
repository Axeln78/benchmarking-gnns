"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl
import random

from train.metrics import accuracy_SBM as accuracy
from train.augmentations import rm_rand_edges


def train_epoch(model, optimizer, device, data_loader, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        #edge removal
        #print('before',batch_graphs.number_of_edges())
        batch_graphs = rm_rand_edges(batch_graphs, random.uniform(0,0.4))
        #print('after',batch_graphs.number_of_edges())
        #rest of the pipe
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc


