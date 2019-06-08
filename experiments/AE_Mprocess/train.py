import os
import time
import torch
import argparse
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

from models import AE



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_sizes', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument("--latent_size", type=int, default=2)
parser.add_argument('--num_processes', type=int, default=6)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--print_every", type=int, default=1000)

args = parser.parse_args()

sides_3 = np.load('../data/force_torque_sensor/Dataset/3_sides/Data/data.npy')

device = torch.device('cuda:0')


def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std

def un_normalize(normalized_data, input_data):
    mu = np.mean(input_data,axis=0)
    std = np.std(input_data,axis=0)
    return normalized_data*std+mu

def loss_fn(recon_x, x):
    # criterion = nn.BCEWithLogitsLoss(reduction='sum')
    criterion = nn.MSELoss()
    loss = criterion(recon_x, x)
    return loss

def train(args):
    
    # ae = torch.load('ae_june5.pkl').float()
    ae = AE(latent_size=args.latent_size).float()
    # print(ae)
    train_data = feature_normalize(sides_3)
    # test_data =feature_normalize(sides_3)
    
    # number of subprocesses to use for data loading
    num_workers = 4
    # how many samples per batch to load
    batch_size = args.batch_sizes
    # percentage of training set to use as validation
    valid_size = 0.2

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,num_workers=num_workers,sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,sampler=valid_sampler)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    
    # optimizer = torch.optim.Adam(ae.parameters(), lr=args.learning_rate)
    logs = defaultdict(list)
    use_cuda = torch.cuda.is_available
    if use_cuda:
        ae.to(device)
    
    epochs = args.epochs
    for epoch in range(epochs):
        lr_havel = args.learning_rate
        optimizer = torch.optim.Adam(ae.parameters(), lr=lr_havel)
        for iteration, d in enumerate(train_loader):
            if use_cuda:
                input_data = d.to(device).float()
            else:
                input_data = d.float()
            optimizer.zero_grad()
            recon_data = ae(input_data)
            loss = loss_fn(recon_data, input_data)
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(train_loader)-1:
                correct = 0
                total = 0
                for valid in valid_loader:
                    if use_cuda:
                        input_valid = valid.to(device).float()
                    else:
                        input_valid = valid.float()

                    output_valid = ae(input_valid)
                    valid_loss = loss_fn(output_valid, input_valid)

                    total +=1
                    if valid_loss < 0.5:
                        correct += 1

                valid_acc = correct / total  

                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d} |Train_loss {:9.4f} |Valid_accuary{:9.4f} |Valid_loss{:9.4f} |Learning_rate{:9.4f}".format(
                    epoch, epochs, iteration, len(train_loader)-1, loss.item(), valid_acc, valid_loss, lr_havel))
        lr_havel = lr_havel*0.999
        torch.save(ae, 'ae.pkl')  


if __name__ == '__main__':
    latent_size=args.latent_size
    
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    # dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    model = AE(latent_size=latent_size).float()
    print(model)
    model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(args,))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

