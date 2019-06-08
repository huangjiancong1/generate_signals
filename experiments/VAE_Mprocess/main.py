# from __future__ import print_function
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.multiprocessing as mp

# # from train import train, test

# # Training settings
# parser = argparse.ArgumentParser()
# parser.add_argument('--num_workers', type=int, default=4)
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument("--latent_size", type=int, default=2)
# parser.add_argument('--num_processes', type=int, default=2)
# parser.add_argument("--learning_rate", type=float, default=0.001)
# parser.add_argument("--print_every", type=int, default=100)

# args = parser.parse_args()

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# if __name__ == '__main__':
#     args = parser.parse_args()

#     use_cuda = args.cuda and torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

#     torch.manual_seed(args.seed)
#     mp.set_start_method('spawn')

#     model = Net().to(device)
#     model.share_memory() # gradients are allocated lazily, so they are not shared here

#     processes = []
#     for rank in range(args.num_processes):
#         p = mp.Process(target=train, args=(rank, args, model, device, dataloader_kwargs))
#         # We first train the model across `num_processes` processes
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()

#     # Once training is complete, we can test the model
#     test(args, model, device, dataloader_kwargs)