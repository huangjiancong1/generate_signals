import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import idx2onehot


class VAE(nn.Module):

    def __init__(self, latent_size):

        super().__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        
        batch_size = x.size(0)

        means, log_var = self.encoder(x)

        std = torch.exp(0.5 * log_var).cuda()
        eps = torch.randn([batch_size, self.latent_size]).cuda()
        mul = torch.mul(eps, std).cuda()
        z = torch.add(mul,means).cuda()

        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).cuda()

        recon_x = self.decoder(z)

        return recon_x


# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,8).cuda()

#         self.linear_means = nn.Linear(8, latent_size)
#         self.linear_log_var = nn.Linear(8, latent_size)

#     def forward(self, x):

#         x = F.relu(self.fc1(x))
        
#         means = self.linear_means(x)
#         log_vars = self.linear_log_var(x)

#         return means, log_vars


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc2 = nn.Linear(latent_size,8)
#         self.fc3 = nn.Linear(8,12)


#     def forward(self, z):

#         x = F.relu(self.fc2(z)).cuda()
#         x = torch.sigmoid(self.fc3(x))

#         return x

class Encoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()


        self.fc1 = nn.Linear(12,32)
        self.fc2 = nn.Linear(32,8)

        self.linear_means = nn.Linear(8, latent_size)
        self.linear_log_var = nn.Linear(8, latent_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()

        self.fc3 = nn.Linear(latent_size,8)
        self.fc4 = nn.Linear(8,32)
        self.fc5 = nn.Linear(32,12)


    def forward(self, z):

        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))

        return z
