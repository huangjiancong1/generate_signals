import torch
import torch.nn as nn
import torch.nn.functional as F





class AE(nn.Module):

    def __init__(self, latent_size):

        super().__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        
        x = self.encoder(x)
#         x = x.unsqueeze_(2)
        x = self.decoder(x)

        return x


class Encoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()


        self.fc1 = nn.Linear(12,8).cuda()
        self.fc2 = nn.Linear(8, latent_size)
        
        self.batch_norm1 = nn.BatchNorm1d(8)
#         self.batch_norm2 = nn.BatchNorm1d(latent_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = F.relu(self.fc2(x))
        return x


class Decoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()

        self.fc3 = nn.Linear(latent_size,8)
        self.fc4 = nn.Linear(8,12)

        self.batch_norm3 = nn.BatchNorm1d(8)
#         self.batch_norm4 = nn.BatchNorm1d(12)

        
    def forward(self, x):

        x = F.relu(self.fc3(x)).cuda()
        x = self.batch_norm3(x)
        x = F.relu(self.fc4(x))

        return x
