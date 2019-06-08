import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import idx2onehot


class AE(nn.Module):

    def __init__(self, latent_size):

        super().__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        
        x = self.encoder(x)
        recon_x = self.decoder(x)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()


        self.fc1 = nn.Linear(12,8)
        self.fc2 = nn.Linear(8, latent_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()

        self.fc3 = nn.Linear(latent_size,8)
        self.fc4 = nn.Linear(8,12)


    def forward(self, z):

        x = F.relu(self.fc4(z))
        # x = torch.sigmoid(self.fc4(x))

        return x
