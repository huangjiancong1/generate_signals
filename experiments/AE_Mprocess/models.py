import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device('cuda:0')


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

############################### 12,784,1000,500,250  Hinton's reduction paper #########

# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,784)
#         self.fc2 = nn.Linear(784,1000)
#         self.fc3 = nn.Linear(1000,500)
#         self.fc4 = nn.Linear(500,250)
#         self.fc5 = nn.Linear(250, latent_size)

#         self.fc_bn1 = nn.BatchNorm1d(784)
#         self.fc_bn2 = nn.BatchNorm1d(1000)
#         self.fc_bn3 = nn.BatchNorm1d(500)
#         self.fc_bn4 = nn.BatchNorm1d(250)

#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         x = F.relu(self.fc3(x))
#         x = self.fc_bn3(x)
#         x = F.relu(self.fc4(x))
#         x = self.fc_bn4(x)
#         x = F.relu(self.fc5(x))
#         return x


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc6 = nn.Linear(latent_size,250)
#         self.fc7 = nn.Linear(250,500)
#         self.fc8 = nn.Linear(500,1000)
#         self.fc9 = nn.Linear(1000,784)
#         self.fc10 = nn.Linear(784,12)

#         self.fc_bn5 = nn.BatchNorm1d(250)
#         self.fc_bn6 = nn.BatchNorm1d(500)
#         self.fc_bn7 = nn.BatchNorm1d(1000)
#         self.fc_bn8 = nn.BatchNorm1d(784)


#     def forward(self, x):

#         x = F.relu(self.fc6(x))
#         x = self.fc_bn5(x)
#         x = F.relu(self.fc7(x))
#         x = self.fc_bn6(x)
#         x = F.relu(self.fc8(x))
#         x = self.fc_bn7(x)
#         x = F.relu(self.fc9(x))
#         x = self.fc_bn8(x)
#         x = F.relu(self.fc10(x))

#         return x

############################### 12,32,64,8 with batch normalizetion #########
class Encoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()


        self.fc1 = nn.Linear(12,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,8)
        self.fc4 = nn.Linear(8, latent_size)
        
        self.fc_bn1 = nn.BatchNorm1d(32)
        self.fc_bn2 = nn.BatchNorm1d(64)
        self.fc_bn3 = nn.BatchNorm1d(8)
        

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc_bn1(x)
        x = F.relu(self.fc2(x))
        x = self.fc_bn2(x)
        x = F.relu(self.fc3(x))
        x = self.fc_bn3(x)
        x = F.relu(self.fc4(x))
        return x


class Decoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()

        self.fc4 = nn.Linear(latent_size,8)
        self.fc5 = nn.Linear(8,64)
        self.fc6 = nn.Linear(64,32)
        self.fc7 = nn.Linear(32,12)
        
        self.fc_bn4 = nn.BatchNorm1d(8)
        self.fc_bn5 = nn.BatchNorm1d(64)
        self.fc_bn6 = nn.BatchNorm1d(32)


    def forward(self, x):

        x = F.relu(self.fc4(x))
        x = self.fc_bn4(x)
        x = F.relu(self.fc5(x))
        x = self.fc_bn5(x)
        x = F.relu(self.fc6(x))
        x = self.fc_bn6(x)
        x = F.relu(self.fc7(x))

        return x

# ############################### 12,32,64,8 with batch normalizetion Dropout #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,32)
#         self.fc2 = nn.Linear(32,64)
#         self.fc3 = nn.Linear(64,8)
#         self.fc4 = nn.Linear(8, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(32)
#         self.fc_bn2 = nn.BatchNorm1d(64)
#         self.fc_bn3 = nn.BatchNorm1d(8)
        
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc3(x))
#         x = self.fc_bn3(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc4(x))
#         return x


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc4 = nn.Linear(latent_size,8)
#         self.fc5 = nn.Linear(8,64)
#         self.fc6 = nn.Linear(64,32)
#         self.fc7 = nn.Linear(32,12)
        
#         self.fc_bn4 = nn.BatchNorm1d(8)
#         self.fc_bn5 = nn.BatchNorm1d(64)
#         self.fc_bn6 = nn.BatchNorm1d(32)

#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):

#         x = F.relu(self.fc4(x))
#         x = self.fc_bn4(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc5(x))
#         x = self.fc_bn5(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc6(x))
#         x = self.fc_bn6(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc7(x))

#         return x


############################### 12,32,8 no batch normalizetion #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,32)
#         self.fc2 = nn.Linear(32,8)
#         self.fc3 = nn.Linear(8, latent_size)

#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return x


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc4 = nn.Linear(latent_size,8)
#         self.fc5 = nn.Linear(8,32)
#         self.fc6 = nn.Linear(32,12)


#     def forward(self, x):

#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = F.relu(self.fc6(x))

#         return x
    

############################### 12,32,8 with batch normalizetion #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,32)
#         self.fc2 = nn.Linear(32,8)
#         self.fc3 = nn.Linear(8, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(32)
#         self.fc_bn2 = nn.BatchNorm1d(8)

#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         x = F.relu(self.fc3(x))
#         return x


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc4 = nn.Linear(latent_size,8)
#         self.fc5 = nn.Linear(8,32)
#         self.fc6 = nn.Linear(32,12)
        
#         self.fc_bn3 = nn.BatchNorm1d(8)
#         self.fc_bn4 = nn.BatchNorm1d(32)


#     def forward(self, x):

#         x = F.relu(self.fc4(x))
#         x = self.fc_bn3(x)
#         x = F.relu(self.fc5(x))
#         x = self.fc_bn4(x)
#         x = F.relu(self.fc6(x))

#         return x
